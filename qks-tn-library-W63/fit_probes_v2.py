"""Encoder probe v2: strong CNN probes with augmentation, three stems feeding one (8, 8, C) trunk.

Arms (all share the CNN trunk of ``fit_probes.py``; only the stem from the 32x32x3 [0,1] image differs):

  a  frozen trained encoder: checkpoint PatchQKSEncodingLayer kernel+bias (frozen) -> [cos, sin], C = 6
     (one job per tree seed 0, 1, 2; restore and ordering checks as in v1)
  b  identity stem: space-to-depth 4 -> (8, 8, 48)                        (probe seeds 0, 1)
  c  learned stem: trainable 4x4 stride-4 conv with k filters + bias, initialised like the patch
     encoder, -> [cos, sin], C = 2k, k in {3, 14}, trained end to end     (probe seeds 0, 1)

Protocol (identical for every arm): random crop from the 4-pixel zero-padded image + random horizontal
flip on probe-train only (applied to the image, before the stem); AdamW lr 1e-3, wd 5e-4, cosine decay
to 0 over 100 epochs, batch 256, no early stopping, final-epoch accuracy; same 45k / 5k / test split as v1.

Smoke test::

    MODAL_ENVIRONMENT=env-ai-algs python3 -m modal run -e env-ai-algs qks-tn-library/fit_probes_v2.py --limit 2000 --epochs 2 --smoke-arms a0,b0,c3_0,c14_0

Full run (9 jobs)::

    MODAL_ENVIRONMENT=env-ai-algs python3 -m modal run -e env-ai-algs qks-tn-library/fit_probes_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("qks-tn-fit-probes-v2")

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .pip_install("tensorflow-addons==0.23.0", "tensornetwork==0.4.6", "scipy", "numpy<2",
                 "datasets", "huggingface_hub", "pillow", "pyarrow")
    .add_local_dir(os.path.join(HERE, "common_functions"), remote_path="/root/qks/common_functions")
    .add_local_file(os.path.join(HERE, "run_ff_experiment.py"), remote_path="/root/qks/run_ff_experiment.py")
)

results_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)

SWEEP = "chi8-cifar"
RUN = "cifar10_B_pervertex_patch4s4_g1_morton_chi8_s{seed}"
CKPT_TOL = 0.002
SPLIT_SEED = 12345  # same fixed split as fit_probes.py
N_VAL = 5000
LR, WD, BATCH = 1e-3, 5e-4, 256

# job name -> (arm, tree_seed, k, probe_seed)
FULL_JOBS = {
    "a_tree0": ("a", 0, None, 0), "a_tree1": ("a", 1, None, 1), "a_tree2": ("a", 2, None, 2),
    "b_ps0": ("b", None, None, 0), "b_ps1": ("b", None, None, 1),
    "c_k3_ps0": ("c", None, 3, 0), "c_k3_ps1": ("c", None, 3, 1),
    "c_k14_ps0": ("c", None, 14, 0), "c_k14_ps1": ("c", None, 14, 1),
}
SMOKE_JOBS = ["a_tree0", "b_ps0", "c_k3_ps0", "c_k14_ps0"]


def _acc(probs, y):
    import numpy as np

    return float(np.mean(np.argmax(probs, -1) == y))


def _predict(model, Xu8, bs=1024):
    import numpy as np

    return np.concatenate([model.predict_on_batch(Xu8[i : i + bs].astype("float32").reshape(-1, 32, 32, 3) / 255.0) for i in range(0, len(Xu8), bs)])


# --------------------------------------------------------------------------------------
# stems and trunk
# --------------------------------------------------------------------------------------
def _make_layers():
    import tensorflow as tf

    class CosSin(tf.keras.layers.Layer):
        def call(self, theta):
            return tf.concat([tf.cos(theta), tf.sin(theta)], axis=-1)

    class FrozenConv(tf.keras.layers.Layer):
        """Fixed 4x4 stride-4 VALID conv + bias (the checkpoint's patch-encoder angles, raster order)."""

        def __init__(self, kernel, bias, stride, **kw):
            super().__init__(**kw)
            self.k = tf.constant(kernel, dtype=tf.float32)
            self.b = tf.constant(bias, dtype=tf.float32)
            self.stride = int(stride)

        def call(self, x):
            return tf.nn.conv2d(x, self.k, strides=self.stride, padding="VALID") + self.b

    class S2D(tf.keras.layers.Layer):
        def call(self, x):
            return tf.nn.space_to_depth(x, 4)

    return CosSin, FrozenConv, S2D


def _trunk_layers():
    import tensorflow as tf
    L = tf.keras.layers

    def block(f):
        return [L.Conv2D(f, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU()]

    return block(128) + block(128) + [L.MaxPool2D(2)] + block(256) + [L.GlobalAveragePooling2D(), L.Dense(10, activation="softmax")]


def _stem_layers(arm, k, enc, sigma):
    """List of Keras layers mapping (B, 32, 32, 3) in [0,1] to the (B, 8, 8, C) grid."""
    import numpy as np
    import tensorflow as tf
    CosSin, FrozenConv, S2D = _make_layers()
    if arm == "a":
        return [FrozenConv(enc["kernel"], enc["bias"], 4, name="frozen_enc"), CosSin(name="cos_sin")]
    if arm == "b":
        return [S2D(name="s2d")]
    conv = tf.keras.layers.Conv2D(
        k, 4, strides=4, padding="valid", use_bias=True, name="learned_stem",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=sigma),
        bias_initializer=tf.keras.initializers.RandomUniform(0.0, 2 * np.pi),
    )
    return [conv, CosSin(name="cos_sin")]


def _build(arm, k, enc, sigma, total_steps):
    import tensorflow as tf

    stem = _stem_layers(arm, k, enc, sigma)
    inp = tf.keras.Input((32, 32, 3))
    x = inp
    for l in stem:
        x = l(x)
    stem_model = tf.keras.Model(inp, x, name="stem")
    out = x
    for l in _trunk_layers():
        out = l(out)
    model = tf.keras.Model(inp, out)
    sched = tf.keras.optimizers.schedules.CosineDecay(LR, total_steps, alpha=0.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=sched, weight_decay=WD),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, stem_model


def _aug_dataset(Xu8, y, seed):
    """Per-sample random crop of the 4-pixel zero-padded image + horizontal flip; shuffled each epoch."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((Xu8.reshape(-1, 32, 32, 3), y.astype("int32")))
    ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)

    def aug(x, t):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])  # zero pad (black) in [0, 1] units
        x = tf.image.random_crop(x, [32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, t

    return ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def _write(out, out_subdir, name):
    d = f"/out/fit-probes-v2/{out_subdir}" if out_subdir else "/out/fit-probes-v2"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{name}.json", "w") as f:
        json.dump(out, f, indent=1)
    results_vol.commit()


@app.cls(
    image=image, gpu="L40S", cpu=6, memory=16384, timeout=4 * 60 * 60,
    volumes={"/out": results_vol, "/data": data_vol},
)
class Prober:
    @modal.method()
    def run(self, name: str, arm: str, tree_seed: int, k: int, probe_seed: int, limit: int, epochs: int, out_subdir: str) -> dict:
        import time

        import numpy as np
        import tensorflow as tf

        sys.path.insert(0, "/root/qks")
        sys.path.insert(0, "/root/qks/common_functions")
        import qks_tn_lenet as ln
        import run_ff_experiment as rfe

        t0 = time.time()
        log = lambda s: print(f"[{name}] {s}", flush=True)  # noqa: E731
        results_vol.reload()
        data_vol.reload()
        tf.get_logger().setLevel("ERROR")
        log(f"gpus={tf.config.list_physical_devices('GPU')}")
        out = dict(name=name, arm=arm, tree_seed=tree_seed, k=k, probe_seed=probe_seed, limit=limit, epochs=epochs)

        X_train, y_train_f, X_test, y_test_f = rfe.load_data("cifar10", "/data", log=log)
        if limit:
            X_train, y_train_f = X_train[:limit], y_train_f[:limit]
        y_train, y_test = y_train_f.astype(int), y_test_f.astype(int)
        n = len(y_train)
        order = np.random.RandomState(SPLIT_SEED).permutation(n)
        n_val = N_VAL if n > 10 * N_VAL else max(n // 10, 1)
        iv, ia = order[:n_val], order[n_val:]
        out["split"] = dict(seed=SPLIT_SEED, n_probe_train=int(len(ia)), n_probe_val=int(len(iv)), n_test=int(len(y_test)))

        enc = None
        sigma = None
        if arm == "a":
            # ---- restore the tree checkpoint exactly as in v1 and take the encoder
            run_dir = f"/out/{SWEEP}/{RUN.format(seed=tree_seed)}"
            cfg = json.load(open(f"{run_dir}/ckpt/cfg.json"))
            hist = json.load(open(f"{run_dir}/ckpt/cycle-015/history.json"))["epochs"]
            hist_val_acc = float(hist[-1]["val_acc"])
            assert len(hist) == 15
            cfg["cache_dir"] = "/data"
            bs = int(cfg["batch_size"])
            mdl, qks, tn_layer, counts = rfe.build_model(cfg, 50000 // bs, p=3072, n_classes=10)
            head = mdl.layers[-1]
            assert isinstance(qks, ln.PatchQKSEncodingLayer) and list(mdl.layers) == [qks, tn_layer, head]
            w0 = float(sum(float(tf.reduce_sum(tf.abs(v))) for v in mdl.trainable_variables))
            tf.train.Checkpoint(model=mdl, optimizer=mdl.optimizer).restore(f"{run_dir}/ckpt/cycle-015/ckpt").expect_partial()
            w1 = float(sum(float(tf.reduce_sum(tf.abs(v))) for v in mdl.trainable_variables))
            assert w0 != w1, "restore did not change any weight"
            test_ds = rfe.make_dataset(X_test, y_test_f, bs, shuffle=False, seed=0)
            _, test_acc = mdl.evaluate(test_ds, verbose=0)
            trust = dict(test_acc=float(test_acc), history_val_acc=hist_val_acc, abs_diff=abs(float(test_acc) - hist_val_acc), tol=CKPT_TOL)
            trust["passed"] = trust["abs_diff"] <= CKPT_TOL
            out["trust"] = trust
            log(f"trust check {trust}")
            if not trust["passed"]:
                out["error"] = "trust check failed"
                _write(out, out_subdir, name)
                return out
            assert qks.groups == 1 and qks.qubits == 3 and tuple(qks.grid) == (8, 8) and qks.share_positions and qks.pad == 0
            enc = dict(kernel=qks.kernel.numpy(), bias=qks.bias.numpy())
            # ---- v1 features by the v1 route: angles() in leaf order -> undo perm -> raster grid -> [cos, sin]
            inv_perm = tf.constant(np.argsort(qks.perm.numpy()), dtype=tf.int64)

            def v1_feats(Xu8):
                outl, rt = [], 0.0
                for i in range(0, len(Xu8), 256):
                    xb = tf.constant(Xu8[i : i + 256].astype("float32") / 255.0)
                    a_leaf = qks.angles(xb)
                    th = tf.gather(a_leaf, inv_perm, axis=1)
                    rt = max(rt, float(tf.reduce_max(tf.abs(tf.gather(th, qks.perm, axis=1) - a_leaf))))
                    g = tf.reshape(th, [-1, 8, 8, 3])
                    outl.append(tf.concat([tf.cos(g), tf.sin(g)], -1).numpy())
                return np.concatenate(outl), rt

            chk_sets = {"train": X_train[:2000], "test": X_test[:2000]}
            sm_in = tf.keras.Input((32, 32, 3))
            _sx = sm_in
            for _l in _stem_layers("a", None, enc, None):
                _sx = _l(_sx)
            _stem_only = tf.keras.Model(sm_in, _sx)
            checks = {}
            for nm, Xs in chk_sets.items():
                f1, rt = v1_feats(Xs)
                f2 = _predict(_stem_only, Xs)
                assert f2.shape == f1.shape == (len(Xs), 8, 8, 6), (f1.shape, f2.shape)
                checks[f"{nm}_stem_vs_v1_max_abs"] = float(np.max(np.abs(f1 - f2)))
                checks[f"{nm}_perm_roundtrip_max_abs"] = rt
            out["stem_checks"] = checks
            log(f"stem checks {checks}")
            assert all(v < 1e-5 for v in checks.values()), checks
            del mdl, qks, tn_layer, head
        elif arm == "c":
            sigma = 0.125 * float(np.sqrt(784.0 / (4 * 4 * 3))) * 1.0  # same dimensional rule as build_model, sigma_scale 1
            out["init_sigma"] = sigma
        else:
            Z = np.random.RandomState(0).randint(0, 256, (8, 3072)).astype("uint8")
            ref = (Z.astype("float32").reshape(-1, 8, 4, 8, 4, 3) / 255.0).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 8, 8, 48)
            got = tf.nn.space_to_depth(tf.constant(Z.astype("float32").reshape(-1, 32, 32, 3) / 255.0), 4).numpy()
            out["stem_checks"] = dict(s2d_vs_v1_max_abs=float(np.max(np.abs(ref - got))))
            assert out["stem_checks"]["s2d_vs_v1_max_abs"] == 0.0

        # ---- train
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(2000 + probe_seed)
        steps = len(ia) // BATCH
        model, stem_model = _build(arm, k, enc, sigma, epochs * steps)
        grid_shape = tuple(stem_model.output_shape[1:])
        out["stem_output_shape"] = list(grid_shape)
        out["n_params_total"] = int(model.count_params())
        out["n_params_trainable"] = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
        log(f"stem output {grid_shape}, params total {out['n_params_total']} trainable {out['n_params_trainable']}, steps/epoch {steps}")
        Xa, ya = X_train[ia], y_train[ia]
        Xv, yv = X_train[iv], y_train[iv]
        val_ds = tf.data.Dataset.from_tensor_slices((Xv.astype("float32").reshape(-1, 32, 32, 3) / 255.0, yv.astype("int32"))).batch(1024)

        class Prog(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
                    log(f"epoch {epoch + 1}/{epochs} loss {logs['loss']:.4f} aug-acc {logs['accuracy']:.4f} val_acc {logs['val_accuracy']:.4f} ({time.time() - t0:.0f}s)")

        h = model.fit(_aug_dataset(Xa, ya, 2000 + probe_seed), validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[Prog()])
        res = dict(train=_acc(_predict(model, Xa), ya), val=_acc(_predict(model, Xv), yv), test=_acc(_predict(model, X_test), y_test))
        out["result"] = res
        out["history"] = {k_: [float(v) for v in vs] for k_, vs in h.history.items()}
        if arm == "c":
            w = model.get_layer("learned_stem").get_weights()
            out["learned_stem_stats"] = dict(kernel_std=float(w[0].std()), bias_min=float(w[1].min()), bias_max=float(w[1].max()))
        log(f"final {res}")
        out["wall_s"] = time.time() - t0
        _write(out, out_subdir, name)
        return out


# --------------------------------------------------------------------------------------
def _sem(v):
    import statistics

    return statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def _summary(rows):
    import statistics

    rows = sorted(rows, key=lambda r: r["name"])
    ok = [r for r in rows if "result" in r]
    lines = ["# Encoder probe v2 (augmented CNN probes, 100-epoch cosine AdamW)", "",
             "## Per job (train = unaugmented probe-train 45k, val 5k, test 10k; final-epoch weights, no restore-best, no early stopping)", "",
             "Note: the val column is not held out for arm (a): the tree, and therefore its encoder, trained on all 50k training images, "
             "including these 5k. For arms (b) and (c) val is held out from probe training.", "",
             "| job | C | params (trainable) | train | val | test |", "|---|---|---|---|---|---|"]
    for r in ok:
        q = r["result"]
        lines.append(f"| {r['name']} | {r['stem_output_shape'][-1]} | {r['n_params_trainable']} | {q['train']:.4f} | {q['val']:.4f} | {q['test']:.4f} |")
    groups = [("a", "a. frozen trained encoder (tree seeds 0-2)", lambda r: r["arm"] == "a"),
              ("b", "b. identity stem / calibration (probe seeds 0-1)", lambda r: r["arm"] == "b"),
              ("c3", "c. learned stem k=3", lambda r: r["arm"] == "c" and r["k"] == 3),
              ("c14", "c. learned stem k=14", lambda r: r["arm"] == "c" and r["k"] == 14)]
    lines += ["", "## Arm means ± s.e.m.", "", "| arm | n | train | val | test |", "|---|---|---|---|---|"]
    means = {}
    for key, label, f in groups:
        g = [r for r in ok if f(r)]
        if not g:
            continue
        cells = []
        for sp in ("train", "val", "test"):
            v = [r["result"][sp] for r in g]
            cells.append(f"{statistics.mean(v):.4f} ± {_sem(v):.4f}")
        means[key] = [r["result"]["test"] for r in g]
        lines.append(f"| {label} | {len(g)} | " + " | ".join(cells) + " |")
    lines += ["", "## Decision quantities (test accuracy)", ""]
    mean = lambda v: statistics.mean(v)  # noqa: E731
    if "b" in means:
        mb = mean(means["b"])
        lines.append(f"1. Calibration (b) mean test acc = {mb:.4f}" + ("  -- BELOW 0.70: the probe is still too weak to decide." if mb < 0.70 else "  (>= 0.70)"))
    if "a" in means:
        ma = mean(means["a"])
        s = f"2. (a) mean test acc = {ma:.4f} ± {_sem(means['a']):.4f}; vs 0.65: {100 * (ma - 0.65):+.2f} pp"
        if "b" in means:
            s += f"; vs (b): {100 * (ma - mean(means['b'])):+.2f} pp"
        lines.append(s)
    for key in ("c3", "c14"):
        if key in means:
            m = mean(means[key])
            s = f"3. (c) {key} mean test acc = {m:.4f}" + (f" ± {_sem(means[key]):.4f}" if len(means[key]) > 1 else "")
            if "b" in means:
                s += f"; vs (b): {100 * (m - mean(means['b'])):+.2f} pp"
            lines.append(s)
    if "c3" in means and "c14" in means:
        lines.append(f"   (c) k=3 minus k=14: {100 * (mean(means['c3']) - mean(means['c14'])):+.2f} pp")
    lines += ["", "## Stem checks", ""]
    for r in ok:
        if "stem_checks" in r:
            lines.append(f"- {r['name']}: {json.dumps(r['stem_checks'])}" + (f"; trust {json.dumps(r['trust'])}" if "trust" in r else ""))
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(limit: int = 0, epochs: int = 100, smoke_arms: str = ""):
    """``limit`` > 0 = smoke test on the first ``limit`` training images (results go to a ``smoke`` subfolder).
    ``smoke_arms`` = comma-separated job names (default: one job per arm)."""
    sub = "smoke" if limit else ""
    names = [s for s in smoke_arms.split(",") if s] or (SMOKE_JOBS if limit else list(FULL_JOBS))
    handles = []
    for nm in names:
        arm, ts, k, ps = FULL_JOBS[nm]
        handles.append((nm, Prober().run.spawn(nm, arm, ts if ts is not None else -1, k if k is not None else 0, ps, int(limit), int(epochs), sub)))
    out_dir = Path(HERE) / "results" / "fit-probes-v2" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nm, h in handles:
        try:
            r = h.get()
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {nm}: {exc}")
            continue
        (out_dir / f"{nm}.json").write_text(json.dumps(r, indent=1))
        rows.append(r)
    (out_dir / "summary.md").write_text(_summary(rows))
    print(_summary(rows))
    print(f"wrote {out_dir}")
