"""Fit probes for the chi=8 CIFAR-10 L4 model: does the encoder or the fixed readout limit the fit?

Per seed (one L40S job each), on the checkpoints ``chi8-cifar/<run>/ckpt/cycle-015/ckpt`` of the
``qks-tn-results`` Volume:

A. trust check: rebuild the model with ``run_ff_experiment.build_model``, restore the checkpoint and
   require the test accuracy to match the history's last ``val_acc`` within 0.002;
B. feature extraction (inference mode, 50k train + 10k test): encoder angles in raster position
   order -> (8, 8, 6) features [cos, sin], and the 64 root probabilities;
C. probes: encoder CNN, raw-pixel calibration CNN, linear and MLP readout probes, and the model's own
   fixed readout, all on one fixed 45k / 5k split of the training set.

Smoke test::

    MODAL_ENVIRONMENT=env-ai-algs python3 -m modal run -e env-ai-algs qks-tn-library/fit_probes.py --limit 2000 --probe-epochs 2 --seeds 0

Full run::

    MODAL_ENVIRONMENT=env-ai-algs python3 -m modal run -e env-ai-algs qks-tn-library/fit_probes.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("qks-tn-fit-probes")

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
SPLIT_SEED = 12345  # fixed probe-train / probe-val split, identical for every probe and seed
N_VAL = 5000


@app.function(image=image, cpu=4, memory=16384, timeout=60 * 60, volumes={"/data": data_vol})
def prepare_data() -> dict:
    """Cache CIFAR-10 on the data Volume once (before the seed jobs start, to avoid write races)."""
    sys.path.insert(0, "/root/qks")
    import run_ff_experiment as rfe

    data_vol.reload()
    Xtr, ytr, Xte, yte = rfe.load_data("cifar10", "/data", log=lambda s: print(s, flush=True))
    data_vol.commit()
    return dict(n_train=int(len(ytr)), n_test=int(len(yte)), dtype=str(Xtr.dtype))


# --------------------------------------------------------------------------------------
# helpers (run inside the container)
# --------------------------------------------------------------------------------------
def _batched(fn, X, bs):
    import numpy as np

    return np.concatenate([fn(X[i : i + bs]) for i in range(0, len(X), bs)], axis=0)


def _acc(logits, y):
    import numpy as np

    return float(np.mean(np.argmax(logits, -1) == y))


def _fit_probe(make_model, data, epochs, seed, lr=1e-3, batch=256, patience=8):
    """Train one probe: Adam, early stopping on probe-val accuracy, restore best. Returns metrics."""
    import numpy as np
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    (Xa, ya), (Xv, yv), (Xt, yt) = data
    m = make_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True)
    h = m.fit(Xa, ya, validation_data=(Xv, yv), batch_size=batch, epochs=epochs, callbacks=[cb], verbose=0)
    pr = lambda X: m.predict(X, batch_size=2048, verbose=0)  # noqa: E731
    res = dict(
        train=_acc(pr(Xa), ya), val=_acc(pr(Xv), yv), test=_acc(pr(Xt), yt),
        epochs_run=len(h.history["loss"]),
        best_epoch=int(np.argmax(h.history["val_accuracy"])) + 1,
        n_params=int(m.count_params()),
    )
    del m
    return res


def _cnn(in_shape):
    import tensorflow as tf
    L = tf.keras.layers

    def block(f):
        return [L.Conv2D(f, 3, padding="same", use_bias=False), L.BatchNormalization(), L.ReLU()]

    return tf.keras.Sequential(
        [L.InputLayer(input_shape=in_shape)] + block(128) + block(128) + [L.MaxPool2D(2)] + block(256)
        + [L.GlobalAveragePooling2D(), L.Dense(10, activation="softmax")]
    )


def _linear(d, l2):
    import tensorflow as tf
    L = tf.keras.layers
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    return tf.keras.Sequential([L.InputLayer(input_shape=(d,)), L.Dense(10, activation="softmax", kernel_regularizer=reg)])


def _mlp(d):
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential(
        [L.InputLayer(input_shape=(d,)), L.Dense(128, activation="relu"), L.Dense(256, activation="relu"),
         L.Dense(256, activation="relu"), L.Dense(10, activation="softmax")]
    )


@app.cls(
    image=image, gpu="L40S", cpu=4, memory=16384, timeout=3 * 60 * 60,
    volumes={"/out": results_vol, "/data": data_vol},
)
class Prober:
    @modal.method()
    def run(self, seed: int, limit: int, probe_epochs: int, out_subdir: str) -> dict:
        import time

        import numpy as np
        import tensorflow as tf

        sys.path.insert(0, "/root/qks")
        sys.path.insert(0, "/root/qks/common_functions")
        import qks_tn_lenet as ln
        import run_ff_experiment as rfe

        t0 = time.time()
        log = lambda s: print(f"[seed {seed}] {s}", flush=True)  # noqa: E731
        results_vol.reload()
        data_vol.reload()
        tf.get_logger().setLevel("ERROR")
        gpus = tf.config.list_physical_devices("GPU")
        log(f"gpus={gpus}")

        run_dir = f"/out/{SWEEP}/{RUN.format(seed=seed)}"
        ckpt_prefix = f"{run_dir}/ckpt/cycle-015/ckpt"
        cfg = json.load(open(f"{run_dir}/ckpt/cfg.json"))
        hist = json.load(open(f"{run_dir}/ckpt/cycle-015/history.json"))["epochs"]
        hist_val_acc = float(hist[-1]["val_acc"])
        assert len(hist) == 15, len(hist)
        cfg["cache_dir"] = "/data"
        out = dict(seed=seed, run=RUN.format(seed=seed), limit=limit, probe_epochs=probe_epochs,
                   history_last_val_acc=hist_val_acc, cfg=cfg)

        # ------------------------------------------------------------------ A. trust check
        X_train, y_train_f, X_test, y_test_f = rfe.load_data("cifar10", "/data", log=log)
        if limit:
            X_train, y_train_f = X_train[:limit], y_train_f[:limit]
        y_train, y_test = y_train_f.astype(int), y_test_f.astype(int)
        bs = int(cfg["batch_size"])
        steps_per_epoch = 50000 // bs  # the training run used the full 50k (limit_train=None)
        assert cfg.get("limit_train") in (None, 0)
        p = int(X_train.shape[1])
        model, qks, tn_layer, counts = rfe.build_model(cfg, steps_per_epoch, p=p, n_classes=rfe.N_CLASSES["cifar10"])
        head = model.layers[-1]
        assert isinstance(qks, ln.PatchQKSEncodingLayer) and list(model.layers) == [qks, tn_layer, head], model.layers
        w_before = float(sum(float(tf.reduce_sum(tf.abs(v))) for v in model.trainable_variables))
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        ckpt.restore(ckpt_prefix).expect_partial()
        w_after = float(sum(float(tf.reduce_sum(tf.abs(v))) for v in model.trainable_variables))
        log(f"params={counts['total']} sum|w| before restore {w_before:.4f} after {w_after:.4f}")
        assert w_before != w_after, "restore did not change any weight"

        train_ds = rfe.make_dataset(X_train, y_train_f, bs, shuffle=False, seed=0)
        test_ds = rfe.make_dataset(X_test, y_test_f, bs, shuffle=False, seed=0)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        train_loss, train_acc_full = model.evaluate(train_ds, verbose=0)
        trust = dict(test_acc=float(test_acc), history_val_acc=hist_val_acc, abs_diff=abs(float(test_acc) - hist_val_acc),
                     train_acc_full=float(train_acc_full), n_train=int(len(y_train)), test_loss=float(test_loss),
                     train_loss=float(train_loss), tol=CKPT_TOL)
        trust["passed"] = trust["abs_diff"] <= CKPT_TOL
        out["trust"] = trust
        log(f"trust check: {trust}")
        if not trust["passed"]:
            out["error"] = "trust check failed; probes not run"
            _write(out, out_subdir, seed)
            return out

        # ------------------------------------------------------------------ B. features
        perm = qks.perm.numpy()
        inv_perm = tf.constant(np.argsort(perm), dtype=tf.int64)
        gh, gw = qks.grid
        assert qks.groups == 1 and qks.qubits == 3 and (gh, gw) == (8, 8), (qks.groups, qks.qubits, qks.grid)
        M = head.const

        checks = dict(perm_roundtrip_max_abs=0.0, conv_vs_angles_max_abs=0.0, probs_min=1e9, probs_sum_max_dev=0.0,
                      readout_vs_model_max_abs=0.0, readout_vs_model_argmax_mismatch=0)

        @tf.function
        def extract(xb):
            a_leaf = qks.angles(xb)  # (B, 64, 3) leaf-slot order: a_leaf[:, l] = theta[:, perm[l]]
            theta = tf.gather(a_leaf, inv_perm, axis=1)  # back to register / raster order r = y*8 + x (groups = 1)
            roundtrip = tf.reduce_max(tf.abs(tf.gather(theta, qks.perm, axis=1) - a_leaf))
            x4 = tf.reshape(xb, [-1, 32, 32, 3])
            conv = tf.nn.conv2d(x4, qks.kernel, strides=qks.stride, padding="VALID") + qks.bias  # (B, 8, 8, 3), raster
            grid = tf.reshape(theta, [-1, gh, gw, 3])
            conv_diff = tf.reduce_max(tf.abs(conv - grid))
            probs = tn_layer(qks(xb))
            logits_manual = tf.linalg.matvec(M, probs)
            logits_model = model(xb, training=False)
            return grid, probs, logits_manual, logits_model, roundtrip, conv_diff

        def extract_all(X):
            feats, probs_l, lm_l = [], [], []
            for i in range(0, len(X), 256):
                xb = tf.constant(X[i : i + 256].astype("float32") / 255.0)
                grid, probs, lman, lmod, rt, cd = (t.numpy() for t in extract(xb))
                feats.append(np.concatenate([np.cos(grid), np.sin(grid)], axis=-1).astype("float32"))
                probs_l.append(probs)
                lm_l.append(lman)
                checks["perm_roundtrip_max_abs"] = max(checks["perm_roundtrip_max_abs"], float(rt))
                checks["conv_vs_angles_max_abs"] = max(checks["conv_vs_angles_max_abs"], float(cd))
                checks["readout_vs_model_max_abs"] = max(checks["readout_vs_model_max_abs"], float(np.max(np.abs(lman - lmod))))
                checks["readout_vs_model_argmax_mismatch"] += int(np.sum(np.argmax(lman, -1) != np.argmax(lmod, -1)))
            probs = np.concatenate(probs_l)
            checks["probs_min"] = min(checks["probs_min"], float(probs.min()))
            checks["probs_sum_max_dev"] = max(checks["probs_sum_max_dev"], float(np.max(np.abs(probs.sum(-1) - 1.0))))
            return np.concatenate(feats), probs.astype("float64"), np.concatenate(lm_l)

        Ftr, Ptr, Ltr = extract_all(X_train)
        Fte, Pte, Lte = extract_all(X_test)
        out["feature_checks"] = checks
        log(f"feature shapes {Ftr.shape} {Ptr.shape}; checks {checks}")
        assert checks["perm_roundtrip_max_abs"] == 0.0, "perm round trip not exact"
        assert checks["conv_vs_angles_max_abs"] < 1e-3, checks
        assert checks["probs_min"] >= -1e-6 and checks["probs_sum_max_dev"] < 1e-4, checks
        assert checks["readout_vs_model_argmax_mismatch"] == 0 and checks["readout_vs_model_max_abs"] < 1e-5, checks
        fixed_test_acc_ext = _acc(Lte, y_test)
        out["feature_checks"]["fixed_readout_test_acc_from_probs"] = fixed_test_acc_ext
        out["feature_checks"]["fixed_readout_train_acc_from_probs"] = _acc(Ltr, y_train)
        assert abs(fixed_test_acc_ext - trust["test_acc"]) < 1e-6

        # raw-pixel grid at the same resolution: non-overlapping 4x4x3 patches flattened -> (8, 8, 48)
        def s2d(X):
            Z = X.astype("float32").reshape(-1, 8, 4, 8, 4, 3) / 255.0  # (N, gy, py, gx, px, c)
            return np.ascontiguousarray(Z.transpose(0, 1, 3, 2, 4, 5).reshape(-1, 8, 8, 48))

        Rtr, Rte = s2d(X_train), s2d(X_test)

        # ------------------------------------------------------------------ C. probes
        n = len(y_train)
        order = np.random.RandomState(SPLIT_SEED).permutation(n)
        n_val = N_VAL if n > 10 * N_VAL else max(n // 10, 1)
        iv, ia = order[:n_val], order[n_val:]
        out["split"] = dict(seed=SPLIT_SEED, n_probe_train=int(len(ia)), n_probe_val=int(len(iv)), n_test=int(len(y_test)))

        def split(Xtr_, Xte_):
            return (Xtr_[ia], y_train[ia]), (Xtr_[iv], y_train[iv]), (Xte_, y_test)

        probes = {}

        # 5. fixed readout reference (the model's own readout on the same split)
        probes["fixed_readout"] = dict(train=_acc(Ltr[ia], y_train[ia]), val=_acc(Ltr[iv], y_train[iv]), test=_acc(Lte, y_test))
        log(f"fixed readout {probes['fixed_readout']}")

        pseed = 1000 + seed
        # 1. encoder probe
        probes["encoder_cnn"] = _fit_probe(lambda: _cnn((8, 8, 6)), split(Ftr, Fte), probe_epochs, pseed)
        log(f"encoder_cnn {probes['encoder_cnn']}")
        # 2. calibration: same CNN on raw-pixel space-to-depth grid
        probes["pixel_cnn"] = _fit_probe(lambda: _cnn((8, 8, 48)), split(Rtr, Rte), probe_epochs, pseed)
        log(f"pixel_cnn {probes['pixel_cnn']}")

        # 3./4. readout probes on [p, log(p + 1e-6)], standardised with probe-train statistics
        def rfeat(P):
            return np.concatenate([P, np.log(P + 1e-6)], axis=-1)

        Gtr, Gte = rfeat(Ptr), rfeat(Pte)
        mu, sd = Gtr[ia].mean(0), Gtr[ia].std(0) + 1e-8
        Gtr, Gte = ((Gtr - mu) / sd).astype("float32"), ((Gte - mu) / sd).astype("float32")
        lin_all = {}
        for l2 in (0.0, 1e-4, 1e-3, 1e-2):
            lin_all[str(l2)] = _fit_probe(lambda l2=l2: _linear(128, l2), split(Gtr, Gte), probe_epochs, pseed)
            log(f"linear l2={l2}: {lin_all[str(l2)]}")
        best_l2 = max(lin_all, key=lambda k: lin_all[k]["val"])  # chosen on probe-val only
        probes["readout_linear"] = dict(lin_all[best_l2], l2=float(best_l2))
        out["linear_l2_sweep"] = lin_all
        probes["readout_mlp"] = _fit_probe(lambda: _mlp(128), split(Gtr, Gte), probe_epochs, pseed)
        log(f"readout_mlp {probes['readout_mlp']}")

        out["probes"] = probes
        best_readout = max(probes["readout_linear"]["val"], probes["readout_mlp"]["val"])
        best_name = "readout_linear" if probes["readout_linear"]["val"] >= probes["readout_mlp"]["val"] else "readout_mlp"
        out["decisions"] = dict(
            encoder_cnn_test=probes["encoder_cnn"]["test"], lenet5_bar=0.65,
            pixel_cnn_test=probes["pixel_cnn"]["test"],
            best_readout_probe_by_val=best_name, best_readout_probe_test=probes[best_name]["test"],
            fixed_readout_test=probes["fixed_readout"]["test"],
            best_readout_minus_fixed_pp=100 * (probes[best_name]["test"] - probes["fixed_readout"]["test"]),
            best_readout_by_test_pp=100 * (max(probes["readout_linear"]["test"], probes["readout_mlp"]["test"]) - probes["fixed_readout"]["test"]),
        )
        out["wall_s"] = time.time() - t0
        _write(out, out_subdir, seed)
        return out


def _write(out, out_subdir, seed):
    d = f"/out/fit-probes/{out_subdir}" if out_subdir else "/out/fit-probes"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/seed{seed}.json", "w") as f:
        json.dump(out, f, indent=1)
    results_vol.commit()


# --------------------------------------------------------------------------------------
def _sem(v):
    import statistics

    return statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def _summary(rows):
    import statistics

    names = [("encoder_cnn", "1. encoder CNN on (8,8,6)"), ("pixel_cnn", "2. calibration CNN on raw (8,8,48)"),
             ("readout_linear", "3. readout linear"), ("readout_mlp", "4. readout MLP"), ("fixed_readout", "5. fixed readout")]
    ok = [r for r in rows if "probes" in r]
    lines = ["# Fit probes, chi=8 CIFAR-10 L4 (cycle-015 checkpoints)", ""]
    lines += ["## Trust check", "", "| seed | test acc | history val_acc | |diff| | full-train acc | passed |", "|---|---|---|---|---|---|"]
    for r in rows:
        t = r["trust"]
        lines.append(f"| {r['seed']} | {t['test_acc']:.4f} | {t['history_val_acc']:.4f} | {t['abs_diff']:.4f} | {t['train_acc_full']:.4f} | {t['passed']} |")
    if not ok:
        return "\n".join(lines) + "\n"
    lines += ["", "## Accuracy (probe-train 45k / probe-val 5k / test 10k)", "",
              "| probe | seed | train | val | test |", "|---|---|---|---|---|"]
    for key, label in names:
        for r in ok:
            q = r["probes"][key]
            lines.append(f"| {label} | {r['seed']} | {q['train']:.4f} | {q['val']:.4f} | {q['test']:.4f} |")
        cells = []
        for sp in ("train", "val", "test"):
            v = [r["probes"][key][sp] for r in ok]
            cells.append(f"{statistics.mean(v):.4f} ± {_sem(v):.4f}")
        lines.append(f"| **{label}** | **mean ± s.e.m. (n={len(ok)})** | " + " | ".join(cells) + " |")
    lines += ["", "## Decision quantities (test accuracy)", "",
              "| seed | encoder CNN | LeNet-5 bar | pixel CNN | best readout probe (by val) | fixed readout | best readout - fixed (pp) |",
              "|---|---|---|---|---|---|---|"]
    for r in ok:
        d = r["decisions"]
        lines.append(f"| {r['seed']} | {d['encoder_cnn_test']:.4f} | 0.65 | {d['pixel_cnn_test']:.4f} | {d['best_readout_probe_test']:.4f} "
                     f"({d['best_readout_probe_by_val']}) | {d['fixed_readout_test']:.4f} | {d['best_readout_minus_fixed_pp']:+.2f} |")
    gaps = [r["decisions"]["best_readout_minus_fixed_pp"] for r in ok]
    enc = [r["decisions"]["encoder_cnn_test"] for r in ok]
    pix = [r["decisions"]["pixel_cnn_test"] for r in ok]
    lines.append(f"| **mean ± s.e.m.** | {statistics.mean(enc):.4f} ± {_sem(enc):.4f} | 0.65 | {statistics.mean(pix):.4f} ± {_sem(pix):.4f} | | | "
                 f"{statistics.mean(gaps):+.2f} ± {_sem(gaps):.2f} |")
    lines += ["", "## Linear readout probe: L2 sweep (val acc, chosen on probe-val)", "", "| seed | " + " | ".join(["0", "1e-4", "1e-3", "1e-2"]) + " | chosen |", "|---|---|---|---|---|---|"]
    for r in ok:
        sw = r["linear_l2_sweep"]
        lines.append(f"| {r['seed']} | " + " | ".join(f"{sw[k]['val']:.4f}" for k in ("0.0", "0.0001", "0.001", "0.01")) + f" | {r['probes']['readout_linear']['l2']} |")
    lines += ["", "## Feature checks", ""]
    for r in ok:
        lines.append(f"- seed {r['seed']}: {json.dumps(r['feature_checks'])}")
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(seeds: str = "0,1,2", limit: int = 0, probe_epochs: int = 60):
    """``limit`` > 0 = smoke test on the first ``limit`` training images (results go to a ``smoke`` subfolder)."""
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    sub = "smoke" if limit else ""
    print(prepare_data.remote())
    handles = [(s, Prober().run.spawn(s, int(limit), int(probe_epochs), sub)) for s in seed_list]
    out_dir = Path(HERE) / "results" / "fit-probes" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, h in handles:
        try:
            r = h.get()
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED seed {s}: {exc}")
            continue
        (out_dir / f"seed{s}.json").write_text(json.dumps(r, indent=1))
        rows.append(r)
    rows.sort(key=lambda r: r["seed"])
    (out_dir / "summary.md").write_text(_summary(rows))
    print(_summary(rows))
    print(f"wrote {out_dir}")
