"""CIFAR-10 classical ladder for CPTP W13, trained remotely on Modal.

Copy-and-extend of run_cnn_baseline.py (W4). Rung 1 is W4's tree CNN. Each later rung adds one
tensor-network constraint; see orch/2026-11-02/W13/RUNGS.md for the exact definitions.

Examples (from repository root):
  python3 -m modal run -e env-ai-algs qks-tn-library/run_ladder.py --phase smoke
  python3 -m modal run --detach -e env-ai-algs qks-tn-library/run_ladder.py --phase fit
"""

from __future__ import annotations

import functools
import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SWEEP = "cptp-v6-w13-ladder"
SPLIT_SEED = 20261102
# W4 selections for the tree CNN (REPORT-W4.md): (lr, wd) by augmentation flag.
LRWD = {1: (0.0003, 0.0), 0: (0.001, 0.0001)}

# rung -> architecture. stem: patch-stem channels; widths: output channels of each tree level
# (3 quad levels or 6 binary levels); topo: quad = 2x2 stride-2 merges (4 children), binary = Morton
# pair merges (x, y, x, y, x, y); merge: conv (conv+BN+ReLU), untied (per-node conv+BN+ReLU),
# relu (pair concat -> linear + norm + ReLU), bilinear (pair -> x^T A_k y, + norm except at a 64-wide root);
# norm: bn BatchNorm (W4), ln LayerNorm over channels per node, l2 per-node L2 normalization; a "+ln" suffix on the rung id overrides;
# head: mlp (Flatten, Dense 1024 ReLU, Dense 10), linear64 (Dense 10 on 64 root outputs),
# blocksum (softmax over 64 root outputs, fixed block-sum into 10 class probabilities).
SPEC = {
    "1": dict(stem=128, widths=(256, 512, 512), topo="quad", merge="conv", norm="bn", head="mlp"),
    "2": dict(stem=8, widths=(8, 8, 8), topo="quad", merge="conv", norm="bn", head="mlp"),
    "2u": dict(stem=8, widths=(8, 8, 8), topo="quad", merge="untied", norm="bn", head="mlp"),
    "4a": dict(stem=8, widths=(8,) * 6, topo="binary", merge="relu", norm="bn", head="mlp"),
    "4": dict(stem=8, widths=(8,) * 6, topo="binary", merge="bilinear", norm="ln", head="mlp"),
    "5L": dict(stem=8, widths=(8,) * 5 + (64,), topo="binary", merge="bilinear", norm="ln", head="linear64"),
    "5": dict(stem=8, widths=(8,) * 5 + (64,), topo="binary", merge="bilinear", norm="ln", head="blocksum"),
}
REGISTERED = ("1", "2", "4", "5")  # rung 3 (sharing) is identical to rung 2 for a conv tree; see RUNGS.md
DEFAULT_RUNGS = "1,2,2u,4a,4a+ln,4,5L,5"
DEFAULT_NOAUG_RUNGS = "1,5"

app = modal.App("cptp-v6-w13-ladder")
image = modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu").pip_install("numpy<2")
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)
result_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)


def _name(rung, augment, lr, wd, seed, epochs):
    return f"r{rung}-a{int(augment)}-lr{lr:g}-wd{wd:g}-s{seed}-e{epochs}"


def fixed_readout_matrix(n_outcomes: int, n_classes: int):
    """Same block 0/1 assignment as common_functions.qks_tn_ff.fixed_readout_matrix."""
    import numpy as np

    q, r = divmod(n_outcomes, n_classes)
    M = np.zeros((n_classes, n_outcomes), dtype="float32")
    j = 0
    for c in range(n_classes):
        take = q + (1 if c < r else 0)
        M[c, j:j + take] = 1.0
        j += take
    return M


@functools.lru_cache(maxsize=None)
def _layers():
    import tensorflow as tf
    import numpy as np

    class SplitPair:
        @staticmethod
        def children(x, axis):
            _, h, w, c = x.shape
            if axis == "x":  # Morton bit 0: horizontal neighbors
                r = tf.reshape(x, [-1, h, w // 2, 2, c])
                return r[:, :, :, 0], r[:, :, :, 1]
            r = tf.reshape(x, [-1, h // 2, 2, w, c])
            return r[:, :, 0], r[:, :, 1]

    class ReluPairMerge(tf.keras.layers.Layer):
        """Shared (per level) linear map of the concatenated two children; BN and ReLU follow."""

        def __init__(self, out, axis, **kw):
            super().__init__(**kw)
            self.out, self.axis = out, axis

        def build(self, shape):
            self.w = self.add_weight("w", (2 * shape[-1], self.out), initializer="glorot_uniform")

        def call(self, x):
            a, b = SplitPair.children(x, self.axis)
            return tf.einsum("bhwi,io->bhwo", tf.concat([a, b], -1), self.w)

    class BilinearPairMerge(tf.keras.layers.Layer):
        """out_k = sum_ij A[k,i,j] a_i b_j, one tensor A per level shared across nodes."""

        def __init__(self, out, axis, **kw):
            super().__init__(**kw)
            self.out, self.axis = out, axis

        def build(self, shape):
            c = shape[-1]
            self.A = self.add_weight("A", (self.out, c, c),
                                     initializer=tf.keras.initializers.RandomNormal(stddev=1.0 / np.sqrt(c)))

        def call(self, x):
            a, b = SplitPair.children(x, self.axis)
            t = tf.einsum("bhwi,kij->bhwkj", a, self.A)
            return tf.einsum("bhwkj,bhwj->bhwk", t, b)

    class UntiedQuadMerge(tf.keras.layers.Layer):
        """2x2 stride-2 merge with separate weights at every node (no weight sharing)."""

        def __init__(self, out, **kw):
            super().__init__(**kw)
            self.out = out

        def build(self, shape):
            _, h, w, c = shape
            lim = float(np.sqrt(6.0 / (4 * c + 4 * self.out)))  # Conv2D glorot_uniform limit for a 2x2 kernel
            self.w = self.add_weight("w", (h // 2, w // 2, 4 * c, self.out),
                                     initializer=tf.keras.initializers.RandomUniform(-lim, lim))

        def call(self, x):
            return tf.einsum("bhwi,hwio->bhwo", tf.nn.space_to_depth(x, 2), self.w)

    class BlockLogProb(tf.keras.layers.Layer):
        """Fixed readout: softmax over outcomes, sum blocks into class probabilities, return their logs."""

        def __init__(self, n_outcomes, n_classes, **kw):
            super().__init__(**kw)
            self.mask = tf.constant(np.where(fixed_readout_matrix(n_outcomes, n_classes) > 0, 0.0, -1e9),
                                    dtype=tf.float32)

        def call(self, z):
            full = tf.reduce_logsumexp(z, axis=-1, keepdims=True)
            cls = tf.reduce_logsumexp(z[:, None, :] + self.mask[None], axis=-1)
            return cls - full

    class L2Norm(tf.keras.layers.Layer):
        """Per-node L2 normalization to norm sqrt(C) (unit mean square), the analogue of a unit-norm state."""

        def call(self, x):
            c = float(x.shape[-1])
            return x * tf.math.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) / c + 1e-6)

    return dict(L2Norm=L2Norm, SplitPair=SplitPair, ReluPairMerge=ReluPairMerge, BilinearPairMerge=BilinearPairMerge,
                UntiedQuadMerge=UntiedQuadMerge, BlockLogProb=BlockLogProb)


def split_rung(rung: str):
    """"4+l2" -> ("4", "l2"); plain "4" -> ("4", None). norm in {bn, l2, ln} overrides the spec's norm after each level."""
    base, _, norm = rung.partition("+")
    return base, (norm or None)


def build_model(rung: str):
    import tensorflow as tf

    L = tf.keras.layers
    S = _layers()
    base, norm = split_rung(rung)
    spec = SPEC[base]
    norm = norm or spec["norm"]
    inp = L.Input((32, 32, 3))
    x = L.Conv2D(spec["stem"], 4, strides=4, padding="valid", activation="relu", name="patch4s4")(inp)
    for i, width in enumerate(spec["widths"]):
        last = i == len(spec["widths"]) - 1
        axis = "x" if i % 2 == 0 else "y"
        if spec["merge"] == "conv":
            x = L.Conv2D(width, 2, strides=2, padding="valid", use_bias=False, name=f"tree{i}")(x)
        elif spec["merge"] == "untied":
            x = S["UntiedQuadMerge"](width, name=f"tree{i}")(x)
        elif spec["merge"] == "relu":
            x = S["ReluPairMerge"](width, axis, name=f"tree{i}")(x)
        else:
            x = S["BilinearPairMerge"](width, axis, name=f"tree{i}")(x)
        wide_root = spec["merge"] == "bilinear" and last and width > 8
        if not wide_root:
            if norm == "bn":
                x = L.BatchNormalization(name=f"bn{i}")(x)
            elif norm == "l2":
                x = S["L2Norm"](name=f"norm{i}")(x)
            else:
                x = L.LayerNormalization(name=f"norm{i}")(x)
        if spec["merge"] != "bilinear":
            x = L.Activation("relu")(x)
    x = L.Flatten()(x)
    if spec["head"] == "mlp":
        x = L.Dense(1024, activation="relu", name="tree_readout")(x)
        out = L.Dense(10)(x)
    elif spec["head"] == "linear64":
        out = L.Dense(10, name="linear_head")(x)
    else:
        out = S["BlockLogProb"](64, 10, name="block_sum_readout")(x)
    return tf.keras.Model(inp, out, name="ladder_" + rung.replace("+", "_"))


def param_counts(rung: str):
    model = build_model(rung)
    trainable = sum(int(v.shape.num_elements()) for v in model.trainable_variables)
    return dict(total=int(model.count_params()), trainable=trainable,
                tree=sum(int(v.shape.num_elements()) for v in model.trainable_variables
                         if re.match(r"(patch4s4|tree\d|bn\d|norm\d)", v.name)))


def check_rung(rung: str, batch: int = 8, seed: int = 0):
    """Output shape (batch, 10), finite loss, and nonzero finite gradients for every trainable variable."""
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = build_model(rung)
    x = tf.constant(np.random.default_rng(seed).uniform(size=(batch, 32, 32, 3)).astype("float32"))
    y = tf.constant(np.arange(batch) % 10, dtype=tf.int32)
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, out, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    assert tuple(out.shape) == (batch, 10), (rung, out.shape)
    assert np.isfinite(float(loss)), (rung, float(loss))
    zero = [v.name for v, g in zip(model.trainable_variables, grads)
            if g is None or not np.all(np.isfinite(g.numpy())) or float(tf.reduce_sum(tf.abs(g))) == 0.0]
    assert not zero, (rung, "zero or missing gradient", zero)
    if SPEC[split_rung(rung)[0]]["head"] == "blocksum":
        p = np.exp(out.numpy()).sum(-1)
        assert np.allclose(p, 1.0, atol=1e-4), (rung, "class probabilities do not sum to 1", p)
    return dict(rung=rung, out_shape=list(out.shape), loss=float(loss), n_variables=len(grads))


def _load_data():
    import numpy as np
    import tensorflow as tf

    root = Path("/data/cifar10")
    paths = [root / f"{s}.npy" for s in ("X_train", "y_train", "X_test", "y_test")]
    if all(p.exists() for p in paths):
        x_train, y_train, x_test, y_test = [np.load(p) for p in paths]
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_test = x_test.reshape(-1, 32, 32, 3)
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.reshape(-1).astype("int32")
    y_test = y_test.reshape(-1).astype("int32")
    rng = np.random.default_rng(SPLIT_SEED)
    val_parts = [rng.permutation(np.flatnonzero(y_train == c))[:500] for c in range(10)]
    val_idx = np.sort(np.concatenate(val_parts))
    keep = np.ones(len(y_train), dtype=bool)
    keep[val_idx] = False
    return (x_train[keep], y_train[keep]), (x_train[val_idx], y_train[val_idx]), (x_test, y_test), val_idx


def _dataset(x, y, seed, batch, training=False, augment=False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=training)

    def convert(a, b):
        a = tf.cast(a, tf.float32) / 255.0
        if training and augment:
            def crop_flip(img):
                img = tf.image.resize_with_crop_or_pad(img, 40, 40)
                img = tf.image.random_crop(img, (32, 32, 3))
                return tf.image.random_flip_left_right(img)
            a = tf.map_fn(crop_flip, a, fn_output_signature=tf.float32)
        return a, b

    return ds.map(convert, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


@app.function(image=image, gpu="L40S", cpu=4, memory=16384, timeout=8 * 3600,
              volumes={"/data": data_vol, "/out": result_vol})
def train_one(rung: str, augment: bool, lr: float, wd: float, seed: int,
              epochs: int, phase: str, limit_train: int = 0, subdir: str = ""):
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    (xtr, ytr), (xval, yval), (xte, yte), val_idx = _load_data()
    if limit_train:
        xtr, ytr = xtr[:limit_train], ytr[:limit_train]
    batch = 128
    checks = check_rung(rung) if phase == "smoke" else None
    tf.keras.utils.set_random_seed(seed)
    model = build_model(rung)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    cfg = dict(rung=rung, spec=SPEC[split_rung(rung)[0]], augment=augment, lr=lr, wd=wd, seed=seed, epochs=epochs,
               phase=phase, batch_size=batch, split_seed=SPLIT_SEED, n_train=len(ytr),
               n_val=len(yval), n_test=len(yte), limit_train=limit_train,
               validation_indices=val_idx.tolist())
    name = _name(rung, augment, lr, wd, seed, epochs)
    out = Path("/out") / SWEEP / (subdir or phase) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(cfg, indent=2))
    result_vol.commit()
    train_ds = _dataset(xtr, ytr, seed, batch, training=True, augment=augment)
    val_ds = _dataset(xval, yval, seed, batch)
    initial_loss = float(model.evaluate(_dataset(xtr[:min(len(xtr), 1024)], ytr[:min(len(ytr), 1024)], seed, batch), verbose=0)[0])
    hist = []
    steps = len(ytr) // batch

    class Record(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            row = dict(epoch=epoch + 1, loss=float(logs["loss"]), acc=float(logs["accuracy"]),
                       val_loss=float(logs["val_loss"]), val_acc=float(logs["val_accuracy"]),
                       lr=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
                       ms_per_step=1000 * (time.monotonic() - self.start) / steps)
            hist.append(row)
            (out / "history.json").write_text(json.dumps(hist))
            result_vol.commit()
            print(name, row, flush=True)

    start = time.monotonic()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[Record()])
    finite = all(np.isfinite([h["loss"], h["val_loss"]]).all() for h in hist)
    result = dict(cfg=cfg, params=int(model.count_params()), counts=param_counts(rung),
                  initial_loss=initial_loss, history=hist, checks=checks,
                  final_train=dict(loss=hist[-1]["loss"], acc=hist[-1]["acc"]),
                  final_val=dict(loss=hist[-1]["val_loss"], acc=hist[-1]["val_acc"]),
                  best_val=max(hist, key=lambda x: (x["val_acc"], -x["val_loss"])),
                  diverged=bool((not finite) or hist[-1]["acc"] <= 0.15),
                  ms_per_step=float(np.median([h["ms_per_step"] for h in hist[1:]] or [hist[0]["ms_per_step"]])),
                  wall_s=time.monotonic() - start, devices=[d.device_type for d in tf.config.list_physical_devices()])
    if phase == "fit":
        test_loss, test_acc = model.evaluate(_dataset(xte, yte, seed, batch), verbose=0)
        result["final_test"] = dict(loss=float(test_loss), acc=float(test_acc))
    (out / "result.json").write_text(json.dumps(result))
    result_vol.commit()
    return result


@app.local_entrypoint()
def main(phase: str = "smoke", rungs: str = DEFAULT_RUNGS, noaug_rungs: str = DEFAULT_NOAUG_RUNGS,
         seeds: str = "0,1,2", epochs: int = 63, limit_train: int = 0, lr: float = -1.0, wd: float = -1.0,
         augments: str = "1", subdir: str = ""):
    """fit: rungs with augmentation (`augments`) plus `noaug_rungs` without; lr/wd default to W4's tree selections."""
    if phase not in ("smoke", "tune", "fit"):
        raise ValueError(phase)
    sel = [r for r in rungs.split(",") if r]
    noaug = [r for r in noaug_rungs.split(",") if r]
    for r in sel + noaug:
        if split_rung(r)[0] not in SPEC or split_rung(r)[1] not in (None, "bn", "l2", "ln"):
            raise ValueError(f"unknown rung {r}; known: {sorted(SPEC)}")
    seed_list = [int(s) for s in seeds.split(",") if s]
    settings = [(r, bool(int(a))) for a in augments.split(",") if a for r in sel]
    if phase != "smoke":
        settings += [(r, False) for r in noaug]
    else:
        settings += [(r, False) for r in noaug if (r, False) not in settings]
    local = HERE / "results" / SWEEP / (subdir or phase)
    local.mkdir(parents=True, exist_ok=True)

    def hp(aug):
        d = LRWD[int(aug)]
        return (d[0] if lr < 0 else lr, d[1] if wd < 0 else wd)

    if phase == "smoke":
        jobs = [(r, aug, *hp(aug), 0, 1, phase, limit_train or 10000, subdir) for r, aug in settings]
    else:
        jobs = [(r, aug, *hp(aug), s, epochs, phase, 0, subdir) for r, aug in settings for s in seed_list]
    print(f"phase={phase}, {len(jobs)} Modal L40S jobs", flush=True)
    handles = [train_one.spawn(*j) for j in jobs]
    for j, handle in zip(jobs, handles):
        name = _name(*j[:4], j[4], j[5])
        try:
            result = handle.get()
            (local / f"{name}.json").write_text(json.dumps(result))
            print(name, result.get("final_test", result["final_val"]), flush=True)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
