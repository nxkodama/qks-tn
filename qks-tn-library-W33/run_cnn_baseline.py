"""CIFAR-10 CNN controls for CPTP W4, trained remotely on Modal.

Examples (from repository root):
  python3 -m modal run -e env-ai-algs qks-tn-library/run_cnn_baseline.py --phase smoke
  python3 -m modal run --detach -e env-ai-algs qks-tn-library/run_cnn_baseline.py --phase tune
  python3 -m modal run --detach -e env-ai-algs qks-tn-library/run_cnn_baseline.py --phase fit
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SWEEP = "cptp-v5-cnn"
ARCHS = ("raw", "patch", "tree")
LRS = (0.0003, 0.001)
WDS = (0.0, 0.0001)
SPLIT_SEED = 20261102

app = modal.App("cptp-v5-cnn")
image = modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu").pip_install("numpy<2")
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)
result_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)


def _name(arch, augment, lr, wd, seed, epochs):
    return f"{arch}-a{int(augment)}-lr{lr:g}-wd{wd:g}-s{seed}-e{epochs}"


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


def _model(arch):
    import tensorflow as tf

    layers = tf.keras.layers
    inp = layers.Input((32, 32, 3))
    x = inp
    if arch in ("patch", "tree"):
        x = layers.Conv2D(64 if arch == "patch" else 128, 4, strides=4, padding="valid", activation="relu", name="patch4s4")(x)
    if arch in ("raw", "patch"):
        widths = (64, 128, 256, 256)
        for stage, width in enumerate(widths):
            for sub in range(2):
                x = layers.Conv2D(width, 3, strides=2 if stage and sub == 0 else 1,
                                  padding="same", use_bias=False, name=f"conv{stage}_{sub}")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif arch == "tree":
        for stage, width in enumerate((256, 512, 512)):
            x = layers.Conv2D(width, 2, strides=2, padding="valid", use_bias=False,
                              name=f"tree{stage}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu", name="tree_readout")(x)
    else:
        raise ValueError(arch)
    out = layers.Dense(10)(x)
    return tf.keras.Model(inp, out, name=f"cnn_{arch}")


@app.function(image=image, gpu="L40S", cpu=4, memory=16384, timeout=8 * 3600,
              volumes={"/data": data_vol, "/out": result_vol})
def train_one(arch: str, augment: bool, lr: float, wd: float, seed: int,
              epochs: int, phase: str, limit_train: int = 0):
    import numpy as np
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    (xtr, ytr), (xval, yval), (xte, yte), val_idx = _load_data()
    if limit_train:
        xtr, ytr = xtr[:limit_train], ytr[:limit_train]
    batch = 128
    model = _model(arch)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    cfg = dict(arch=arch, augment=augment, lr=lr, wd=wd, seed=seed, epochs=epochs,
               phase=phase, batch_size=batch, split_seed=SPLIT_SEED, n_train=len(ytr),
               n_val=len(yval), n_test=len(yte), limit_train=limit_train,
               validation_indices=val_idx.tolist())
    name = _name(arch, augment, lr, wd, seed, epochs)
    out = Path("/out") / SWEEP / phase / name
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
    result = dict(cfg=cfg, params=int(model.count_params()), initial_loss=initial_loss, history=hist,
                  final_val=dict(loss=hist[-1]["val_loss"], acc=hist[-1]["val_acc"]),
                  best_val=max(hist, key=lambda x: (x["val_acc"], -x["val_loss"])),
                  ms_per_step=float(np.median([h["ms_per_step"] for h in hist[1:]] or [hist[0]["ms_per_step"]])),
                  wall_s=time.monotonic() - start, devices=[d.device_type for d in tf.config.list_physical_devices()])
    if phase == "fit":
        test_loss, test_acc = model.evaluate(_dataset(xte, yte, seed, batch), verbose=0)
        result["final_test"] = dict(loss=float(test_loss), acc=float(test_acc))
    (out / "result.json").write_text(json.dumps(result))
    result_vol.commit()
    return result


@app.local_entrypoint()
def main(phase: str = "smoke", archs: str = "raw,patch,tree", augments: str = "0,1",
         seeds: str = "0,1,2", epochs: int = 15, limit_train: int = 0):
    if phase not in ("smoke", "tune", "fit"):
        raise ValueError(phase)
    selected_archs = [a for a in archs.split(",") if a]
    selected_augments = [bool(int(a)) for a in augments.split(",") if a]
    selected_seeds = [int(s) for s in seeds.split(",") if s]
    local = HERE / "results" / SWEEP / phase
    local.mkdir(parents=True, exist_ok=True)
    jobs = []
    if phase == "smoke":
        jobs = [(a, aug, 0.001, 0.0001, 0, 1, phase, limit_train or 2048)
                for a in selected_archs for aug in selected_augments]
    elif phase == "tune":
        jobs = [(a, aug, lr, wd, 0, epochs, phase, 0)
                for a in selected_archs for aug in selected_augments for lr in LRS for wd in WDS]
    else:
        tune_dir = HERE / "results" / SWEEP / "tune"
        for a in selected_archs:
            for aug in selected_augments:
                candidates = []
                for lr in LRS:
                    for wd in WDS:
                        p = tune_dir / f"{_name(a, aug, lr, wd, 0, 15)}.json"
                        if not p.exists():
                            raise FileNotFoundError(p)
                        candidates.append(json.loads(p.read_text()))
                best = max(candidates, key=lambda r: (r["final_val"]["acc"], -r["final_val"]["loss"]))
                for seed in selected_seeds:
                    jobs.append((a, aug, best["cfg"]["lr"], best["cfg"]["wd"], seed, epochs, phase, 0))
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
