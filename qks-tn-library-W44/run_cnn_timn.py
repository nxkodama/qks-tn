"""CNN baselines on the W14 timn100 task (first 100 Tiny-ImageNet classes, 32x32), trained on Modal.

Copy of run_cnn_baseline.py (W4) with these changes: data come from the timn100 cache that
run_ff_experiment.load_data writes to the qks-tn-data volume (50,000 train, 5,000 validation images used
as the test set, no holdout split); 100 output classes; learning rate and weight decay are W4's selected
no-augmentation settings (no new tuning); the 5,000 evaluation images are scored every epoch and only the
final-epoch score is reported.

Examples (from repository root):
  python3 -m modal run -e env-ai-algs qks-tn-library/run_cnn_timn.py --phase smoke
  python3 -m modal run --detach -e env-ai-algs qks-tn-library/run_cnn_timn.py --phase fit --epochs 63
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SWEEP = "cptp-v6-w14-cnn"
# W4 (REPORT-W4.md) selected no-augmentation settings on CIFAR-10 validation
SETTINGS = {"raw": (0.001, 0.0001), "patch": (0.001, 0.0), "tree": (0.001, 0.0001)}

app = modal.App("cptp-v6-w14-cnn")
image = modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu").pip_install("numpy<2")
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)
result_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)


def _name(arch, lr, wd, seed, epochs):
    return f"timn100-{arch}-lr{lr:g}-wd{wd:g}-s{seed}-e{epochs}"


def _load_data():
    import numpy as np

    root = Path("/data/timn100")
    x_train, y_train, x_test, y_test = [np.load(root / f"{s}.npy") for s in ("X_train", "y_train", "X_test", "y_test")]
    return (x_train.reshape(-1, 32, 32, 3), y_train.astype("int32")), (x_test.reshape(-1, 32, 32, 3), y_test.astype("int32"))


def _dataset(x, y, seed, batch, training=False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch, drop_remainder=training)
    return ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def _model(arch, n_classes=100):
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
            x = layers.Conv2D(width, 2, strides=2, padding="valid", use_bias=False, name=f"tree{stage}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation="relu", name="tree_readout")(x)
    else:
        raise ValueError(arch)
    out = layers.Dense(n_classes)(x)
    return tf.keras.Model(inp, out, name=f"cnn_{arch}")


@app.function(image=image, gpu="L40S", cpu=4, memory=16384, timeout=8 * 3600,
              volumes={"/data": data_vol, "/out": result_vol})
def train_one(arch: str, lr: float, wd: float, seed: int, epochs: int, phase: str, limit_train: int = 0):
    import numpy as np
    import tensorflow as tf

    data_vol.reload()
    tf.keras.utils.set_random_seed(seed)
    (xtr, ytr), (xte, yte) = _load_data()
    if limit_train:
        xtr, ytr = xtr[:limit_train], ytr[:limit_train]
    batch = 128
    model = _model(arch)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    cfg = dict(task="timn100", arch=arch, augment=False, lr=lr, wd=wd, seed=seed, epochs=epochs, phase=phase,
               batch_size=batch, n_train=len(ytr), n_test=len(yte), limit_train=limit_train)
    name = _name(arch, lr, wd, seed, epochs)
    out = Path("/out") / SWEEP / phase / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(cfg, indent=2))
    result_vol.commit()
    train_ds = _dataset(xtr, ytr, seed, batch, training=True)
    test_ds = _dataset(xte, yte, seed, batch)
    initial_loss = float(model.evaluate(_dataset(xtr[:1024], ytr[:1024], seed, batch), verbose=0)[0])
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
    model.fit(train_ds, validation_data=test_ds, epochs=epochs, verbose=0, callbacks=[Record()])
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    result = dict(cfg=cfg, params=int(model.count_params()), initial_loss=initial_loss, history=hist,
                  final_test=dict(loss=float(test_loss), acc=float(test_acc)),
                  ms_per_step=float(np.median([h["ms_per_step"] for h in hist[1:]] or [hist[0]["ms_per_step"]])),
                  wall_s=time.monotonic() - start, devices=[d.device_type for d in tf.config.list_physical_devices()])
    (out / "result.json").write_text(json.dumps(result))
    result_vol.commit()
    return result


@app.local_entrypoint()
def main(phase: str = "smoke", archs: str = "raw,patch,tree", seeds: str = "0", epochs: int = 63, limit_train: int = 0):
    if phase not in ("smoke", "fit"):
        raise ValueError(phase)
    jobs = []
    for a in [x for x in archs.split(",") if x]:
        lr, wd = SETTINGS[a]
        for s in [int(x) for x in seeds.split(",") if x]:
            jobs.append((a, lr, wd, s, 1 if phase == "smoke" else epochs, phase, limit_train))
    local = HERE / "results" / SWEEP / phase
    local.mkdir(parents=True, exist_ok=True)
    print(f"phase={phase}, {len(jobs)} Modal L40S jobs", flush=True)
    handles = [train_one.spawn(*j) for j in jobs]
    for j, handle in zip(jobs, handles):
        name = _name(*j[:5])
        try:
            result = handle.get()
            (local / f"{name}.json").write_text(json.dumps(result))
            print(name, result["final_test"], flush=True)
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
