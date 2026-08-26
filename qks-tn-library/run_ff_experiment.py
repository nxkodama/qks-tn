"""Train / evaluate QKS+TTN models (baseline and feedforward variants) on MNIST.

Reproduces the data pipeline, optimiser and schedule of the repo notebooks
(``train_binary_classifier.ipynb`` / ``train_multiclass_classifier.ipynb``) and logs
per-epoch training and validation (= test set) metrics to JSON.

Usage (local)::

    .venv/bin/python qks-tn-library/run_ff_experiment.py --task binary --mode B --epochs 15 \
        --seed 0 --out qks-tn-library/runs/binary_B_s0.json

``train(cfg, on_epoch=...)`` is importable (used by ``modal_ff_sweep.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common_functions"))

DEFAULTS = dict(
    task="multiclass",  # binary | multiclass
    mode="baseline",  # baseline | A | B
    impl="new",  # new (qks_tn_ff sweep) | old (original TNLayer; baseline only)
    epochs=15,
    batch_size=32,
    shots=None,
    share_per_layer=True,
    seed=0,
    ff_layers=None,  # e.g. "1,2,3,4,5"
    ff_features="raw",
    ff_null=None,
    freeze_qks=False,
    fast_qks=True,
    limit_train=None,
    eval_every_epoch=True,
    intra_threads=0,
    inter_threads=0,
    cache_dir=os.path.join(HERE, "data_cache"),
    out=None,
    name=None,
    nepisodes=128,
    chi=4,
    sigma=0.125,
    lr=1e-3,
    wd=4e-4,
)

CONFIGS = {
    # name: overrides on top of DEFAULTS (task/seed/epochs supplied separately)
    "C0": dict(mode="baseline", share_per_layer=True),
    "C1": dict(mode="baseline", share_per_layer=False),
    "C2": dict(mode="B", share_per_layer=True),
    "C3": dict(mode="B", share_per_layer=True, ff_null="constant"),
    "C4": dict(mode="B", share_per_layer=False),
    "C5": dict(mode="A", share_per_layer=True),
    "F0": dict(mode="baseline", share_per_layer=True, freeze_qks=True),
    "F2": dict(mode="B", share_per_layer=True, freeze_qks=True),
    "F5": dict(mode="A", share_per_layer=True, freeze_qks=True),
    # phase-2 ablations
    "C2s16": dict(mode="B", share_per_layer=True, shots=16),
    "C2s128": dict(mode="B", share_per_layer=True, shots=128),
    "C2s1024": dict(mode="B", share_per_layer=True, shots=1024),
    "C2root": dict(mode="B", share_per_layer=True, ff_layers="5"),
    "C2top2": dict(mode="B", share_per_layer=True, ff_layers="4,5"),
    "C2c": dict(mode="B", share_per_layer=True, ff_features="centered"),
    "C0old": dict(mode="baseline", share_per_layer=True, impl="old", fast_qks=False),
}


def make_cfg(**overrides) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg["name"] is None:
        parts = [cfg["task"], cfg["mode"], "shared" if cfg["share_per_layer"] else "pervertex"]
        if cfg["ff_null"]:
            parts.append(f"null-{cfg['ff_null']}")
        if cfg["shots"]:
            parts.append(f"shots{cfg['shots']}")
        if cfg["ff_layers"]:
            parts.append("ffl" + str(cfg["ff_layers"]).replace(",", ""))
        if cfg["ff_features"] != "raw":
            parts.append(cfg["ff_features"])
        if cfg["freeze_qks"]:
            parts.append("frozenqks")
        if cfg["impl"] != "new":
            parts.append(cfg["impl"])
        parts.append(f"s{cfg['seed']}")
        cfg["name"] = "_".join(parts)
    return cfg


def _parse_layers(v):
    if v is None or v == "" or v == "all":
        return None
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(",") if x.strip()]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_data(task: str, cache_dir: str):
    """Notebook pipeline: MNIST -> (3,5) filter for binary -> /255 -> float32 -> deskew."""
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"mnist_deskew_{task}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X_train"], z["y_train"], z["X_test"], z["y_test"]
    from tensorflow.keras.datasets import mnist

    from general_functions import deskewAll

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if task == "binary":
        mtr = (y_train == 3) | (y_train == 5)
        mte = (y_test == 3) | (y_test == 5)
        X_train, y_train, X_test, y_test = X_train[mtr], y_train[mtr], X_test[mte], y_test[mte]
        y_train = np.where(y_train == 3, 0, 1)
        y_test = np.where(y_test == 3, 0, 1)
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255
    X_train = deskewAll(X_train.astype("float32")).astype("float32")
    X_test = deskewAll(X_test.astype("float32")).astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    tmp = cache + ".tmp.npz"
    np.savez(tmp, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    os.replace(tmp, cache)
    return X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(cfg: dict, steps_per_epoch: int):
    import tensorflow as tf
    from tensorflow_addons.optimizers import AdamW

    import qks_tn_ff as ff

    p = 784
    nlayers = int(np.log2(cfg["nepisodes"] / np.log2(cfg["chi"])))
    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    if cfg["fast_qks"]:
        qks = ff.FastFeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, cfg["sigma"])
    else:
        qks = ff.FeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, cfg["sigma"])

    if cfg["impl"] == "old":
        if cfg["mode"] != "baseline":
            raise ValueError("impl='old' supports mode='baseline' only")
        import tensornetwork as tn

        import qks_tn as old

        tn.set_default_backend("tensorflow")
        rho0 = qks(tf.zeros((1, p))).numpy()
        uni_array = tf.constant(np.zeros([cfg["chi"]] * 4 + [nlayers]), dtype="complex64")
        obs = [rho0[0, :, :, j] for j in range(cfg["nepisodes"] // 2)]
        nodes_set, edge_order = old.construct_dttn(cfg["chi"], uni_array, obs, cfg["nepisodes"])
        _, path = old.greedy(nodes_set, output_edge_order=edge_order)
        tn_layer = old.TNLayer(cfg["chi"], nlayers, path)
        tn_effective = (cfg["chi"] ** 4 - 1) * nlayers
    else:
        tn_layer = ff.FeedforwardTNLayer(
            chi=cfg["chi"],
            nlayers=nlayers,
            mode=cfg["mode"],
            share_per_layer=cfg["share_per_layer"],
            shots=cfg["shots"],
            ff_layers=_parse_layers(cfg["ff_layers"]),
            ff_features=cfg["ff_features"],
            ff_null=cfg["ff_null"],
            seed=int(cfg["seed"]),
        )
        tn_effective = None

    head = ff.ConstMulBinary() if cfg["task"] == "binary" else ff.ConstMul()
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(p,)), qks, tn_layer, head])
    if cfg["freeze_qks"]:
        qks.trainable = False

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(cfg["lr"], steps_per_epoch)
    wd_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(cfg["wd"], steps_per_epoch)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule),
        metrics=["accuracy"],
    )
    counts = dict(
        total=int(model.count_params()),
        qks=int(qks.count_params()),
        tn=int(tn_layer.count_params()),
        tn_effective=int(tn_effective if tn_effective is not None else tn_layer.count_params()),
        trainable=int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
    )
    return model, qks, tn_layer, counts


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _eval_sampled(model, X, y, batch_size):
    """Evaluate with ``training=True`` forward passes (finite-shot statistics)."""
    import tensorflow as tf

    losses, correct, n = 0.0, 0, 0
    for i in range(0, len(X), batch_size):
        xb, yb = X[i : i + batch_size], y[i : i + batch_size]
        pr = model(xb, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(yb, pr)
        losses += float(tf.reduce_sum(loss))
        correct += int(np.sum(np.argmax(pr.numpy(), -1) == yb))
        n += len(xb)
    return dict(loss=losses / n, acc=correct / n)


def train(cfg: dict, on_epoch=None, log=print) -> dict:
    """Run one experiment; returns the result dict (also written to ``cfg['out']`` if set)."""
    import tensorflow as tf

    if cfg.get("intra_threads"):
        tf.config.threading.set_intra_op_parallelism_threads(int(cfg["intra_threads"]))
    if cfg.get("inter_threads"):
        tf.config.threading.set_inter_op_parallelism_threads(int(cfg["inter_threads"]))
    tf.get_logger().setLevel("ERROR")

    t_start = time.time()
    X_train, y_train, X_test, y_test = load_data(cfg["task"], cfg["cache_dir"])
    if cfg.get("limit_train"):
        X_train, y_train = X_train[: int(cfg["limit_train"])], y_train[: int(cfg["limit_train"])]
    bs = int(cfg["batch_size"])
    steps_per_epoch = len(y_train) // bs

    model, qks, tn_layer, counts = build_model(cfg, steps_per_epoch)
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    log(f"[{cfg['name']}] devices={devices} params={counts} steps/epoch={steps_per_epoch}")

    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(y_train), seed=int(cfg["seed"]), reshuffle_each_iteration=True)
        .batch(bs)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(bs)

    history = []

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()
            self._t_val = None

        def on_test_begin(self, logs=None):
            if getattr(self, "_t0", None) is not None:
                self._t_val = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            now = time.time()
            train_time = (self._t_val or now) - self._t0
            try:
                lr_obj = self.model.optimizer.learning_rate
                lr_val = float(lr_obj(self.model.optimizer.iterations)) if callable(lr_obj) else float(lr_obj)
            except Exception:  # noqa: BLE001
                lr_val = None
            rec = dict(
                epoch=epoch + 1,
                time_s=now - self._t0,
                train_time_s=train_time,
                ms_per_step=1000.0 * train_time / max(steps_per_epoch, 1),
                loss=float(logs.get("loss", np.nan)),
                acc=float(logs.get("accuracy", np.nan)),
                val_loss=float(logs["val_loss"]) if "val_loss" in logs else None,
                val_acc=float(logs["val_accuracy"]) if "val_accuracy" in logs else None,
                lr=lr_val,
            )
            history.append(rec)
            log(
                f"[{cfg['name']}] epoch {rec['epoch']}/{cfg['epochs']} "
                f"loss {rec['loss']:.4f} acc {rec['acc']:.4f} "
                + (f"val_loss {rec['val_loss']:.4f} val_acc {rec['val_acc']:.4f} " if rec["val_loss"] is not None else "")
                + f"({rec['time_s']:.0f}s, {rec['ms_per_step']:.0f} ms/step)"
            )
            if on_epoch is not None:
                on_epoch(dict(cfg=cfg, params=counts, epochs=list(history)))

    model.fit(
        train_ds,
        epochs=int(cfg["epochs"]),
        validation_data=test_ds if cfg["eval_every_epoch"] else None,
        verbose=0,
        callbacks=[EpochLogger()],
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, -1).astype(int).tolist()
    result = dict(
        cfg=cfg,
        git_commit=_git_commit(),
        versions=dict(tensorflow=tf.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        devices=devices,
        params=counts,
        steps_per_epoch=steps_per_epoch,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        epochs=history,
        final_test=dict(loss=float(test_loss), acc=float(test_acc)),
        best_val=(
            max(((h["val_acc"], -h["val_loss"], h["epoch"]) for h in history if h["val_acc"] is not None), default=None)
        ),
        total_wall_s=time.time() - t_start,
        ms_per_step=float(np.median([h["ms_per_step"] for h in history[1:]] or [history[0]["ms_per_step"]])),
        test_predictions=preds,
    )
    if result["best_val"] is not None:
        va, nvl, ep = result["best_val"]
        result["best_val"] = dict(val_acc=va, val_loss=-nvl, epoch=ep)
    if cfg.get("shots"):
        result["final_test_sampled"] = _eval_sampled(model, X_test, y_test, bs)
    log(
        f"[{cfg['name']}] DONE test_loss {test_loss:.4f} test_acc {test_acc:.4f} "
        f"wall {result['total_wall_s']/60:.1f} min ({result['ms_per_step']:.0f} ms/step)"
    )
    if cfg.get("out"):
        os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
        with open(cfg["out"], "w") as f:
            json.dump(result, f)
    return result


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), help="named configuration (overrides mode/share/etc.)")
    ap.add_argument("--task", choices=["binary", "multiclass"])
    ap.add_argument("--mode", choices=["baseline", "A", "B"])
    ap.add_argument("--impl", choices=["new", "old"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--share-per-layer", dest="share_per_layer", action="store_true", default=None)
    ap.add_argument("--no-share-per-layer", dest="share_per_layer", action="store_false")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ff-layers", dest="ff_layers", help="comma-separated layers receiving feedforward, e.g. 4,5")
    ap.add_argument("--ff-features", dest="ff_features", choices=["raw", "centered"])
    ap.add_argument("--ff-null", dest="ff_null", choices=["none", "constant", "shuffle"])
    ap.add_argument("--freeze-qks", dest="freeze_qks", action="store_true", default=None)
    ap.add_argument("--fast-qks", dest="fast_qks", action="store_true", default=None)
    ap.add_argument("--no-fast-qks", dest="fast_qks", action="store_false")
    ap.add_argument("--limit-train", dest="limit_train", type=int)
    ap.add_argument("--no-eval-every-epoch", dest="eval_every_epoch", action="store_false", default=None)
    ap.add_argument("--intra-threads", dest="intra_threads", type=int)
    ap.add_argument("--inter-threads", dest="inter_threads", type=int)
    ap.add_argument("--cache-dir", dest="cache_dir")
    ap.add_argument("--out")
    ap.add_argument("--name")
    return ap


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    overrides = dict(CONFIGS.get(args.pop("config") or "", {}))
    overrides.update({k: v for k, v in args.items() if v is not None})
    if overrides.get("ff_null") == "none":
        overrides["ff_null"] = None
    cfg = make_cfg(**overrides)
    if cfg["out"] is None:
        cfg["out"] = os.path.join(HERE, "runs", cfg["name"] + ".json")
    train(cfg)


if __name__ == "__main__":
    main()
