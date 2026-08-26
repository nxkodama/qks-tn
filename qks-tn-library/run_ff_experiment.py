"""Train / evaluate QKS+TTN models (baseline and feedforward variants) on MNIST.

Reproduces the data pipeline, optimiser and schedule of the repo notebooks
(``train_binary_classifier.ipynb`` / ``train_multiclass_classifier.ipynb``) and logs
per-epoch training and validation (= test set) metrics to JSON.

Usage (local)::

    .venv/bin/python qks-tn-library/run_ff_experiment.py --task binary --mode B --epochs 15 \
        --seed 0 --out qks-tn-library/runs/binary_B_s0.json

``train(cfg, on_epoch=...)`` is importable (used by ``modal_ff_sweep.py``).

Checkpointing: a ``tf.train.Checkpoint`` (weights, AdamW moments, optimizer step counter)
is written to ``cfg['ckpt_dir']`` after every epoch together with the metric history.
Re-running with a larger ``--epochs`` and ``--resume`` continues training from the last
epoch with the learning-rate / weight-decay schedules exactly where they left off.
The schedule's warm-restart cycles are 1, 2, 4, 8, 16, ... epochs, so cycles end at epochs
1, 3, 7, 15, 31, 63; the state at each of these is kept permanently in
``<ckpt_dir>/cycle-<epoch>/`` (restore one with ``--resume-from``), while ``<ckpt_dir>/ckpt-*``
is the rolling latest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common_functions"))

DEFAULTS = dict(
    task="multiclass",  # binary | multiclass | cifar10 | imagenet32
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
    ckpt_dir=None,  # directory for per-epoch tf.train.Checkpoint (None = no checkpointing)
    resume=False,  # resume from the latest checkpoint in ckpt_dir if one exists
    resume_from=None,  # explicit checkpoint prefix to restore instead (e.g. <ckpt_dir>/cycle-015/ckpt)
    name=None,
    nepisodes=128,
    chi=4,
    sigma=0.125,  # QKS weight std for 784-dim inputs; scaled by sqrt(784/p) for other input sizes
    head="auto",  # const (paper's fixed 0/1 readout, <=10 classes) | linear (trainable logits) | auto
    shuffle_buffer=250_000,  # cap on the shuffle buffer (full shuffle below this size)
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
    "C6": dict(mode="A", share_per_layer=False),
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
        if int(cfg["nepisodes"]) != 128:
            parts.append(f"n{cfg['nepisodes']}")  # 128 episodes = 6 tree layers, 256 = 7, 512 = 8
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
N_CLASSES = {"binary": 2, "multiclass": 10, "cifar10": 10, "imagenet32": 1000}
HF_IMAGENET32 = "benjamin-paine/imagenet-1k-32x32"  # ImageNet-1k, centre-crop + Lanczos to 32x32


def _hf_split_to_uint8(repo: str, split: str, log=print):
    """Decode an image-classification HF split to (X uint8 (n, 3072), y int16)."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    label_col = "class_label" if "class_label" in ds.column_names else "label"
    n = len(ds)
    X = np.empty((n, 32 * 32 * 3), dtype=np.uint8)
    y = np.asarray(ds[label_col], dtype=np.int16)
    t0 = time.time()
    for i, img in enumerate(ds["image"] if n < 200_000 else (ex["image"] for ex in ds)):
        X[i] = np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1)
        if i % 200_000 == 0 and i:
            log(f"  decoded {i}/{n} ({time.time() - t0:.0f}s)")
    return X, y


def load_data(task: str, cache_dir: str, log=print):
    """Data pipeline per task.

    binary / multiclass : the notebooks' MNIST pipeline (/255, float32, deskew); returns float32.
    cifar10             : tf.keras CIFAR-10, (n, 3072) uint8 channels-last (scaled to [0,1] in the
                          tf.data pipeline); 10 classes, fixed readout as for MNIST.
    imagenet32          : HF ``benjamin-paine/imagenet-1k-32x32`` train (1,281,167) / validation
                          (50,000), (n, 3072) uint8; 1000 classes -> trainable linear head.
    """
    import numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    if task in ("cifar10", "imagenet32"):
        d = os.path.join(cache_dir, task)
        files = [os.path.join(d, f) for f in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")]
        if not all(os.path.exists(f) for f in files):
            os.makedirs(d, exist_ok=True)
            if task == "cifar10":
                from tensorflow.keras.datasets import cifar10

                (Xtr, ytr), (Xte, yte) = cifar10.load_data()
                Xtr = Xtr.reshape(len(Xtr), -1).astype(np.uint8)
                Xte = Xte.reshape(len(Xte), -1).astype(np.uint8)
                ytr, yte = ytr.reshape(-1).astype(np.int16), yte.reshape(-1).astype(np.int16)
            else:
                log(f"downloading + decoding {HF_IMAGENET32} (train)")
                Xtr, ytr = _hf_split_to_uint8(HF_IMAGENET32, "train", log)
                log(f"downloading + decoding {HF_IMAGENET32} (validation)")
                Xte, yte = _hf_split_to_uint8(HF_IMAGENET32, "validation", log)
                keep = yte < 1000  # label 1000 = "none" (only in the unlabeled test split)
                Xte, yte = Xte[keep], yte[keep]
            for f, a in zip(files, (Xtr, ytr, Xte, yte)):
                np.save(f + ".tmp.npy", a)
                os.replace(f + ".tmp.npy", f)
        X_train, y_train, X_test, y_test = (np.load(f, mmap_mode="r") for f in files)
        return np.ascontiguousarray(X_train), np.asarray(y_train).astype("float32"), np.ascontiguousarray(X_test), np.asarray(y_test).astype("float32")

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


def make_dataset(X, y, batch_size: int, shuffle: bool, seed: int, shuffle_buffer: int = 250_000):
    """tf.data pipeline backed by a CPU tf.Variable (sidesteps the 2 GB graph-constant limit).

    Indices are shuffled (full shuffle up to ``shuffle_buffer`` examples) and batches gathered
    from the variable; uint8 images are scaled to [0, 1] on the fly.
    """
    import numpy as np
    import tensorflow as tf

    n = len(y)
    with tf.device("/CPU:0"):
        X_var = tf.Variable(X, trainable=False, name="X_data")
        y_var = tf.Variable(np.asarray(y, dtype="float32"), trainable=False, name="y_data")
    scale = X_var.dtype == tf.uint8

    def fetch(idx):
        xb = tf.gather(X_var, idx)
        if scale:
            xb = tf.cast(xb, tf.float32) / 255.0
        return xb, tf.gather(y_var, idx)

    ds = tf.data.Dataset.range(n)
    if shuffle:
        ds = ds.shuffle(min(n, shuffle_buffer), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).map(fetch, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(cfg: dict, steps_per_epoch: int, p: int = 784, n_classes: int | None = None):
    import numpy as np
    import tensorflow as tf
    from tensorflow_addons.optimizers import AdamW

    import qks_tn_ff as ff

    n_classes = n_classes or N_CLASSES[cfg["task"]]
    sigma = float(cfg["sigma"]) * float(np.sqrt(784.0 / p))  # keep the kitchen-sink angle scale
    head_kind = cfg.get("head", "auto")
    if head_kind == "auto":
        head_kind = "const" if n_classes <= 10 else "linear"
    nlayers = int(np.log2(cfg["nepisodes"] / np.log2(cfg["chi"])))
    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    if cfg["fast_qks"]:
        qks = ff.FastFeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, sigma)
    else:
        qks = ff.FeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, sigma)

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

    if head_kind == "const":
        if n_classes == 2:
            head = ff.ConstMulBinary()
        elif n_classes == 10:
            head = ff.ConstMul()
        else:
            raise ValueError("the fixed readout supports 2 or 10 classes; use head='linear'")
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        # trainable logits from the 16 readout probabilities (needed when classes > 16)
        head = tf.keras.layers.Dense(n_classes, name="linear_head")
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(p,)), qks, tn_layer, head])
    if cfg["freeze_qks"]:
        qks.trainable = False

    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(cfg["lr"], steps_per_epoch)
    wd_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(cfg["wd"], steps_per_epoch)
    model.compile(
        loss=loss,
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule),
        metrics=["accuracy"],
    )
    counts = dict(
        total=int(model.count_params()),
        qks=int(qks.count_params()),
        tn=int(tn_layer.count_params()),
        tn_effective=int(tn_effective if tn_effective is not None else tn_layer.count_params()),
        head=int(head.count_params()),
        trainable=int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
        input_dim=int(p),
        n_classes=int(n_classes),
        sigma=float(sigma),
        head_kind=head_kind,
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


def _eval_sampled(model, test_ds):
    """Evaluate with ``training=True`` forward passes (finite-shot statistics)."""
    import numpy as np
    import tensorflow as tf

    losses, correct, n = 0.0, 0, 0
    for xb, yb in test_ds:
        pr = model(xb, training=True)
        loss = model.loss(yb, pr)
        losses += float(loss) * int(xb.shape[0])
        correct += int(np.sum(np.argmax(pr.numpy(), -1) == yb.numpy()))
        n += int(xb.shape[0])
    return dict(loss=losses / n, acc=correct / n)


def _log(msg):
    print(msg, flush=True)


def train(cfg: dict, on_epoch=None, log=_log) -> dict:
    """Run one experiment; returns the result dict (also written to ``cfg['out']`` if set)."""
    import numpy as np
    import tensorflow as tf

    if cfg.get("intra_threads"):
        tf.config.threading.set_intra_op_parallelism_threads(int(cfg["intra_threads"]))
    if cfg.get("inter_threads"):
        tf.config.threading.set_inter_op_parallelism_threads(int(cfg["inter_threads"]))
    tf.get_logger().setLevel("ERROR")

    t_start = time.time()
    X_train, y_train, X_test, y_test = load_data(cfg["task"], cfg["cache_dir"], log=log)
    if cfg.get("limit_train"):
        X_train, y_train = X_train[: int(cfg["limit_train"])], y_train[: int(cfg["limit_train"])]
    bs = int(cfg["batch_size"])
    steps_per_epoch = len(y_train) // bs
    p = int(X_train.shape[1])
    n_classes = N_CLASSES[cfg["task"]]

    model, qks, tn_layer, counts = build_model(cfg, steps_per_epoch, p=p, n_classes=n_classes)
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    log(f"[{cfg['name']}] devices={devices} params={counts} steps/epoch={steps_per_epoch} n_train={len(y_train)} n_test={len(y_test)}")

    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    train_ds = make_dataset(X_train, y_train, bs, shuffle=True, seed=int(cfg["seed"]), shuffle_buffer=int(cfg.get("shuffle_buffer", 250_000)))
    test_ds = make_dataset(X_test, y_test, bs, shuffle=False, seed=int(cfg["seed"]))

    history = []
    initial_epoch = 0
    manager = None
    if cfg.get("ckpt_dir"):
        ckpt_dir = cfg["ckpt_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        # model weights + AdamW moments + the optimizer step counter (which drives the
        # CosineDecayRestarts schedules), so training continues exactly where it stopped
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        hist_path = os.path.join(ckpt_dir, "history.json")
        restore_path = None
        if cfg.get("resume_from"):
            restore_path = cfg["resume_from"]
            hist_path = os.path.join(os.path.dirname(restore_path), "history.json")
        elif cfg.get("resume") and manager.latest_checkpoint:
            restore_path = manager.latest_checkpoint
        if restore_path is not None and os.path.exists(hist_path):
            with open(hist_path) as f:
                saved = json.load(f)
            history = saved["epochs"]
            initial_epoch = len(history)
            ckpt.restore(restore_path).expect_partial()
            log(f"[{cfg['name']}] resumed from {restore_path} at epoch {initial_epoch}")
        elif cfg.get("resume") or cfg.get("resume_from"):
            log(f"[{cfg['name']}] no checkpoint to resume from in {ckpt_dir}; starting fresh")
        with open(os.path.join(ckpt_dir, "cfg.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    if initial_epoch >= int(cfg["epochs"]):
        log(f"[{cfg['name']}] already trained {initial_epoch} >= {cfg['epochs']} epochs; evaluating only")

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
            if manager is not None:
                manager.save(checkpoint_number=rec["epoch"])  # rolling latest
                state = dict(cfg=cfg, params=counts, epochs=list(history))
                with open(os.path.join(cfg["ckpt_dir"], "history.json"), "w") as f:
                    json.dump(state, f)
                e = rec["epoch"]
                if (e + 1) & e == 0:
                    # e = 2^k - 1: end of a cosine-annealing cycle (1, 3, 7, 15, 31, 63, ...);
                    # keep this state permanently
                    cycle_dir = os.path.join(cfg["ckpt_dir"], f"cycle-{e:03d}")
                    os.makedirs(cycle_dir, exist_ok=True)
                    ckpt.write(os.path.join(cycle_dir, "ckpt"))
                    with open(os.path.join(cycle_dir, "history.json"), "w") as f:
                        json.dump(state, f)
                    rec["cycle_checkpoint"] = cycle_dir
            log(
                f"[{cfg['name']}] epoch {rec['epoch']}/{cfg['epochs']} "
                f"loss {rec['loss']:.4f} acc {rec['acc']:.4f} "
                + (f"val_loss {rec['val_loss']:.4f} val_acc {rec['val_acc']:.4f} " if rec["val_loss"] is not None else "")
                + f"({rec['time_s']:.0f}s, {rec['ms_per_step']:.0f} ms/step)"
            )
            if on_epoch is not None:
                on_epoch(dict(cfg=cfg, params=counts, epochs=list(history)))

    if initial_epoch < int(cfg["epochs"]):
        model.fit(
            train_ds,
            epochs=int(cfg["epochs"]),
            initial_epoch=initial_epoch,
            validation_data=test_ds if cfg["eval_every_epoch"] else None,
            verbose=0,
            callbacks=[EpochLogger()],
        )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, -1).astype(int)
    test_errors = np.nonzero(preds != y_test.astype(int))[0].astype(int).tolist()
    preds = preds.tolist()
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
        resumed_from_epoch=initial_epoch,
        checkpoint=(manager.latest_checkpoint if manager is not None else None),
        ms_per_step=float(np.median([h["ms_per_step"] for h in history[1:]] or [history[0]["ms_per_step"]])),
        test_predictions=preds,
        test_errors=test_errors,
    )
    if result["best_val"] is not None:
        va, nvl, ep = result["best_val"]
        result["best_val"] = dict(val_acc=va, val_loss=-nvl, epoch=ep)
    if cfg.get("shots"):
        result["final_test_sampled"] = _eval_sampled(model, test_ds)
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
    ap.add_argument("--task", choices=["binary", "multiclass", "cifar10", "imagenet32"])
    ap.add_argument("--head", choices=["auto", "const", "linear"])
    ap.add_argument("--mode", choices=["baseline", "A", "B"])
    ap.add_argument("--impl", choices=["new", "old"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--share-per-layer", dest="share_per_layer", action="store_true", default=None)
    ap.add_argument("--no-share-per-layer", dest="share_per_layer", action="store_false")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--nepisodes", type=int, help="QKS episodes = tree leaves x 2; 128 -> 6 layers, 256 -> 7, 512 -> 8")
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
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", help="checkpoint directory (default runs/ckpt/<name>)")
    ap.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true", help="disable checkpointing")
    ap.add_argument("--resume", action="store_true", default=None, help="continue from the latest checkpoint in --ckpt-dir")
    ap.add_argument("--resume-from", dest="resume_from", help="checkpoint prefix to restore, e.g. runs/ckpt/<name>/cycle-015/ckpt")
    return ap


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    config_name = args.pop("config")
    overrides = dict(CONFIGS.get(config_name or "", {}))
    overrides.update({k: v for k, v in args.items() if v is not None})
    if overrides.get("ff_null") == "none":
        overrides["ff_null"] = None
    no_ckpt = overrides.pop("no_checkpoint", False)
    cfg = make_cfg(**overrides)
    cfg["config"] = config_name
    if cfg["out"] is None:
        cfg["out"] = os.path.join(HERE, "runs", cfg["name"] + ".json")
    if no_ckpt:
        cfg["ckpt_dir"] = None
    elif cfg["ckpt_dir"] is None:
        cfg["ckpt_dir"] = os.path.join(HERE, "runs", "ckpt", cfg["name"])
    train(cfg)


if __name__ == "__main__":
    main()
