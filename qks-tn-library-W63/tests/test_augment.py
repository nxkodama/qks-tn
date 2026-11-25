"""Tests for the W34 training-only augmentation (run_ff_experiment.augment_images / make_dataset).

    .venv/bin/python qks-tn-library-W34/tests/test_augment.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import run_ff_experiment as rfe  # noqa: E402

SHAPE = (32, 32, 3)


def images(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=(n, 32 * 32 * 3), dtype=np.uint8)  # no zero pixels, so zero fill is detectable


def test_default_off_is_identity():
    X = images(64)
    y = np.arange(64) % 10
    for aug in (None, "", "none"):
        a = next(iter(rfe.make_dataset(X, y, 64, shuffle=False, seed=0, augment=aug, image_shape=SHAPE)))[0].numpy()
        b = next(iter(rfe.make_dataset(X, y, 64, shuffle=False, seed=0)))[0].numpy()
        assert np.array_equal(a, b)
    assert rfe.DEFAULTS["augment"] is None
    assert rfe.make_cfg(task="cifar10")["name"] == rfe.make_cfg(task="cifar10", augment=None)["name"]
    assert "aug-flip" in rfe.make_cfg(task="cifar10", augment="flip")["name"]
    try:
        rfe.parse_augment("flip+bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown token accepted")


def test_flip_per_sample():
    x = tf.constant(images(256).astype("float32") / 255.0)
    out = rfe.augment_images(x, "flip", SHAPE).numpy().reshape(-1, *SHAPE)
    ref = x.numpy().reshape(-1, *SHAPE)
    same = np.all(out == ref, axis=(1, 2, 3))
    flipped = np.all(out == ref[:, :, ::-1, :], axis=(1, 2, 3))
    assert np.all(same | flipped)
    assert 0.3 < flipped.mean() < 0.7  # both outcomes occur within one batch, about half each


def test_crop_is_shift_with_zero_pad_and_varies_per_sample():
    n = 256
    ref = (images(n).astype("float32") / 255.0).reshape(n, *SHAPE)
    out = rfe.augment_images(tf.constant(ref.reshape(n, -1)), "crop", SHAPE).numpy().reshape(n, *SHAPE)
    padded = np.pad(ref, ((0, 0), (4, 4), (4, 4), (0, 0)))
    offsets = []
    for i in range(n):
        hits = [(oy, ox) for oy in range(9) for ox in range(9) if np.array_equal(out[i], padded[i, oy:oy + 32, ox:ox + 32])]
        assert len(hits) == 1, f"sample {i} is not a shifted crop"
        offsets.append(hits[0])
    assert len(set(offsets)) > 40  # offsets differ across samples (81 possible)


def test_cutout_square_and_probability():
    n = 512
    ref = (images(n).astype("float32") / 255.0).reshape(n, *SHAPE)
    out = rfe.augment_images(tf.constant(ref.reshape(n, -1)), "cutout", SHAPE).numpy().reshape(n, *SHAPE)
    zero = np.all(out == 0, axis=-1)  # (n, 32, 32)
    changed = zero.any(axis=(1, 2))
    assert 0.35 < changed.mean() < 0.65  # p = 0.5; a square clipped at the border always keeps >= 16 pixels
    for i in np.flatnonzero(changed)[:50]:
        ys, xs = np.nonzero(zero[i])
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        assert h <= 8 and w <= 8 and zero[i].sum() == h * w  # a filled rectangle of at most 8 x 8
    assert np.array_equal(out[~changed], ref[~changed])  # untouched images are unchanged


def test_batches_differ_across_epochs_and_samples():
    X = images(128)
    y = np.arange(128) % 10
    ds = rfe.make_dataset(X, y, 128, shuffle=False, seed=0, augment="flip+crop+cutout", image_shape=SHAPE)  # fixed order, one batch
    e1 = next(iter(ds))[0].numpy()
    e2 = next(iter(ds))[0].numpy()
    assert not np.array_equal(e1, e2)  # same images, different augmentation each epoch
    clean = X.astype("float32") / 255.0
    assert np.mean(np.any(e1 != clean, axis=1)) > 0.9 and np.mean(np.any(e2 != clean, axis=1)) > 0.9
    # per-sample: identical source images get different augmentations within one batch
    Xrep = np.repeat(X[:1], 128, axis=0)
    b = next(iter(rfe.make_dataset(Xrep, y, 128, shuffle=False, seed=0, augment="flip+crop", image_shape=SHAPE)))[0].numpy()
    assert len({row.tobytes() for row in b}) > 20


def test_holdout_split_and_eval_data_unaffected():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 5000).astype(np.int16)
    rng.shuffle(y)
    X = np.zeros((50000, 4), dtype=np.uint8)
    X[:, 0] = np.arange(50000) % 251
    a = rfe.split_cifar10_holdout(X, y)
    rfe.augment_images(tf.zeros([4, 3072]), "flip+crop+cutout", SHAPE)  # augmentation uses TF's RNG, not numpy's split RNG
    b = rfe.split_cifar10_holdout(X, y)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert len(a[0]) == 45000 and len(a[2]) == 5000
    # evaluation datasets are built without augmentation: identical to the plain pipeline
    Xe, ye = images(32), np.zeros(32)
    e = next(iter(rfe.make_dataset(Xe, ye, 32, shuffle=False, seed=0)))[0].numpy()
    assert np.array_equal(e, Xe.astype("float32") / 255.0)
    src = open(os.path.join(HERE, "..", "run_ff_experiment.py")).read()
    assert "test_ds = make_dataset(X_test, y_test, bs, shuffle=False, seed=int(cfg[\"seed\"]))" in src
    assert "val_ds = make_dataset(X_val, y_val, bs, shuffle=False, seed=int(cfg[\"seed\"])) if holdout" in src


def test_configs():
    for base in ("LK16B", "LU8Bf112"):
        for sfx, aug in (("af", "flip"), ("afc", "flip+crop"), ("afco", "flip+crop+cutout")):
            c = rfe.CONFIGS[base + sfx]
            assert c["augment"] == aug
            assert {k: v for k, v in c.items() if k != "augment"} == rfe.CONFIGS[base]


def test_ls16bc0f65_family():
    # W36: the classical hybrid with augmentation differs from LS16Bc0f65 only in the augment field
    base = rfe.CONFIGS["LS16Bc0f65"]
    for sfx, aug in (("af", "flip"), ("afc", "flip+crop"), ("afco", "flip+crop+cutout")):
        c = rfe.CONFIGS["LS16Bc0f65" + sfx]
        assert c["augment"] == aug
        assert {k: v for k, v in c.items() if k != "augment"} == base
    assert base.get("augment") is None
    assert "aug-flip" in rfe.make_cfg(task="cifar10", **rfe.CONFIGS["LS16Bc0f65af"])["name"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
