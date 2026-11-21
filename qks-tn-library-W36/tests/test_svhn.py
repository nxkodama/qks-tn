"""Tests for the ``svhn`` task of run_ff_experiment.py (parsing, label range, holdout split, encoder).

    .venv/bin/python qks-tn-library/tests/test_svhn.py

The parsing test writes a small .mat file in the official layout; it does not download anything.
If the real data are cached in qks-tn-library/data_cache/svhn, the cached arrays are checked as well.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import run_ff_experiment as rfe  # noqa: E402


def _fake_mat(path, n, seed=0):
    from scipy.io import savemat

    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, size=(32, 32, 3, n), dtype=np.uint8)
    y = rng.integers(1, 11, size=(n, 1)).astype(np.uint8)  # official labels 1..10, 10 = digit 0
    savemat(path, {"X": X, "y": y})
    return X, y


def test_parse_shapes_labels_layout():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake_32x32.mat")
        X0, y0 = _fake_mat(path, 40)
        X, y = rfe._svhn_mat_to_uint8(path)
    assert X.shape == (40, 3072) and X.dtype == np.uint8
    assert y.shape == (40,) and y.min() >= 0 and y.max() <= 9
    assert np.array_equal(y, np.where(y0.reshape(-1) == 10, 0, y0.reshape(-1)))
    # channels-last row-major, same layout as CIFAR-10: pixel (r, c, ch) of image i is X[i, (r * 32 + c) * 3 + ch]
    for i in (0, 7, 39):
        assert X[i, (5 * 32 + 9) * 3 + 2] == X0[5, 9, 2, i]


def test_task_tables():
    assert rfe.N_CLASSES["svhn"] == 10 and rfe.IMAGE_SHAPES["svhn"] == (32, 32, 3)
    assert rfe.patch_geometry({"task": "svhn"}) == rfe.patch_geometry({"task": "cifar10"})


def test_holdout_split_is_stratified_and_disjoint():
    rng = np.random.default_rng(0)
    counts = [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]  # official train class counts, labels 0..9
    assert sum(counts) == 73257
    y = np.concatenate([np.full(c, k) for k, c in enumerate(counts)]).astype(np.float32)
    y = y[rng.permutation(len(y))]
    X = np.arange(len(y))[:, None].astype(np.uint8)
    Xt, yt, Xv, yv, idx = rfe.split_svhn_holdout(X, y)
    assert len(yt) == 68257 and len(yv) == 5000 and len(idx) == 5000 and len(set(idx.tolist())) == 5000
    assert np.array_equal(np.bincount(yv.astype(int), minlength=10), np.full(10, 500))
    assert np.array_equal(y[idx], yv)
    assert np.array_equal(np.bincount(yt.astype(int), minlength=10), np.array(counts) - 500)  # train = remainder
    # deterministic
    assert np.array_equal(rfe.split_svhn_holdout(X, y)[4], idx)
    try:
        rfe.split_svhn_holdout(X[:50000], y[:50000])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a truncated training set")


def test_encoder_and_models_run_on_svhn_shape():
    import tensorflow as tf

    for name, chi in (("L4", 8), ("LU8Bf112", 8), ("LK16B", 8)):
        ov = dict(rfe.CONFIGS[name])
        ov.update(task="svhn", seed=0, epochs=1, nepisodes=128, chi=chi, holdout_5000=True)
        cfg = rfe.make_cfg(**ov)
        model, qks, tn_layer, counts = rfe.build_model(cfg, steps_per_epoch=10, p=3072, n_classes=10)
        x = np.random.default_rng(1).random((2, 3072)).astype(np.float32)
        out = model(tf.constant(x)).numpy()
        assert out.shape == (2, 10) and np.all(np.isfinite(out)), name
        assert counts["nlayers"] == 6, (name, counts["nlayers"])  # 8x8 patch grid -> 64 leaves -> 6 layers
        assert "holdout5000" in cfg["name"] and "svhn" in cfg["name"]


def test_cached_real_data_if_present():
    d = os.path.join(rfe.DEFAULTS["cache_dir"], "svhn")
    files = [os.path.join(d, f) for f in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")]
    if not all(os.path.exists(f) for f in files):
        print("  (no cached SVHN data; real-data check skipped)")
        return
    Xtr, ytr, Xte, yte = (np.load(f) for f in files)
    assert Xtr.shape == (73257, 3072) and Xte.shape == (26032, 3072)
    assert ytr.min() == 0 and ytr.max() == 9 and yte.min() == 0 and yte.max() == 9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
