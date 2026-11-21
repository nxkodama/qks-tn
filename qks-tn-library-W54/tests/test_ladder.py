"""Per-rung checks for run_ladder.py: output shape (8, 10), finite loss, nonzero finite gradients on every
trainable variable (and class probabilities summing to 1 for the block-sum readout).

Run: .venv/bin/python qks-tn-library/tests/test_ladder.py   (CPU, a few seconds; does not need modal)
"""
import importlib.util
import sys
from pathlib import Path
from unittest import mock

try:
    import modal  # noqa: F401
except ImportError:  # the venv has no modal; only the model builders are used here
    sys.modules["modal"] = mock.MagicMock()

_spec = importlib.util.spec_from_file_location("run_ladder", Path(__file__).resolve().parents[1] / "run_ladder.py")
ladder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ladder)


def _one(rung):
    r = ladder.check_rung(rung)
    assert r["out_shape"] == [8, 10]


def test_rung_1(): _one("1")
def test_rung_2(): _one("2")
def test_rung_2u(): _one("2u")
def test_rung_4a(): _one("4a")
def test_rung_4a_ln(): _one("4a+ln")
def test_rung_4(): _one("4")
def test_rung_5L(): _one("5L")
def test_rung_5(): _one("5")


def test_blocksum_matches_matrix():
    import numpy as np
    import tensorflow as tf

    z = np.random.default_rng(1).normal(size=(3, 64)).astype("float32")
    got = np.exp(ladder._layers()["BlockLogProb"](64, 10)(tf.constant(z)).numpy())
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    assert np.allclose(got, p @ ladder.fixed_readout_matrix(64, 10).T, atol=1e-5)


def test_binary_tree_pairs_morton_neighbors():
    import numpy as np
    import tensorflow as tf

    S = ladder._layers()["SplitPair"]
    x = tf.constant(np.arange(8 * 8, dtype="float32").reshape(1, 8, 8, 1))  # value = 8*y + x
    a, b = S.children(x, "x")  # horizontal neighbors (x = 2j, 2j+1), as Morton bit 0
    assert a.shape == (1, 8, 4, 1) and float(b[0, 0, 0, 0] - a[0, 0, 0, 0]) == 1.0
    a, b = S.children(x, "y")  # vertical neighbors (y = 2j, 2j+1)
    assert a.shape == (1, 4, 8, 1) and float(b[0, 0, 0, 0] - a[0, 0, 0, 0]) == 8.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
