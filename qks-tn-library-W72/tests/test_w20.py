"""Tests for the ``leaf_dephase`` option of ``qks_tn_cptp.CPTPTNLayer`` (W20).

    .venv/bin/python qks-tn-library/tests/test_w20.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))
sys.path.insert(0, HERE)

import qks_tn_cptp as cp  # noqa: E402
from test_cptp import pure_registers  # noqa: E402

CHI, NL, B = 2, 3, 5


def make(p, R, share=True, ff_dim=None):
    layer = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", share_per_layer=share, kraus_rank=R,
                           ff_dim=ff_dim, leaf_dephase=p, seed=3)
    rng = np.random.default_rng(1)
    for w in layer.pff:  # nonzero feedforward weights, identical across layers built here
        if w is not None:
            w.assign(rng.normal(0, 0.3, size=w.shape).astype(np.float32))
    for w in layer.p0:  # leave the base generators at their seeded values; add a small perturbation on all
        w.assign(w + rng.normal(0, 0.05, size=w.shape).astype(np.float32))
    return layer


def dephase(x, p):
    """x: (B, chi, chi, L). numpy reference of (1 - p) rho + p diag(rho)."""
    x = x.numpy()
    d = np.einsum("bijl,ij->bijl", x, np.eye(CHI))
    return tf.constant((1 - p) * x + p * d, dtype=tf.complex64)


def test_p0_identical_outputs():
    for R, ff_dim in ((CHI, None), (CHI, 20), (2 * CHI, None)):
        x = pure_registers(B, CHI, 2**NL, 0)
        ref = make(None, R, ff_dim=ff_dim)
        for p in (0.0, None):
            new = make(p, R, ff_dim=ff_dim)
            assert new.leaf_dephase is None
            assert np.array_equal(ref(x).numpy(), new(x).numpy()), (R, ff_dim, p)
        assert ref.get_config()["leaf_dephase"] is None


def test_p1_equals_fully_dephased_input():
    for R in (CHI, 2 * CHI):
        x = pure_registers(B, CHI, 2**NL, 2)
        ref = make(None, R)
        new = make(1.0, R)
        xd = dephase(x, 1.0)
        # the dephased input is diagonal: all off-diagonal entries vanish
        off = xd.numpy() * (1 - np.eye(CHI))[None, :, :, None]
        assert np.abs(off).max() == 0
        err = np.abs(ref(xd).numpy() - new(x).numpy()).max()
        assert err < 1e-6, (R, err)
        # and it differs from the coherent input
        assert np.abs(ref(x).numpy() - new(x).numpy()).max() > 1e-4


def test_partial_dephasing_matches_reference_and_normalizes():
    for p in (0.25, 0.5, 0.75):
        for share in (True, False):
            x = pure_registers(B, CHI, 2**NL, 4)
            ref = make(None, 2 * CHI, share=share)
            new = make(p, 2 * CHI, share=share)
            out = new(x).numpy()
            err = np.abs(ref(dephase(x, p)).numpy() - out).max()
            assert err < 1e-6, (p, share, err)
            assert np.allclose(out.sum(-1), 1.0, atol=1e-4)
            assert out.min() > -1e-6


def test_config_roundtrip_and_validation():
    layer = make(0.5, CHI)
    assert layer.get_config()["leaf_dephase"] == 0.5
    assert cp.CPTPTNLayer.from_config(layer.get_config()).leaf_dephase == 0.5
    for bad in (-0.1, 1.5):
        try:
            cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=CHI, leaf_dephase=bad)
        except ValueError:
            continue
        raise AssertionError(bad)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
