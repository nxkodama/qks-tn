"""W62 tests: ff noise on the unitary control (LU8Bf112) and the classical hybrid (LS16Bc0f65), with flip.

    .venv/bin/python qks-tn-library-W62/tests/test_w62.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))
sys.path.insert(0, os.path.join(HERE, ".."))

import qks_tn_stoch as st  # noqa: E402
import run_ff_experiment as rfe  # noqa: E402
from test_stoch import CHI, NL, mixed_registers, randomize  # noqa: E402


def _cfg(name):
    return rfe.make_cfg(**dict(rfe.CONFIGS[name], task="cifar10", seed=0, epochs=1))


def test_w62_configs_differ_only_in_noise_and_augment():
    for base in ("LU8Bf112", "LS16Bc0f65"):
        plain = _cfg(base)
        af = _cfg(base + "af")
        for tag, val in (("n01af", 0.01), ("n03af", 0.03)):
            c = _cfg(base + tag)
            assert c["ff_noise"] == val and c["augment"] == "flip"
            assert c["ff_dropout"] is None and c["ff_mix"] is None
            assert af["ff_noise"] is None and plain["augment"] != "flip"
            d = {k for k in c if c[k] != af.get(k)}
            assert d <= {"ff_noise", "name"}, d
    assert _cfg("LU8Bf112n01af")["kraus_rank"] == 8 and _cfg("LU8Bf112n01af")["ff_dim"] == 112
    assert _cfg("LS16Bc0f65n01af")["stoch"] == "c0" and _cfg("LS16Bc0f65n01af")["ff_dim"] == 65
    assert _cfg("LS16Bc0f65n01af")["name"] != _cfg("LS16Bc0f65af")["name"]


def test_stoch_ff_noise_training_only():
    kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, ff_dim=10, coherent_layers=1, seed=3)
    lay = st.StochTNLayer(**kw, ff_noise=0.05)
    plain = st.StochTNLayer(**kw)
    assert plain.ff_noise == 0.0 and lay.ff_noise == 0.05
    randomize(lay, 0)
    randomize(plain, 0)
    for a, b in zip(lay.weights, plain.weights):
        b.assign(a)
    x = mixed_registers(8, CHI, 2**NL, 40)
    clean = plain(x).numpy()
    assert np.array_equal(lay(x, training=False).numpy(), clean)
    assert np.array_equal(lay(x).numpy(), clean)
    tr = lay(x, training=True).numpy()
    assert np.abs(tr - clean).max() > 1e-5
    # outputs stay a probability distribution under noise (rows are stochastic maps)
    assert np.allclose(tr.sum(axis=-1), 1.0, atol=1e-4)
    # noise 0 with training=True is the clean output
    assert np.array_equal(plain(x, training=True).numpy(), clean)
    assert lay.get_config()["ff_noise"] == 0.05
    try:
        st.StochTNLayer(**kw, ff_noise=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative ff_noise accepted")


def test_unitary_noise_layer_training_only():
    import qks_tn_cptp as cp

    kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, ff_dim=10, init="random", seed=41)
    lay = cp.CPTPTNLayer(**kw, ff_noise=0.01)
    plain = cp.CPTPTNLayer(**kw)
    x = mixed_registers(8, CHI, 2**NL, 40)
    for l, w in enumerate(lay.pff):
        if w is not None:
            w.assign(np.full(w.shape, 0.5, np.float32))
            plain.pff[l].assign(np.full(w.shape, 0.5, np.float32))
    assert np.array_equal(lay(x, training=False).numpy(), plain(x).numpy())
    assert np.abs(lay(x, training=True).numpy() - plain(x).numpy()).max() > 1e-6


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            n += 1
            print("ok", k)
    print(n, "tests passed")
