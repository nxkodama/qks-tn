"""Test for the LS16Bc0f65 config (W26): parameter count within 2% of LK16B, default LS16Bc0 unchanged.

    .venv/bin/python qks-tn-library/tests/test_w26.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))
sys.path.insert(0, os.path.join(HERE, ".."))

import qks_tn_stoch as st  # noqa: E402
import run_ff_experiment as rfe  # noqa: E402


def n_params(name):
    c = dict(rfe.DEFAULTS, **rfe.CONFIGS[name])
    layer = st.StochTNLayer(
        chi=8, nlayers=6, mode=c["mode"], share_per_layer=c["share_per_layer"], kraus_rank=int(c["kraus_rank"]),
        coherent_layers={"full": 0, "c0": 1, "c2": 2, "c3": 3}[c["stoch"]],
        ff_dim=(int(c["ff_dim"]) if c["ff_dim"] else None), root_ff_dim=(int(c["root_ff_dim"]) if c["root_ff_dim"] else None),
        seed=0, init_scale=float(c["stoch_init_scale"]),
    )
    return sum(int(w.shape.num_elements()) for w in layer.trainable_weights)


def test_c0f65_params_within_2pct_of_lk16b():
    n = n_params("LS16Bc0f65")
    assert n == 2314239, n
    assert abs(n / 2314074 - 1) < 0.02


def test_c0_default_unchanged():
    assert n_params("LS16Bc0") == 1232895
    c = rfe.CONFIGS["LS16Bc0f65"]
    assert c["stoch"] == "c0" and c["ff_dim"] == 65 and c["root_ff_dim"] == 32 and c["kraus_rank"] == 16


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
