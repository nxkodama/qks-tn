"""W49: configs LK12Bff32, LK15Bff32 (feedforward width of LK16B, only the Kraus rank changes) and the existing configs.

    .venv/bin/python qks-tn-library-W49/tests/test_w49.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import run_ff_experiment as rfe  # noqa: E402


def _tn(name):
    cfg = rfe.make_cfg(**dict(rfe.CONFIGS[name], task="cifar10", lr=0.002, wd=0.0004, chi=8))
    _, _, tn, counts = rfe.build_model(cfg, 100, p=3072, n_classes=10)
    return cfg, tn, counts["tn"]


def test_w49_configs_differ_from_lk16b_only_in_rank():
    base = rfe.CONFIGS["LK16B"]
    for name, R in (("LK12Bff32", 12), ("LK15Bff32", 15)):
        c = rfe.CONFIGS[name]
        assert c["kraus_rank"] == R and c["ff_dim"] == 32
        assert {k: v for k, v in c.items() if k not in ("kraus_rank", "ff_dim")} == {k: v for k, v in base.items() if k != "kraus_rank"}
        cfg, tn, n = _tn(name)
        assert list(tn.ff_dims) == [32] * 6 and tn.Rroot == 8 and tn.R == R
    assert rfe.CONFIGS["LK15Bff32af"] == dict(rfe.CONFIGS["LK15Bff32"], augment="flip")


def test_w49_parameter_counts_and_existing_configs_unchanged():
    counts = {n: _tn(n)[2] for n in ("LK16B", "LK12Bff32", "LK15Bff32", "LK12B")}
    assert counts == {"LK16B": 2314074, "LK12Bff32": 1360730, "LK15Bff32": 2050202, "LK12B": 1033090}, counts
    assert rfe.CONFIGS["LK12B"]["kraus_rank"] == 12 and "ff_dim" not in rfe.CONFIGS["LK12B"]
    assert rfe.CONFIGS["LK16B"] == dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
