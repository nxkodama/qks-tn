"""Tests for the W63-MECH configs: LK12Bf55af, LU4Bf112(af). Default configs unchanged.

    .venv/bin/python qks-tn-library-W63/tests/test_w63.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))
sys.path.insert(0, os.path.join(HERE, ".."))

import qks_tn_cptp as cptp  # noqa: E402
import run_ff_experiment as rfe  # noqa: E402


def n_params(name, chi):
    c = dict(rfe.DEFAULTS, **rfe.CONFIGS[name])
    layer = cptp.CPTPTNLayer(chi=chi, nlayers=6, mode=c["mode"], share_per_layer=c["share_per_layer"],
                             kraus_rank=int(c["kraus_rank"]), ff_dim=(int(c["ff_dim"]) if c["ff_dim"] else None), seed=0)
    return int(layer.count_params())


def test_lk12bf55af_params_and_flip():
    n = n_params("LK12Bf55af", 8)
    assert n == 2302695, n
    assert abs(n / 2314074 - 1) < 0.02
    c = rfe.CONFIGS["LK12Bf55af"]
    assert c["augment"] == "flip" and c["kraus_rank"] == 12 and c["ff_dim"] == 55
    assert {k: v for k, v in c.items() if k != "augment"} == rfe.CONFIGS["LK12Bf55"]


def test_lu4bf112_is_unitary_at_chi4():
    c = rfe.CONFIGS["LU4Bf112af"]
    assert c["kraus_rank"] == 4 and c["ff_dim"] == 112 and c["augment"] == "flip"
    assert n_params("LU4Bf112af", 4) == n_params("LU4Bf112", 4) > 0


def test_chi_aliases_differ_only_in_chi():
    for a, b, chi in (("LK16Bx4af", "LK16Baf", 4), ("LS16Bc0f65x4af", "LS16Bc0f65af", 4),
                      ("LU8Bf112x4af", "LU8Bf112af", 4), ("LK16Bx16af", "LK16Baf", 16)):
        ca, cb = dict(rfe.CONFIGS[a]), dict(rfe.CONFIGS[b])
        assert ca.pop("chi") == chi and "chi" not in cb
        assert ca == cb, (a, b)
    assert rfe.CONFIGS["LU4Bf112af"]["chi"] == 4


def test_defaults_unchanged():
    assert n_params("LK16B", 8) == 2314074
    assert n_params("LU8Bf112", 8) == 2317770
    assert rfe.CONFIGS["LU8Bf112af"]["augment"] == "flip" and rfe.CONFIGS["LK16Baf"]["kraus_rank"] == 16


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
