"""W17 tests: head_scale option, matched-root configs.

    .venv/bin/python qks-tn-library/tests/test_w17.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import run_ff_experiment as rfe  # noqa: E402


def _cfg(name, **kw):
    ov = dict(rfe.CONFIGS[name])
    ov.update(task="timn100", seed=0, epochs=1, chi=8, **kw)
    return rfe.make_cfg(**ov)


def _build(cfg):
    return rfe.build_model(cfg, steps_per_epoch=10, p=3072)


def test_default_name_and_model_unchanged():
    c = _cfg("LU8Bf132R131")
    assert "hs" not in c["name"] and c.get("head_scale") is None
    model, _, _, counts = _build(c)
    assert not any(l.name == "head_scale" for l in model.layers)
    assert counts["head_kind"] == "linear" and counts["head"] == 64 * 100 + 100


def test_head_scale_layer_only_when_set_and_has_no_parameters():
    c = _cfg("LU8Bf132R131H64")
    assert c["name"].endswith("hs64.0_s0") or "_hs64.0_" in c["name"]
    model, _, _, counts = _build(c)
    names = [l.name for l in model.layers]
    assert names[-2:] == ["head_scale", "linear_head"], names
    assert counts["head"] == 6500 and counts["n_outcomes"] == 64
    base = _build(_cfg("LU8Bf132R131"))[3]
    assert base["trainable"] == counts["trainable"] and base["tn"] == counts["tn"]


def test_head_scale_multiplies_the_readout_input():
    import tensorflow as tf

    m1, *_ = _build(_cfg("LU8Bf132R131H1"))
    m4, *_ = _build(_cfg("LU8Bf132R131H4"))
    for a, b in zip(m1.weights, m4.weights):
        b.assign(a)
    x = tf.random.stateless_uniform((2, 3072), seed=[1, 2])
    z = tf.Variable(m1.layers[-1].kernel)  # same Dense weights
    out1 = m1(x).numpy()
    out4 = m4(x).numpy()
    feats = tf.keras.Sequential(m1.layers[:-1])(x)  # probabilities (gain 1)
    k, b = m1.layers[-1].kernel.numpy(), m1.layers[-1].bias.numpy()
    np.testing.assert_allclose(out1, feats.numpy() @ k + b, atol=1e-5)
    np.testing.assert_allclose(out4, 4.0 * feats.numpy() @ k + b, atol=1e-4)


def test_root_parameters_matched_across_the_three_arms():
    root = {}
    for n in ("LK16BW16FH1", "LU8Bf132R131H1", "LS16Bc0R131H1"):
        model, _, tn, counts = _build(_cfg(n))
        assert counts["n_outcomes"] == 64
        w = [v for v in tn.weights if "root" in v.name.lower() or v.name.endswith("l5")]
        root[n] = sum(int(np.prod(v.shape)) for v in tn.weights if "l5" in v.name)
    print("root params", root)
    vals = list(root.values())
    assert max(vals) / min(vals) < 1.001, root
    assert abs(root["LK16BW16FH1"] - 540639) <= 200


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
