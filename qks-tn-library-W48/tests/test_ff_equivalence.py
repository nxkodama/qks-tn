"""Equivalence and sanity tests for ``qks_tn_ff``.

Run with the project venv, either as a script::

    .venv/bin/python qks-tn-library/tests/test_ff_equivalence.py

or with pytest if installed.  The tests compare the new batched sweep against the
original ``qks_tn.TNLayer`` (tensornetwork contraction) and check the feedforward modes
reduce to the baseline when their coupling parameters vanish.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import tensornetwork as tn  # noqa: E402

tn.set_default_backend("tensorflow")

import qks_tn as old  # noqa: E402
import qks_tn_ff as ff  # noqa: E402

NEP, CHI, P, SIGMA, NL = 128, 4, 784, 0.125, 6
D, K = CHI**2, CHI**4 - 1

_FIXTURE = {}


def fixture(nb: int = 8, seed: int = 0):
    """Random inputs, a QKS layer, its density matrices, and the old TNLayer + path."""
    key = (nb, seed)
    if key not in _FIXTURE:
        tf.keras.utils.set_random_seed(seed)
        X = tf.random.uniform((nb, P))
        QKS = old.FeatureEncodingLayer(NEP, CHI, P, SIGMA)
        rho = QKS.call(X)
        uni_array = tf.constant(np.zeros([CHI] * 4 + [NL]), dtype="complex64")
        obs = [rho[0, :, :, j].numpy() for j in range(NEP // 2)]
        nodes_set, edge_order = old.construct_dttn(CHI, uni_array, obs, NEP)
        _, path = old.greedy(nodes_set, output_edge_order=edge_order)
        old_layer = old.TNLayer(CHI, NL, path)
        _FIXTURE[key] = dict(X=X, QKS=QKS, rho=rho, old_layer=old_layer, path=path)
    return _FIXTURE[key]


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _shared_params0(fx):
    return fx["old_layer"].params.numpy()[:, :NL]


# ---------------------------------------------------------------------------------------
def test_a_baseline_forward_matches_old_layer():
    fx = fixture()
    new = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", init_params0=_shared_params0(fx))
    out_old = fx["old_layer"](fx["rho"]).numpy()
    out_new = new(fx["rho"]).numpy()
    diff = _maxabs(out_old, out_new)
    assert out_new.shape == (8, D)
    assert diff < 1e-5, f"forward mismatch {diff:.3e}"


def test_a2_baseline_gradient_matches_old_layer():
    fx = fixture()
    new = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", init_params0=_shared_params0(fx))
    w = tf.constant(np.random.default_rng(1).normal(size=(8, D)).astype("float32"))
    with tf.GradientTape() as tape:
        loss_old = tf.reduce_sum(fx["old_layer"](fx["rho"]) * w)
    g_old = tape.gradient(loss_old, fx["old_layer"].params).numpy()
    with tf.GradientTape() as tape:
        loss_new = tf.reduce_sum(new(fx["rho"]) * w)
    g_new = tape.gradient(loss_new, new.params0).numpy()
    assert g_old.shape == (K, 2**NL - 1) and g_new.shape == (K, NL)
    assert np.all(g_old[:, NL:] == 0.0), "old layer has non-zero gradient on dead columns"
    rel = _maxabs(g_old[:, :NL], g_new) / (np.max(np.abs(g_old[:, :NL])) + 1e-12)
    assert rel < 1e-4, f"gradient mismatch rel {rel:.3e}"


def test_b_modeB_zero_coupling_equals_baseline():
    fx = fixture()
    p0 = _shared_params0(fx)
    base = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", init_params0=p0)
    modeB = ff.FeedforwardTNLayer(CHI, NL, mode="B", init_params0=p0)
    modeB_shots = ff.FeedforwardTNLayer(CHI, NL, mode="B", init_params0=p0, shots=64)
    ob = base(fx["rho"]).numpy()
    assert _maxabs(ob, modeB(fx["rho"]).numpy()) < 1e-6
    assert _maxabs(ob, modeB(fx["rho"], training=True).numpy()) < 1e-6
    assert _maxabs(ob, modeB_shots(fx["rho"], training=False).numpy()) < 1e-6
    # zero coupling: sampled p changes nothing because H_k = 0
    assert _maxabs(ob, modeB_shots(fx["rho"], training=True).numpy()) < 1e-6
    assert modeB.params_ff.shape == (K, NL - 1, 2 * CHI)


def test_c_modeA_zero_deltas_equals_baseline():
    fx = fixture()
    p0 = _shared_params0(fx)
    base = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", init_params0=p0)
    modeA = ff.FeedforwardTNLayer(CHI, NL, mode="A", init_params0=p0)
    diff = _maxabs(base(fx["rho"]).numpy(), modeA(fx["rho"]).numpy())
    assert diff < 1e-6, f"mode A mismatch {diff:.3e}"
    assert modeA.params_ffA.shape == (K, NL - 1, CHI**2)


def _randomise_coupling(layer, std=0.3, seed=2):
    rng = np.random.default_rng(seed)
    if layer.params_ff is not None:
        layer.params_ff.assign(rng.normal(scale=std, size=layer.params_ff.shape).astype("float32"))
    if layer.params_ffA is not None:
        layer.params_ffA.assign(rng.normal(scale=std, size=layer.params_ffA.shape).astype("float32"))


def test_d_normalisation_all_modes():
    fx = fixture()
    p0 = _shared_params0(fx)
    for mode in ("baseline", "A", "B"):
        for share in (True, False):
            kw = {}
            if share:
                kw["init_params0"] = p0
            layer = ff.FeedforwardTNLayer(CHI, NL, mode=mode, share_per_layer=share, seed=3, **kw)
            _randomise_coupling(layer)
            for training in (False, True):
                out = layer(fx["rho"], training=training).numpy()
                assert np.all(np.isfinite(out))
                assert np.max(np.abs(out.sum(-1) - 1.0)) < 1e-5, (mode, share, training)
                assert out.min() > -1e-6, (mode, share, training)


def test_e_unitarity():
    fx = fixture()
    eye = np.eye(D)
    for mode in ("baseline", "A", "B"):
        layer = ff.FeedforwardTNLayer(CHI, NL, mode=mode, seed=4)
        _randomise_coupling(layer)
        for l in range(NL):
            n = layer.n_vertices[l]
            ffin = tf.constant(np.random.default_rng(l).dirichlet(np.ones(CHI), size=(8, n, 2)).reshape(8, n, 2 * CHI).astype("float32"))
            U = layer.build_unitaries(l, ffin if mode == "B" else None).numpy()
            Ur = U.reshape(-1, D, D)
            err = np.max(np.abs(np.conj(np.transpose(Ur, (0, 2, 1))) @ Ur - eye))
            assert err < 1e-5, (mode, l, err)


def test_f_basis_matches_scatter_construction():
    """Copy of the Hermitian assembly in ``TNLayer.call`` for one parameter column."""
    rng = np.random.default_rng(5)
    params = rng.normal(size=(K, 1)).astype("float32")
    d, n = D, D * (D - 1) // 2
    ind_re = np.stack(np.triu_indices(d, 1)).T
    ind_im = np.stack(np.triu_indices(d, 1)).T
    ind_diag = np.stack(np.diag_indices(d - 1)).T
    H = tf.constant(np.zeros((d, d), dtype="complex64"))
    H = tf.tensor_scatter_nd_add(H, ind_re, tf.cast(params[:n, 0], "complex64"))
    H = H + tf.transpose(H)
    H = tf.tensor_scatter_nd_add(H, ind_diag, tf.cast(params[2 * n :, 0], "complex64"))
    H = tf.tensor_scatter_nd_add(H, np.array([[d - 1, d - 1]]), [-tf.math.reduce_sum(tf.cast(params[2 * n :, 0], "complex128"))])
    im = tf.constant(np.zeros((d, d), dtype="complex64"))
    im = tf.tensor_scatter_nd_add(H, ind_im, tf.cast(params[n : 2 * n, 0], "complex64"))
    im = im - tf.transpose(im)
    H = (H + 1j * im).numpy()
    H_new = ff.params_to_hermitian(tf.constant(params), tf.constant(ff.hermitian_basis(d))).numpy()[0]
    assert _maxabs(H, H_new) < 1e-6
    assert _maxabs(H_new, np.conj(H_new.T)) < 1e-7
    assert abs(np.trace(H_new)) < 1e-5


def test_g_fast_feature_encoding_matches_original():
    tf.keras.utils.set_random_seed(7)
    slow = old.FeatureEncodingLayer(NEP, CHI, P, SIGMA)
    tf.keras.utils.set_random_seed(7)
    fast = ff.FastFeatureEncodingLayer(NEP, CHI, P, SIGMA)
    assert _maxabs(slow.W.numpy(), fast.W.numpy()) == 0.0 and _maxabs(slow.b.numpy(), fast.b.numpy()) == 0.0
    X = tf.random.uniform((8, P))
    diff = _maxabs(slow.call(X).numpy(), fast(X).numpy())
    assert diff < 1e-6, f"fast QKS mismatch {diff:.3e}"


def test_h_shot_sampling():
    tf.random.set_seed(11)
    p = tf.Variable(np.random.default_rng(11).dirichlet(np.ones(CHI), size=(6, 5)).astype("float32"))
    M = 32
    w = tf.constant(np.random.default_rng(12).normal(size=p.shape).astype("float32"))
    with tf.GradientTape() as tape:
        q = ff.sample_shots(p, M)
        loss = tf.reduce_sum(q * w)
    g = tape.gradient(loss, p).numpy()
    assert _maxabs(g, w.numpy()) < 1e-6, "straight-through gradient must equal d(sum p*w)/dp = w"
    counts = q.numpy() * M
    assert np.allclose(counts, np.round(counts), atol=1e-4) and np.allclose(counts.sum(-1), M, atol=1e-3)
    draws = np.mean([ff.sample_shots(p, M).numpy() for _ in range(400)], axis=0)
    assert _maxabs(draws, p.numpy()) < 0.03, "sampled statistics must be unbiased"


def test_i_modeB_finite_and_finite_difference():
    fx = fixture()
    layer = ff.FeedforwardTNLayer(CHI, NL, mode="B", init_params0=_shared_params0(fx))
    _randomise_coupling(layer)
    w = tf.constant(np.random.default_rng(13).normal(size=(8, D)).astype("float32"))

    def loss_fn():
        return tf.reduce_sum(layer(fx["rho"]) * w)

    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [layer.params0, layer.params_ff])
    assert all(np.all(np.isfinite(g.numpy())) for g in grads)
    g_ff = grads[1].numpy()
    rng = np.random.default_rng(14)
    eps = 1e-2
    for _ in range(3):
        idx = tuple(int(rng.integers(s)) for s in layer.params_ff.shape)
        base = layer.params_ff.numpy()
        plus = base.copy()
        plus[idx] += eps
        minus = base.copy()
        minus[idx] -= eps
        layer.params_ff.assign(plus)
        lp = float(loss_fn())
        layer.params_ff.assign(minus)
        lm = float(loss_fn())
        layer.params_ff.assign(base)
        fd = (lp - lm) / (2 * eps)
        an = g_ff[idx]
        assert abs(fd - an) <= 5e-2 * max(abs(an), abs(fd), 1e-2), (idx, fd, an)


def test_j_per_vertex_baseline_with_tiled_params_equals_shared():
    fx = fixture()
    p0 = _shared_params0(fx)
    shared = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", init_params0=p0)
    tiled = np.concatenate([np.repeat(p0[:, l : l + 1], 2 ** (NL - 1 - l), axis=1) for l in range(NL)], axis=1)
    assert tiled.shape == (K, 2**NL - 1)
    pv = ff.FeedforwardTNLayer(CHI, NL, mode="baseline", share_per_layer=False, init_params0=tiled)
    diff = _maxabs(shared(fx["rho"]).numpy(), pv(fx["rho"]).numpy())
    assert diff < 1e-6, f"per-vertex mismatch {diff:.3e}"
    assert pv.count_params() == K * (2**NL - 1)


def test_k_param_counts():
    assert ff.FeedforwardTNLayer(CHI, NL, mode="baseline").count_params() == K * NL
    assert ff.FeedforwardTNLayer(CHI, NL, mode="B").count_params() == K * NL + K * (NL - 1) * 2 * CHI
    assert ff.FeedforwardTNLayer(CHI, NL, mode="A").count_params() == K * NL + K * (NL - 1) * CHI**2
    assert ff.FeedforwardTNLayer(CHI, NL, mode="B", ff_layers=[NL - 1]).count_params() == K * NL + K * 2 * CHI


def test_l_keras_model_trains_one_step():
    fx = fixture()
    for mode in ("baseline", "A", "B"):
        tf.keras.utils.set_random_seed(0)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(P,)),
                ff.FastFeatureEncodingLayer(NEP, CHI, P, SIGMA),
                ff.FeedforwardTNLayer(CHI, NL, mode=mode, seed=0),
                ff.ConstMul(),
            ]
        )
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        y = np.arange(8) % 10
        h = model.fit(fx["X"].numpy(), y.astype("float32"), batch_size=8, epochs=2, verbose=0)
        assert np.all(np.isfinite(h.history["loss"])), mode
        assert h.history["loss"][1] < h.history["loss"][0] + 1e-3, (mode, h.history["loss"])


# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f}s): {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
