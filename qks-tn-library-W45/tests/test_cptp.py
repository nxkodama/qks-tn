"""Tests for ``qks_tn_cptp.CPTPTNLayer``.

    .venv/bin/python qks-tn-library/tests/test_cptp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import qks_tn_cptp as cp  # noqa: E402
import qks_tn_ff as ff  # noqa: E402

CHI, NL, B = 2, 3, 5


def pure_registers(batch, chi, nleaves, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(batch, nleaves, chi)) + 1j * rng.normal(size=(batch, nleaves, chi))
    psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
    rho = np.einsum("bli,blj->bijl", psi, psi.conj())  # (B, chi, chi, L)
    return tf.constant(rho, dtype=tf.complex64)


def test_rank_chi_equals_unitary_layer():
    for mode in ("baseline", "B"):
        for share in (True, False):
            x = pure_registers(B, CHI, 2**NL, 0)
            ref = ff.FeedforwardTNLayer(chi=CHI, nlayers=NL, mode=mode, share_per_layer=share, seed=3)
            new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode=mode, share_per_layer=share, kraus_rank=CHI, seed=3)
            if mode == "B":  # nonzero feedforward weights, identical in both layers
                rng = np.random.default_rng(1)
                w = rng.normal(0, 0.3, size=ref.params_ff.shape).astype(np.float32)
                ref.params_ff.assign(w)
                off = 0
                for l in range(1, NL):
                    nu = 1 if share else ref.n_vertices[l]
                    new.pff[l].assign(w[:, off : off + nu])
                    off += nu
            err = np.abs(ref(x).numpy() - new(x).numpy()).max()
            assert err < 1e-5, (mode, share, err)


def test_embedded_init_reproduces_unitary_at_higher_rank():
    x = pure_registers(B, CHI, 2**NL, 1)
    ref = ff.FeedforwardTNLayer(chi=CHI, nlayers=NL, mode="baseline", share_per_layer=False, seed=4)
    for R in (4, 8):
        new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=R, share_per_layer=False, init_noise=0.0, seed=4)
        err = np.abs(ref(x).numpy() - new(x).numpy()).max()
        assert err < 1e-5, (R, err)


def test_trace_preserving_and_isometry():
    for mode in ("baseline", "B"):
        x = pure_registers(B, CHI, 2**NL, 2)
        new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode=mode, kraus_rank=6, init="random", init_noise=0.5, seed=5)
        if mode == "B":
            for w in new.pff:
                if w is not None:
                    w.assign(np.random.default_rng(0).normal(0, 0.3, size=w.shape).astype(np.float32))
        assert np.allclose(new(x).numpy().sum(-1), 1.0, atol=1e-4)
        for l in range(NL):
            V = new.isometry(l).numpy()[0]
            assert np.allclose(V @ V.conj().T, np.eye(CHI**2), atol=1e-4), l


def test_gradients_nonzero_in_extra_environment_block():
    x = pure_registers(B, CHI, 2**NL, 3)
    new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=4, init_noise=0.0, seed=6)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(new(x) * tf.random.stateless_normal([B, CHI**2], seed=[1, 2]))
    g = tape.gradient(loss, new.p0[0])
    assert g is not None and np.isfinite(g.numpy()).all() and np.abs(g.numpy()).max() > 0


def test_finite_difference():
    x = pure_registers(3, CHI, 2**NL, 4)
    new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=4, init="random", init_noise=0.3, seed=7)
    tgt = tf.random.stateless_normal([3, CHI**2], seed=[3, 4])

    def f():
        return tf.reduce_sum(new(x) * tgt)

    with tf.GradientTape() as tape:
        loss = f()
    g = tape.gradient(loss, new.p0[0]).numpy()
    idx = (5, 0)
    eps = 1e-2
    w = new.p0[0]
    base = w.numpy().copy()
    plus = base.copy(); plus[idx] += eps
    minus = base.copy(); minus[idx] -= eps
    w.assign(plus); fp = f().numpy()
    w.assign(minus); fm = f().numpy()
    w.assign(base)
    fd = (fp - fm) / (2 * eps)
    assert abs(fd - g[idx]) < 5e-3 * max(1.0, abs(fd)), (fd, g[idx])


def test_wide_root_embedded_reproduces_unitary_readout():
    x = pure_registers(B, CHI, 2**NL, 5)
    ref = ff.FeedforwardTNLayer(chi=CHI, nlayers=NL, mode="baseline", share_per_layer=True, seed=8)
    new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=4, root_rank=6, init_noise=0.0, seed=8)
    assert new.n_outcomes == CHI * 6
    out = new(x).numpy()
    assert out.shape == (B, CHI * 6)
    assert np.allclose(out.sum(-1), 1.0, atol=1e-4)
    grid = out.reshape(B, CHI, 6)  # keep-major: index k * Rroot + e
    assert np.allclose(grid[:, :, CHI:], 0.0, atol=1e-5)
    assert np.abs(grid[:, :, :CHI].reshape(B, -1) - ref(x).numpy()).max() < 1e-5


def test_wide_root_trace_preserving_with_feedforward():
    x = pure_registers(B, CHI, 2**NL, 6)
    new = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, root_rank=8, init="random", seed=9)
    for w in new.pff:
        if w is not None:
            w.assign(np.random.default_rng(1).normal(0, 0.3, size=w.shape).astype(np.float32))
    out = new(x).numpy()
    assert out.shape == (B, CHI * 8) and np.allclose(out.sum(-1), 1.0, atol=1e-4)
    V = new.isometry(NL - 1).numpy()[0]
    assert np.allclose(V @ V.conj().T, np.eye(CHI**2), atol=1e-4)


def test_haar_random_init_differs_from_embedding_and_is_isometric():
    x = pure_registers(B, CHI, 2**NL, 7)
    emb = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=6, init_noise=0.0, seed=10)
    rnd = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=6, init="random", seed=10)
    assert np.abs(emb(x).numpy() - rnd(x).numpy()).max() > 1e-3
    for l in range(NL):
        V = rnd.isometry(l).numpy()[0]
        assert np.allclose(V @ V.conj().T, np.eye(CHI**2), atol=1e-4)
    # the root keeps the baseline unitary in both
    assert np.abs(emb.isometry(NL - 1).numpy() - rnd.isometry(NL - 1).numpy()).max() < 1e-6


def test_root_fold_keeps_unitary_readout_width():
    x = pure_registers(B, CHI, 2**NL, 8)
    ref = ff.FeedforwardTNLayer(chi=CHI, nlayers=NL, mode="baseline", share_per_layer=True, seed=11)
    emb = cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=4, root_rank=8, root_fold=True, init_noise=0.0, seed=11)
    assert emb.n_outcomes == CHI**2
    assert np.abs(emb(x).numpy() - ref(x).numpy()).max() < 1e-5
    rnd = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, root_rank=8, root_fold=True, init="random", seed=12)
    out = rnd(x).numpy()
    assert out.shape == (B, CHI**2) and np.allclose(out.sum(-1), 1.0, atol=1e-4)
    # folding is a sum over outcomes of the unfolded model with the same parameters
    wide = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, root_rank=8, init="random", seed=12)
    for a, b in zip(wide.weights, rnd.weights):
        a.assign(b)
    full = wide(x).numpy().reshape(B, CHI, 8)  # (k, e), e = a * chi + b
    folded = full.reshape(B, CHI, 8 // CHI, CHI).sum(2)  # (k, b)
    assert np.abs(folded.reshape(B, -1) - out).max() < 1e-5


def test_ff_dim_default_unchanged_and_extension_runs():
    x = pure_registers(B, CHI, 2**NL, 9)
    a = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, seed=13)
    b = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, ff_dim=8, seed=13)
    assert np.abs(a(x).numpy() - b(x).numpy()).max() < 1e-6
    # unitary control with extended features: more ff parameters, no environment, trace preserving
    u = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, ff_dim=20, root_ff_dim=30, seed=13)
    assert u.pff[1].shape[-1] == 20 and u.pff[NL - 1].shape[-1] == 30
    for l, w in enumerate(u.pff):
        if w is not None:
            w.assign(np.random.default_rng(l).normal(0, 0.3, size=w.shape).astype(np.float32))
    out = u(x).numpy()
    assert np.allclose(out.sum(-1), 1.0, atol=1e-4)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(u(x) * tf.random.stateless_normal([B, CHI**2], seed=[5, 6]))
    g = tape.gradient(loss, u.pff[NL - 1])
    assert g is not None and np.abs(g.numpy()[..., 8:]).max() > 0  # extra (monomial) features receive gradient
    f = tf.constant(np.random.default_rng(3).uniform(size=(2, 3, 4)), dtype=tf.float32)
    for n_total in (14, 30):  # degree 2 only, then into degree 3
        ref = np.stack([np.prod(f.numpy()[..., list(c)], axis=-1) for c in cp.monomial_indices(4, n_total)], axis=-1)
        assert np.abs(cp.monomial_features(f, n_total).numpy() - ref).max() < 1e-6
    # baseline monomials: base (4) + degree 2 (10) = 14 features
    assert len(cp.monomial_indices(4, 14)) == 10


def _set_pff(layer, scale=0.3, seed=0):
    for l, w in enumerate(layer.pff):
        if w is not None:
            w.assign(np.random.default_rng(seed + l).normal(0, scale, size=w.shape).astype(np.float32))


def test_meas_bases_default_unchanged_and_k1_identical():
    x = pure_registers(B, CHI, 2**NL, 10)
    ref = ff.FeedforwardTNLayer(chi=CHI, nlayers=NL, mode="B", share_per_layer=True, seed=14)
    a = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, seed=14)
    k1 = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, meas_bases=1, seed=14)
    w = np.random.default_rng(1).normal(0, 0.3, size=ref.params_ff.shape).astype(np.float32)
    ref.params_ff.assign(w)
    off = 0
    for l in range(1, NL):
        a.pff[l].assign(w[:, off : off + 1])
        k1.pff[l].assign(w[:, off : off + 1])
        off += 1
    assert [v.name for v in a.weights] == [v.name for v in k1.weights]
    assert not any("mb" in v.name for v in a.weights)
    assert a.get_config()["meas_bases"] is None
    assert np.abs(a(x).numpy() - ref(x).numpy()).max() < 1e-5  # default reproduces the unitary layer
    assert np.array_equal(a(x).numpy(), k1(x).numpy())  # K = 1 is bit-identical to the default
    # also at R > chi
    r1 = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, init="random", seed=15)
    r2 = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, init="random", meas_bases=1, seed=15)
    _set_pff(r1)
    _set_pff(r2)
    assert np.array_equal(r1(x).numpy(), r2(x).numpy())


def test_meas_bases_feature_count_and_trace_preserving():
    x = pure_registers(B, CHI, 2**NL, 11)
    for share in (True, False):
        for K in (2, 3):
            m = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, meas_bases=K, share_per_layer=share, seed=16)
            for l in range(1, NL):  # each child passes K * chi features, two children per vertex
                assert m.pff[l].shape[-1] == 2 * K * CHI and m.ff_dims[l] == 2 * K * CHI
            assert m.pmb[NL - 1] is None and all(m.pmb[l] is not None for l in range(NL - 1))
            _set_pff(m)
            out = m(x).numpy()
            assert out.shape == (B, CHI**2) and np.allclose(out.sum(-1), 1.0, atol=1e-4)
            # each basis probability vector of a child sums to one; basis 0 matches the computational basis
            nv = m.n_vertices[0]
            sig = np.random.default_rng(2).normal(size=(B, nv, CHI, CHI)) + 1j * np.random.default_rng(3).normal(size=(B, nv, CHI, CHI))
            sig = sig @ sig.conj().swapaxes(-1, -2)
            sig /= np.trace(sig, axis1=-2, axis2=-1)[..., None, None].real
            p = m.meas_probs(0, tf.constant(sig, dtype=tf.complex64)).numpy().reshape(B, nv, K, CHI)
            assert np.allclose(p.sum(-1), 1.0, atol=1e-4)
            assert np.allclose(p[:, :, 0], np.real(np.diagonal(sig, axis1=-2, axis2=-1)), atol=1e-5)
            assert (p >= -1e-6).all()
            for l in range(NL):
                V = m.isometry(l).numpy()[0]
                assert np.allclose(V @ V.conj().T, np.eye(CHI**2), atol=1e-4)


def test_meas_bases_extra_basis_unitaries_get_gradients_and_change_output():
    x = pure_registers(B, CHI, 2**NL, 12)
    m = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, meas_bases=3, seed=17)
    _set_pff(m)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(m(x) * tf.random.stateless_normal([B, CHI**2], seed=[7, 8]))
    grads = tape.gradient(loss, m.pmb[: NL - 1])
    for l, g in enumerate(grads):
        assert g is not None and np.isfinite(g.numpy()).all() and np.abs(g.numpy()).max() > 1e-6, l
    # the extra bases enter the prediction: changing one changes the output
    k1 = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=CHI, seed=17)
    assert len(k1.weights) < len(m.weights)
    before = m(x).numpy()
    m.pmb[0].assign(m.pmb[0].numpy() + 0.3)
    assert np.abs(m(x).numpy() - before).max() > 1e-4
    # a basis unitary stays unitary after a parameter step
    U = m.meas_unitaries(0).numpy()
    assert np.allclose(U @ np.conj(np.swapaxes(U, -1, -2)), np.eye(CHI), atol=1e-4)


def test_meas_bases_rejects_incompatible_options():
    for kw in (dict(mode="baseline"), dict(ff_dim=40), dict(ff_width=4)):
        kw.setdefault("mode", "B")
        try:
            cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=CHI, meas_bases=2, **kw)
        except ValueError:
            continue
        raise AssertionError(kw)


def _two_layer_manual(layer, x, cut):
    """Reference forward of a 2-layer mode-B layer (child layer 0, root layer 1) written out by hand.

    ``cut=True`` wraps the child's feedforward features in ``tf.stop_gradient`` before they enter the root isometry.
    ``layer`` must have ``ff_stop_grad = False`` so that the cut is only the one applied here.
    """
    chi, d, R = layer.chi, layer.d, layer.R
    msgs = tf.transpose(x, [0, 3, 1, 2])
    outs = None
    ff_input = None
    for l in range(2):
        n = layer.n_vertices[l]
        Rl = R if l == 0 else layer.Rroot
        rho = tf.einsum("nvim,nvjo->nvijmo", msgs[:, 0::2], msgs[:, 1::2])
        rho = tf.reshape(rho, [-1, n, d, d])
        V = layer.isometry(l, ff_input)
        sigma = tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))
        diag = tf.math.real(tf.linalg.diag_part(sigma))
        if l == 0:
            s6 = tf.reshape(sigma, [-1, n, Rl, chi, Rl, chi])
            msgs = tf.einsum("nvcdCe,cC->nvde", s6, tf.eye(Rl, dtype=sigma.dtype))
            p = tf.reduce_sum(tf.reshape(diag, [-1, n, Rl, chi]), axis=-1)
            ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)
            if cut:
                ff_input = tf.stop_gradient(ff_input)
        else:
            outs = tf.reshape(tf.transpose(tf.reshape(diag, [-1, Rl, chi]), [0, 2, 1]), [-1, layer.n_outcomes])
    return outs


def test_ff_stop_grad_blocks_gradient_through_features_only():
    x = pure_registers(B, CHI, 4, 20)
    kw = dict(chi=CHI, nlayers=2, mode="B", kraus_rank=4, init="random", seed=21)
    plain = cp.CPTPTNLayer(**kw)
    sg = cp.CPTPTNLayer(**kw, ff_stop_grad=True)
    _set_pff(plain, 0.5)
    _set_pff(sg, 0.5)
    assert sg.get_config()["ff_stop_grad"] is True and plain.get_config()["ff_stop_grad"] is False
    assert sg.count_params() == plain.count_params()
    # forward output identical
    assert np.array_equal(plain(x).numpy(), sg(x).numpy())
    w = tf.constant(np.random.default_rng(5).normal(size=(B, plain.n_outcomes)).astype(np.float32))

    def grads(fn, layer):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(w * fn())
        return tape.gradient(loss, [layer.p0[0], layer.p0[1], layer.pff[1]])

    g_plain = grads(lambda: plain(x), plain)
    g_sg = grads(lambda: sg(x), sg)
    # the by-hand forward reproduces the layer, so the by-hand cut is a trustworthy reference
    assert np.abs(_two_layer_manual(plain, x, cut=False).numpy() - plain(x).numpy()).max() < 1e-5
    g_cut = grads(lambda: _two_layer_manual(plain, x, cut=True), plain)
    # child parameters: stop-gradient equals "features cut by hand"; it differs from the full gradient
    child_cut = np.abs(g_sg[0].numpy() - g_cut[0].numpy()).max()
    child_ff_part = np.abs(g_plain[0].numpy() - g_sg[0].numpy()).max()
    assert child_cut < 1e-6, child_cut
    assert child_ff_part > 1e-4, child_ff_part  # the feedforward path carried a nonzero gradient in the plain model
    # parent parameters (feedforward weights and base unitary) are unaffected by the stop-gradient
    for i in (1, 2):
        assert np.abs(g_sg[i].numpy() - g_plain[i].numpy()).max() < 1e-6
        assert np.abs(g_sg[i].numpy()).max() > 1e-4
    # with the parent's feedforward weights at zero the feature gradient vanishes: both models agree
    for lay in (plain, sg):
        lay.pff[1].assign(np.zeros(lay.pff[1].shape, dtype=np.float32))
    z_plain = grads(lambda: plain(x), plain)
    z_sg = grads(lambda: sg(x), sg)
    assert np.abs(z_plain[0].numpy() - z_sg[0].numpy()).max() < 1e-6


def test_ff_null_shuffle():
    kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, init="random", seed=31)
    plain = cp.CPTPTNLayer(**kw)
    shuf = cp.CPTPTNLayer(**kw, ff_null="shuffle")
    _set_pff(plain, 0.5)
    _set_pff(shuf, 0.5)
    assert shuf.count_params() == plain.count_params() and shuf.get_config()["ff_null"] == "shuffle"
    x = pure_registers(8, CHI, 2**NL, 30)
    y_plain, y_shuf = plain(x).numpy(), shuf(x).numpy()
    assert np.abs(y_plain - y_shuf).max() > 1e-3  # per-sample features differ, so the output changes
    assert np.allclose(y_shuf.sum(axis=-1), 1.0, atol=1e-4)  # still a probability distribution
    # identical samples: a permutation changes nothing
    x_same = tf.repeat(x[:1], 8, axis=0)
    assert np.abs(plain(x_same).numpy() - shuf(x_same).numpy()).max() < 1e-5
    # zero feedforward weights: the features do not enter, so the shuffle changes nothing
    for lay in (plain, shuf):
        for w in lay.pff:
            if w is not None:
                w.assign(np.zeros(w.shape, dtype=np.float32))
    assert np.abs(plain(x).numpy() - shuf(x).numpy()).max() < 1e-5
    # default config unchanged and invalid values rejected
    assert plain.get_config()["ff_null"] is None
    try:
        cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, ff_null="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown ff_null should be rejected")  # W45: "constant" is now valid


def _record_ff(layer, x):
    """Run ``layer(x)`` and return the feedforward inputs given to ``isometry`` per layer (None for layer 0)."""
    rec = {}
    orig = layer.isometry

    def wrapped(l, ff_input=None):
        rec[l] = None if ff_input is None else ff_input.numpy()
        return orig(l, ff_input)

    layer.isometry = wrapped
    try:
        layer(x)
    finally:
        layer.isometry = orig
    return rec


def _grads_p0(layer, x, w):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w * layer(x))
    return [g.numpy() for g in tape.gradient(loss, [layer.p0[0], layer.p0[1], layer.p0[2], layer.pff[1], layer.pff[2]])]


def test_ff_stop_grad_from_layer_blocks_only_higher_channels():
    x = pure_registers(B, CHI, 8, 40)
    kw = dict(chi=CHI, nlayers=3, mode="B", kraus_rank=4, init="random", seed=41)
    plain, sg1, sg2 = (cp.CPTPTNLayer(**kw, ff_stop_grad=v) for v in (False, True, 2))
    for lay in (plain, sg1, sg2):
        _set_pff(lay, 0.5)
    cfgs = [lay.get_config()["ff_stop_grad"] for lay in (plain, sg1, sg2)]
    assert cfgs == [False, True, 2], cfgs
    assert plain.count_params() == sg1.count_params() == sg2.count_params()
    y = plain(x).numpy()
    assert np.array_equal(y, sg1(x).numpy()) and np.array_equal(y, sg2(x).numpy())  # forward unchanged
    w = tf.constant(np.random.default_rng(6).normal(size=(B, plain.n_outcomes)).astype(np.float32))
    g0, g1, g2 = (_grads_p0(lay, x, w) for lay in (plain, sg1, sg2))
    # parameters of the last layer and the parent weights of layer 2 are unaffected by any cut
    for i in (2, 4):
        assert np.abs(g0[i] - g1[i]).max() < 1e-6 and np.abs(g0[i] - g2[i]).max() < 1e-6
    # layer-1 parameters: features emitted by layer 1 (into layer 2) are cut in both sg1 and sg2, not in plain
    assert np.abs(g1[1] - g2[1]).max() < 1e-6
    assert np.abs(g0[1] - g2[1]).max() > 1e-4
    # layer-0 parameters: the channel layer 0 -> layer 1 is kept by sg2 and cut by sg1; plain also has the channel above
    assert np.abs(g1[0] - g2[0]).max() > 1e-4
    assert np.abs(g0[0] - g2[0]).max() > 1e-4
    # with 2 layers the only channel is layer 0 -> layer 1, so k = 2 is the plain model
    kw2 = dict(chi=CHI, nlayers=2, mode="B", kraus_rank=4, init="random", seed=42)
    a, b = cp.CPTPTNLayer(**kw2), cp.CPTPTNLayer(**kw2, ff_stop_grad=2)
    _set_pff(a, 0.5)
    _set_pff(b, 0.5)
    w2 = tf.constant(np.random.default_rng(7).normal(size=(B, a.n_outcomes)).astype(np.float32))
    ga, gb = (_grads_p0_two(lay, x[..., :4], w2) for lay in (a, b))
    assert all(np.abs(u - v).max() < 1e-6 for u, v in zip(ga, gb))


def _grads_p0_two(layer, x, w):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w * layer(x))
    return [g.numpy() for g in tape.gradient(loss, [layer.p0[0], layer.p0[1], layer.pff[1]])]


def test_ff_null_constant():
    kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, init="random", seed=51)
    plain = cp.CPTPTNLayer(**kw)
    const = cp.CPTPTNLayer(**kw, ff_null="constant")
    _set_pff(plain, 0.5)
    _set_pff(const, 0.5)
    assert const.count_params() == plain.count_params() and const.get_config()["ff_null"] == "constant"
    x = pure_registers(6, CHI, 2**NL, 50)
    rec = _record_ff(const, x)
    rec_plain = _record_ff(plain, x)
    for l in (1, 2):
        assert rec[l] is not None and rec[l].shape == rec_plain[l].shape  # same width as the plain features
        assert np.allclose(rec_plain[l].sum(axis=-1), 2.0, atol=1e-4)
    # the isometry does not depend on the features (same for every sample and every feature value)
    f_a = tf.constant(np.random.default_rng(1).uniform(size=rec[1].shape).astype(np.float32))
    f_b = tf.constant(np.random.default_rng(2).uniform(size=rec[1].shape).astype(np.float32))
    assert np.abs(const.isometry(1, f_a).numpy() - const.isometry(1, f_b).numpy()).max() < 1e-6
    assert np.abs(plain.isometry(1, f_a).numpy() - plain.isometry(1, f_b).numpy()).max() > 1e-4
    # output is the same function of the registers for every batch composition (per-sample information removed)
    y_full = const(x).numpy()
    y_one = const(x[:1]).numpy()
    assert np.abs(y_full[:1] - y_one).max() < 1e-5
    assert np.allclose(y_full.sum(axis=-1), 1.0, atol=1e-4)
    # no gradient reaches the features
    f = tf.Variable(f_a)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.math.real(const.isometry(1, f)))
    g = tape.gradient(loss, f)
    assert g is None or np.abs(g.numpy()).max() == 0.0
    # zero feedforward weights: constant model equals the plain model
    for lay in (plain, const):
        for wgt in lay.pff:
            if wgt is not None:
                wgt.assign(np.zeros(wgt.shape, dtype=np.float32))
    assert np.abs(plain(x).numpy() - const(x).numpy()).max() < 1e-5
    assert plain.get_config()["ff_null"] is None


def test_ff_source_kept_register():
    kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, init="random", seed=61)
    plain = cp.CPTPTNLayer(**kw)
    kept = cp.CPTPTNLayer(**kw, ff_source="kept")
    _set_pff(plain, 0.5)
    _set_pff(kept, 0.5)
    assert kept.count_params() == plain.count_params() and kept.get_config()["ff_source"] == "kept"
    assert plain.get_config()["ff_source"] is None
    x = pure_registers(6, CHI, 2**NL, 60)
    rec = _record_ff(kept, x)
    # features entering layer 1: diagonal of the kept registers of the two layer-0 children, width 2 chi, widened to 2R
    assert rec[1].shape[-1] == 2 * CHI
    assert np.allclose(rec[1].reshape(6, -1, 2, CHI).sum(axis=-1), 1.0, atol=1e-4)
    # independent reference from layer 0: sigma = V^dagger rho V, kept probability = sum over the environment
    d = kept.d
    msgs = tf.transpose(x, [0, 3, 1, 2])
    n = kept.n_vertices[0]
    rho = tf.reshape(tf.einsum("nvim,nvjo->nvijmo", msgs[:, 0::2], msgs[:, 1::2]), [-1, n, d, d])
    V = kept.isometry(0, None)
    sigma = tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))
    diag = tf.math.real(tf.linalg.diag_part(sigma)).numpy().reshape(6, n, kept.R, CHI)
    p_kept = diag.sum(axis=2)
    ref = np.concatenate([p_kept[:, 0::2], p_kept[:, 1::2]], axis=-1)
    assert np.abs(rec[1] - ref).max() < 1e-5
    # the model differs from the environment-feature model, the output is a distribution
    y_plain, y_kept = plain(x).numpy(), kept(x).numpy()
    assert np.abs(y_plain - y_kept).max() > 1e-3
    assert np.allclose(y_kept.sum(axis=-1), 1.0, atol=1e-4)
    # gradients flow through the kept-register features into layer 0
    w = tf.constant(np.random.default_rng(3).normal(size=(6, kept.n_outcomes)).astype(np.float32))
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w * kept(x))
    assert np.abs(tape.gradient(loss, kept.p0[0]).numpy()).max() > 1e-6
    for bad in (dict(ff_width=4), dict(ff_map="hidden"), dict(meas_bases=2), dict(mode="baseline")):
        try:
            cp.CPTPTNLayer(chi=CHI, nlayers=NL, kraus_rank=4, **{**dict(mode="B"), **bad}, ff_source="kept")
        except ValueError:
            pass
        else:
            raise AssertionError(f"ff_source='kept' should reject {bad}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
