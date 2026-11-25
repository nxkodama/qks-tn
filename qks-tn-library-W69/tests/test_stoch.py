"""Tests for ``qks_tn_stoch.StochTNLayer``.

    .venv/bin/python qks-tn-library/tests/test_stoch.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import qks_tn_cptp as cp  # noqa: E402
import qks_tn_ff as ff  # noqa: E402
import qks_tn_stoch as st  # noqa: E402

CHI, NL, B = 2, 3, 5


def mixed_registers(batch, chi, nleaves, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(batch, nleaves, chi, chi)) + 1j * rng.normal(size=(batch, nleaves, chi, chi))
    rho = np.einsum("blij,blkj->blik", a, a.conj())
    rho /= np.trace(rho, axis1=-2, axis2=-1)[..., None, None]
    return tf.constant(np.transpose(rho, (0, 2, 3, 1)), dtype=tf.complex64)  # (B, chi, chi, L)


def randomize(layer, seed):
    """Nonzero feedforward weights so the conditioned maps differ from the unconditioned ones."""
    rng = np.random.default_rng(seed)
    for w in layer.Wk:
        if w is None:
            continue
        w.assign(rng.normal(0, 0.3, size=w.shape).astype(np.float32))


def test_maps_are_stochastic():
    for share in (True, False):
        for R in (CHI, 2 * CHI):
            layer = st.StochTNLayer(chi=CHI, nlayers=NL, mode="B", share_per_layer=share, kraus_rank=R, seed=1)
            randomize(layer, 0)
            rng = np.random.default_rng(2)
            for l in layer.cls_layers:
                ff_in = None
                if layer.Wk[l] is not None:
                    ff_in = tf.constant(rng.uniform(size=(B, layer.n_vertices[l], 2 * R)), dtype=tf.float32)
                T = layer.transition_matrices(l, ff_in).numpy()
                assert T.shape[-2:] == (CHI**2, CHI * layer.env[l])
                assert (T >= 0).all()
                assert np.abs(T.sum(-1) - 1).max() < 1e-5, (share, R, l)


def test_outputs_sum_to_one():
    for coherent0 in (False, True):
        for mode in ("baseline", "B"):
            for ffd in (None, 11):
                layer = st.StochTNLayer(chi=CHI, nlayers=NL, mode=mode, kraus_rank=4, coherent_layer0=coherent0,
                                        ff_dim=ffd, seed=3)
                if mode == "B":
                    randomize(layer, 1)
                x = mixed_registers(B, CHI, 2**NL, 0)
                out = layer(x).numpy()
                assert out.shape == (B, CHI**2)
                assert (out >= -1e-7).all()
                assert np.abs(out.sum(-1) - 1).max() < 1e-5, (coherent0, mode, ffd)


def test_environment_feature_count_matches_cptp():
    for R in (4, 8):
        for ffd in (None, 20):
            cl = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=R, ff_dim=ffd, seed=0)
            sl = st.StochTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=R, ff_dim=ffd, seed=0)
            assert sl.ff_dims == cl.ff_dims
            assert sl.ff_layers == cl.ff_layers
            for l in sl.ff_layers:
                assert sl.Wk[l].shape[1] == cl.pff[l].shape[-1]
    # default feature count is 2R (the child environment probabilities), checked at the LK16B size
    big = st.StochTNLayer(chi=8, nlayers=6, mode="B", kraus_rank=16, seed=0)
    assert big.ff_dims == [32] * 6
    ref = cp.CPTPTNLayer(chi=8, nlayers=6, mode="B", kraus_rank=16, seed=0)
    assert big.ff_dims == ref.ff_dims and big.ff_layers == ref.ff_layers


def test_gradients_nonzero():
    for coherent0 in (False, True):
        layer = st.StochTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, coherent_layer0=coherent0, seed=5)
        x = mixed_registers(B, CHI, 2**NL, 1)
        y = tf.constant(np.arange(B) % CHI**2)
        with tf.GradientTape() as tape:
            p = layer(x)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, p))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables) > 0
        for v, g in zip(layer.trainable_variables, grads):
            assert g is not None, v.name
            g = g.numpy() if hasattr(g, "numpy") else np.asarray(tf.convert_to_tensor(g))
            assert np.isfinite(g).all() and np.abs(g).max() > 0, v.name


def test_root_matches_numpy_reference():
    """One layer (the root): out = reshape of |U0|^2 applied to the product of the leaf diagonals."""
    seed, scale = 7, 0.3
    layer = st.StochTNLayer(chi=CHI, nlayers=1, mode="baseline", kraus_rank=CHI, init_scale=scale, seed=seed)
    x = mixed_registers(B, CHI, 2, 4)
    out = layer(x).numpy()
    base = ff.random_hermitian_params(CHI**2, 1, np.random.default_rng(seed))[:, 0] * scale
    U0 = expm(1j * cp.hermitian_from_params_np(base, CHI**2))
    diag = np.real(np.diagonal(x.numpy(), axis1=1, axis2=2))  # (B, L, chi)
    r = np.einsum("bi,bj->bij", diag[:, 0], diag[:, 1]).reshape(B, -1)
    p = r @ np.abs(U0) ** 2  # index e*chi + k
    ref = p.reshape(B, CHI, CHI).transpose(0, 2, 1).reshape(B, -1)
    assert np.abs(out - ref).max() < 1e-5


# ---- W16: first k layers coherent (coherent_layers=k) ------------------------------------------------------

def numpy_hybrid_reference(layer, x):
    """Independent numpy forward of the hybrid tree (coherent layers 0..k-1, classical k..), from the layer's weights."""
    chi, d, k = layer.chi, layer.d, layer.n_coh
    xs = np.transpose(x.numpy(), (0, 3, 1, 2)).astype(np.complex128)  # (B, L, chi, chi)
    Bn = xs.shape[0]
    msgs = xs
    probs = np.real(np.diagonal(xs, axis1=-2, axis2=-1))
    ffin = None
    for l in range(layer.nlayers):
        n, R = layer.n_vertices[l], layer.env[l]
        dm = chi * R
        diag = np.zeros((Bn, n, dm))
        sig_all = np.zeros((Bn, n, dm, dm), dtype=np.complex128)
        if l >= k:
            T = layer.transition_matrices(l, None if ffin is None else tf.constant(ffin, dtype=tf.float32)).numpy()
            r = np.einsum("bvi,bvj->bvij", probs[:, 0::2], probs[:, 1::2]).reshape(Bn, n, d)
            for b in range(Bn):
                for v in range(n):
                    Tm = T[b if T.shape[0] > 1 else 0, v if T.shape[1] > 1 else 0]
                    diag[b, v] = r[b, v] @ Tm
        else:
            P = (layer.p0 if l == 0 else layer.pcoh[l]).numpy()
            Pff = None if (l == 0 or layer.pcoh_ff[l] is None) else layer.pcoh_ff[l].numpy()
            for b in range(Bn):
                for v in range(n):
                    u = 0 if layer.share_per_layer else v
                    H = cp.hermitian_from_params_np(P[:, u].astype(np.float64), dm)
                    if Pff is not None:
                        for q in range(Pff.shape[-1]):
                            H = H + ffin[b, v, q] * cp.hermitian_from_params_np(Pff[:, u, q].astype(np.float64), dm)
                    V = expm(1j * H)[:d, :]
                    rho = np.einsum("im,jo->ijmo", msgs[b, 2 * v], msgs[b, 2 * v + 1]).reshape(d, d)
                    sig = V.T @ rho @ V.conj()
                    sig_all[b, v] = sig
                    diag[b, v] = np.real(np.diag(sig))
        if l < layer.nlayers - 1:
            dr = diag.reshape(Bn, n, R, chi)
            probs = dr.sum(2)
            if layer.mode == "B":
                p = dr.sum(-1)
                ffin = np.concatenate([p[:, 0::2], p[:, 1::2]], axis=-1)
            if l < k - 1:
                s6 = sig_all.reshape(Bn, n, R, chi, R, chi)
                msgs = np.einsum("bvcdce->bvde", s6)
        else:
            return diag.reshape(Bn, R, chi).transpose(0, 2, 1).reshape(Bn, -1)


def test_coherent_depth_matches_numpy_reference():
    for k in (1, 2, 3):
        for share in (True, False):
            layer = st.StochTNLayer(chi=CHI, nlayers=4, mode="B", share_per_layer=share, kraus_rank=4,
                                    coherent_layers=k, seed=11)
            rng = np.random.default_rng(k)
            for w in layer.Wk + layer.pcoh_ff:
                if w is not None:
                    w.assign(rng.normal(0, 0.2, size=w.shape).astype(np.float32))
            x = mixed_registers(3, CHI, 16, k)
            out = layer(x).numpy()
            ref = numpy_hybrid_reference(layer, x)
            assert out.shape == ref.shape == (3, CHI**2)
            assert np.abs(out - ref).max() < 2e-5, (k, share, np.abs(out - ref).max())


def test_k1_reproduces_previous_coherent_layer0():
    """Golden output of the pre-W16 ``coherent_layer0=True`` build (seed 7, randomize seed 1)."""
    gold = np.array([[0.12366046, 0.26602292, 0.401121, 0.20919518], [0.09573331, 0.35075662, 0.3251419, 0.22836751]])
    for kw in (dict(coherent_layer0=True), dict(coherent_layers=1)):
        layer = st.StochTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, seed=7, **kw)
        randomize(layer, 1)
        out = layer(mixed_registers(5, CHI, 2**NL, 0)).numpy()
        assert np.abs(out[:2] - gold).max() < 1e-6, kw


def test_coherent_depth_shapes_normalization_gradients():
    for k in (2, 3):
        layer = st.StochTNLayer(chi=CHI, nlayers=4, mode="B", kraus_rank=4, coherent_layers=k, seed=5)
        assert layer.cls_layers == list(range(k, 4))
        assert [l for l in range(4) if layer.pcoh[l] is not None] == list(range(1, k))
        assert [l for l in range(4) if layer.pcoh_ff[l] is not None] == list(range(1, k))  # feedforward in coherent layers >= 1
        rng = np.random.default_rng(0)
        for w in layer.Wk + layer.pcoh_ff:
            if w is not None:
                w.assign(rng.normal(0, 0.2, size=w.shape).astype(np.float32))
        x = mixed_registers(B, CHI, 16, 2)
        y = tf.constant(np.arange(B) % CHI**2)
        with tf.GradientTape() as tape:
            p = layer(x)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, p))
        out = p.numpy()
        assert out.shape == (B, CHI**2) and (out >= -1e-7).all() and np.abs(out.sum(-1) - 1).max() < 1e-5
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == len(layer.trainable_variables) > 0
        for v, g in zip(layer.trainable_variables, grads):
            assert g is not None, v.name
            g = g.numpy() if hasattr(g, "numpy") else np.asarray(tf.convert_to_tensor(g))
            assert np.isfinite(g).all() and np.abs(g).max() > 0, (k, v.name)
    for bad in (4, -1):
        try:
            st.StochTNLayer(chi=CHI, nlayers=4, mode="B", kraus_rank=4, coherent_layers=bad, seed=0)
        except ValueError:
            continue
        raise AssertionError("coherent_layers out of range accepted")


def test_w60_ff_noise_training_only():
    """ff_noise: clean at evaluation, noisy at training, zero noise reproduces the plain layer, outputs stay normalized."""
    for coh in (None, 1):
        kw = dict(chi=CHI, nlayers=NL, mode="B", kraus_rank=4, coherent_layers=coh, seed=21)
        lay = st.StochTNLayer(**kw, ff_noise=0.05)
        plain = st.StochTNLayer(**kw)
        for m in (lay, plain):
            randomize(m, 3)
        x = mixed_registers(16, CHI, 2**NL, 22)
        assert np.array_equal(lay(x, training=False).numpy(), plain(x).numpy())
        assert np.array_equal(lay(x).numpy(), plain(x).numpy())
        yn = lay(x, training=True).numpy()
        assert np.abs(yn - plain(x).numpy()).max() > 1e-5
        assert np.abs(yn.sum(-1) - 1.0).max() < 1e-4 and np.isfinite(yn).all()
        zero = st.StochTNLayer(**kw, ff_noise=0.0)
        randomize(zero, 3)
        assert np.array_equal(zero(x, training=True).numpy(), plain(x).numpy())
    assert abs(lay.get_config()["ff_noise"] - 0.05) < 1e-9
    try:
        st.StochTNLayer(**kw, ff_noise=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative ff_noise accepted")


def test_w60_noise_configs():
    sys.path.insert(0, os.path.join(HERE, ".."))
    import run_ff_experiment as rfe

    def mk(name):
        return rfe.make_cfg(**dict(rfe.CONFIGS[name], task="cifar10", seed=0, epochs=1))

    for noisy, base in (("LU8Bf112n01af", "LU8Bf112af"), ("LS16Bc0f65n01af", "LS16Bc0f65af")):
        c, b = mk(noisy), mk(base)
        assert c["ff_noise"] == 0.01 and c["augment"] == "flip" and b["ff_noise"] is None
        assert {k for k in c if c[k] != b.get(k)} <= {"ff_noise", "name"}
    assert mk("LU8Bf112n01af")["kraus_rank"] == 8 and not mk("LU8Bf112n01af").get("stoch")
    assert mk("LS16Bc0f65n01af")["stoch"] == "c0" and mk("LS16Bc0f65n01af")["ff_dim"] == 65
    ref = mk("LK16Bn01af")
    assert ref["ff_noise"] == 0.01 and ref["augment"] == "flip"


def test_w69_noise02_config():
    sys.path.insert(0, os.path.join(HERE, ".."))
    import run_ff_experiment as rfe

    def mk(name):
        return rfe.make_cfg(**dict(rfe.CONFIGS[name], task="cifar10", seed=0, epochs=1))

    c, b, n1 = mk("LS16Bc0f65n02af"), mk("LS16Bc0f65af"), mk("LS16Bc0f65n01af")
    assert c["ff_noise"] == 0.02 and c["augment"] == "flip" and b["ff_noise"] is None
    assert {k for k in c if c[k] != b.get(k)} <= {"ff_noise", "name"}
    assert {k for k in c if c[k] != n1.get(k)} <= {"ff_noise", "name"}
    assert c["stoch"] == "c0" and c["ff_dim"] == 65


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
