"""Tests for ``cptp_nodes`` (W48): replica of the layer forward pass, Kraus spectrum, truncation, renormalization.

    .venv/bin/python qks-tn-library-W48/tests/test_nodes.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))

import cptp_nodes as nd  # noqa: E402
import qks_tn_cptp as cp  # noqa: E402

CHI, NL, B, R = 2, 4, 6, 4


def mixed_registers(batch, chi, nleaves, seed):
    rng = np.random.default_rng(seed)
    rho = 0
    for _ in range(2):
        psi = rng.normal(size=(batch, nleaves, chi)) + 1j * rng.normal(size=(batch, nleaves, chi))
        psi /= np.linalg.norm(psi, axis=-1, keepdims=True)
        rho = rho + 0.5 * np.einsum("bli,blj->bijl", psi, psi.conj())
    return tf.constant(rho, dtype=tf.complex64)


def random_layer(seed=3, noise=0.5, init="random", **kw):
    layer = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=R, init=init, init_noise=noise, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    for w in layer.pff:
        if w is not None:
            w.assign(rng.normal(0, 0.3, size=w.shape).astype(np.float32))
    return layer


def test_replica_matches_layer_call():
    x = mixed_registers(B, CHI, 2**NL, 0)
    for share in (True, False):
        layer = random_layer(share_per_layer=share)
        ref = layer(x).numpy()
        out = nd.forward_nodes(layer, x).numpy()
        assert np.abs(ref - out).max() < 1e-5, (share, np.abs(ref - out).max())
        out_c, st = nd.forward_nodes(layer, x, collect=True)
        assert np.abs(ref - out_c.numpy()).max() < 1e-5
        assert len(st) == NL


def test_gram_trace_and_canonical_kraus_are_orthogonal():
    layer = random_layer()
    V = layer.isometry(1, tf.random.stateless_uniform([B, 2, 2 * R], seed=[1, 2]))
    K = nd.kraus_from_isometry(V, CHI, R)
    G = nd.gram(K).numpy()
    assert np.allclose(np.trace(G, axis1=-2, axis2=-1).real, CHI**2, atol=1e-4)  # trace preserving: Tr G = d
    assert np.allclose(G, np.conj(np.swapaxes(G, -1, -2)), atol=1e-5)
    lam, U = np.linalg.eigh(G)
    Kp = np.einsum("bnci,bnckj->bnikj", U, K.numpy())  # K'_i = sum_c U[c, i] K_c
    Gp = np.einsum("bnikj,bnlkj->bnil", np.conj(Kp), Kp)
    assert np.allclose(Gp, np.einsum("bni,il->bnil", lam, np.eye(R)), atol=1e-4)


def test_embedded_init_has_kraus_rank_chi_with_flat_spectrum():
    layer = cp.CPTPTNLayer(chi=CHI, nlayers=NL, mode="B", kraus_rank=R, init_noise=0.0, seed=2)
    x = mixed_registers(B, CHI, 2**NL, 1)
    _, st = nd.forward_nodes(layer, x, collect=True)
    for l in range(NL - 1):
        s = st[l]
        assert np.allclose(s["eff_rank_sv"].numpy(), CHI, atol=1e-3), l
        assert np.allclose(s["eff_rank_choi"].numpy(), CHI, atol=1e-3), l
        assert np.allclose(s["choi_tail"].numpy(), 0.0, atol=1e-4), l
        assert np.all(s["env_entropy"].numpy() <= np.log(CHI) + 1e-4)


def test_distance_to_init_zero_for_same_channel_and_positive_otherwise():
    x = mixed_registers(B, CHI, 2**NL, 2)
    a = random_layer(seed=4, noise=0.0, init="embed")
    for w in a.pff:
        if w is not None:
            w.assign(tf.zeros_like(w))
    _, st = nd.forward_nodes(a, x, collect=True, init_layer=a)
    assert np.abs(st[0]["dist_init"].numpy()).max() < 1e-3
    b = random_layer(seed=4)
    _, st = nd.forward_nodes(b, x, collect=True, init_layer=a)
    assert st[0]["dist_init"].numpy().min() > 0.01


def test_truncation_choi_rank_and_identity_at_full_rank():
    x = mixed_registers(B, CHI, 2**NL, 3)
    layer = random_layer(seed=5)
    full = nd.forward_nodes(layer, x).numpy()
    same = nd.forward_nodes(layer, x, trunc=dict(r=[R] * (NL - 1), variant="choi", renorm="tp")).numpy()
    assert np.abs(full - same).max() < 1e-4
    V = layer.isometry(1, tf.random.stateless_uniform([B, 2, 2 * R], seed=[3, 4]))
    K = nd.kraus_from_isometry(V, CHI, R)
    for r in (1, 2, 3):
        K2 = nd.truncate_kraus(K, r, "choi", "trace")
        lam = np.linalg.eigvalsh(nd.gram(K2).numpy())
        assert (lam[..., : R - r] < 1e-4).all() and (lam[..., R - r :] > 1e-4).all(), r
    # the projected channel equals the sum of the top-r canonical Kraus terms
    G = nd.gram(K).numpy()
    lam, U = np.linalg.eigh(G)
    Kp = np.einsum("bnci,bnckj->bnikj", U, K.numpy())[:, :, R - 2 :]
    K2 = nd.truncate_kraus(K, 2, "choi", "trace").numpy()
    rho = np.eye(CHI**2)[None, None] / CHI**2
    out_a = np.einsum("bnckj,jm,bnclm->kl", K2, rho[0, 0], np.conj(K2))
    out_b = np.einsum("bnikj,jm,bnilm->kl", Kp, rho[0, 0], np.conj(Kp))
    assert np.abs(out_a - out_b).max() < 1e-4


def test_newton_schulz_inverse_sqrt_and_tp_renormalization():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)))
    w = rng.uniform(0.05, 1.0, size=(8, 8))
    A = np.einsum("bij,bj,bkj->bik", Q, w, np.conj(Q)).astype(np.complex64)
    ref = np.einsum("bij,bj,bkj->bik", Q, w**-0.5, np.conj(Q))
    got = nd.inv_sqrt_psd_ns(tf.constant(A)).numpy()
    assert np.abs(ref - got).max() < 5e-3, np.abs(ref - got).max()  # bias from the eps regularization, relative about 1e-3
    layer = random_layer(seed=6)
    V = layer.isometry(1, tf.random.stateless_uniform([B, 2, 2 * R], seed=[5, 6]))
    K = nd.kraus_from_isometry(V, CHI, R)
    K2 = nd.truncate_kraus(K, 2, "choi", "tp").numpy()  # r = chi: invertible A in general
    S = np.einsum("bnckj,bnckm->bnjm", np.conj(K2), K2)
    assert np.abs(S - np.eye(CHI**2)).max() < 5e-3, np.abs(S - np.eye(CHI**2)).max()


def test_truncated_forward_is_normalized_and_changes_output():
    x = mixed_registers(B, CHI, 2**NL, 4)
    layer = random_layer(seed=7)
    full = nd.forward_nodes(layer, x).numpy()
    for variant in ("choi", "norm"):
        for renorm in ("tp", "trace"):
            out = nd.forward_nodes(layer, x, trunc=dict(r=[1] * (NL - 1), variant=variant, renorm=renorm)).numpy()
            assert np.isfinite(out).all() and np.allclose(out.sum(-1), 1.0, atol=1e-3), (variant, renorm)
            assert np.abs(out - full).max() > 1e-3
    one = nd.forward_nodes(layer, x, trunc=dict(r=[None, 1, None], variant="choi", renorm="tp")).numpy()
    assert np.abs(one - full).max() > 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
