"""W48: Kraus structure of trained CPTP tree nodes and truncation of the channels without retraining.

Conventions (match ``qks_tn_cptp.CPTPTNLayer.call``).  A non-root vertex has isometry rows ``V`` of shape
``(d, chi * R)`` with column index ``c * chi + k`` (``c`` environment, ``k`` kept).  The Kraus operators are
``K_c[k, j] = V[j, c * chi + k]`` (shape ``(chi, d)``) and the output state is ``sum_c K_c rho K_c^dagger``.
With feedforward the isometry depends on the sample and the vertex, so every (sample, vertex) pair has its own channel.

Spectrum.  ``G[c, e] = sum_{k,j} conj(K_c[k, j]) K_e[k, j]`` (``R x R``, Hermitian, trace ``d`` for a trace-preserving
channel).  Its eigenvalues are the eigenvalues of the Choi matrix, so the singular values of the stacked Kraus operators
(rows ``vec K_c``) are ``sqrt(lambda)``.  The canonical Kraus operators are ``K'_i = sum_c U[c, i] K_c`` with
``G U = U diag(lambda)``; they satisfy ``<K'_i, K'_j> = lambda_i delta_ij``.

Truncation.  ``variant='choi'`` keeps the top-``r`` canonical Kraus operators and expresses the truncated channel back in
the original environment labels, ``K''_c = sum_e P[c, e] K_e`` with ``P = conj(U_top) U_top^T`` (an ``R x R`` projector), so
the feature slot ``c`` of the parent keeps its meaning.  ``variant='norm'`` keeps the ``r`` original Kraus operators with
the largest Frobenius norm and zeroes the others.  Renormalization ``renorm='tp'``: ``K~_c = K''_c A^{-1/2}`` with
``A = sum_c K''_c^dagger K''_c`` (inverse square root by coupled Newton-Schulz iterations), which gives
``sum_c K~_c^dagger K~_c = I`` when ``A`` is invertible, followed by division of the output state and the features by the
output trace.  ``renorm='trace'`` applies only the division by the output trace.  A channel with fewer than ``chi``
Kraus operators has ``A`` singular and cannot be trace preserving; the trace division is then the only guarantee.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

GPU_EIGH = False  # W49: use tf.linalg.eigh (GPU) in the ``haar`` variant instead of CPU LAPACK; ``choi`` always uses LAPACK as in W48


def kraus_from_isometry(V, chi, R):
    """(B|1, n|nu, d, chi*R) -> Kraus operators (.., R, chi, d)."""
    s = V.shape
    V5 = tf.reshape(V, [s[0], s[1], s[2], R, chi])  # j, c, k
    return tf.transpose(V5, [0, 1, 3, 4, 2])  # c, k, j


def gram(K):
    """G[..., c, e] = sum_{k,j} conj(K_c) K_e : (.., R, R)."""
    return tf.einsum("...ckj,...ekj->...ce", tf.math.conj(K), K)


def inv_sqrt_psd_ns(A, iters=40, eps=1e-4):  # eps: relative ridge, see below
    """Inverse square root of a PSD matrix batch (..., d, d) by coupled Newton-Schulz iterations.

    ``A`` is replaced by ``A + eps * ||A|| I`` (``||A||`` the infinity norm) so that null directions of a singular ``A`` do not
    make the iteration diverge; directions with eigenvalue below about ``eps * ||A||`` are only partly renormalized.
    """
    d = A.shape[-1]
    eye = tf.eye(d, dtype=A.dtype)
    c = tf.reduce_max(tf.reduce_sum(tf.abs(A), axis=-1), axis=-1)  # infinity norm >= spectral norm
    c = tf.maximum(c, 1e-12)
    cc = tf.cast(c, A.dtype)[..., None, None]
    Y = A / cc + eps * eye
    Z = tf.broadcast_to(eye, tf.shape(A))
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - tf.matmul(Z, Y))
        Y = tf.matmul(Y, T)
        Z = tf.matmul(T, Z)
    return Z / tf.cast(tf.sqrt(c), A.dtype)[..., None, None]  # (A/c + eps I)^(-1/2) / sqrt(c)


def haar_unitary(n, seed):
    """Haar-random n x n unitary (numpy, complex128): QR of a complex Ginibre matrix with the phases of R fixed."""
    rng = np.random.default_rng(seed)
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    Q, Rm = np.linalg.qr(Z)
    d = np.diag(Rm)
    return Q * (d / np.abs(d))[None, :]


def truncate_kraus(K, r, variant="choi", renorm="tp", ns_iters=40, rho=None, draw=None, ns_eps=1e-4):
    """Replace the channel of every (sample, vertex) by ``r`` of its Kraus operators; K: (B, n, R, chi, d).

    Variants (W49 adds the last four; ``choi`` and ``norm`` are unchanged):
      choi   : top-``r`` canonical Kraus operators (largest Choi eigenvalues), written back in the environment labels.
      norm   : the ``r`` original operators ``K_c`` with the largest Frobenius norm (diagonal of the Gram matrix).
      haar   : ``draw`` is a Haar unitary ``Q`` (R x R); the kept subspace is spanned by the columns ``0..r-1`` of ``Q``
               read in the Gram eigenbasis of each (sample, vertex): ``M = U Q[:, :r]``, ``P = conj(M) M^T``.
      subset : ``draw`` is a permutation of ``range(R)``; the operators ``K_c`` with ``c`` in ``draw[:r]`` are kept (the same
               labels for every sample and vertex).
      diagp  : needs ``rho`` (B, n, d, d); the ``r`` operators with the largest environment probability
               ``p_c = Tr(K_c rho K_c^dagger)`` of that sample and vertex are kept.
    """
    R = K.shape[2]
    if r >= R:
        return K
    if variant in ("choi", "haar"):
        G = gram(K)
        if variant == "haar" and GPU_EIGH:
            lam, U = tf.linalg.eigh(G)
            U = U.numpy()
        else:
            lam, U = np.linalg.eigh(G.numpy())  # ascending; CPU LAPACK, 16 x 16
        if variant == "choi":
            Ut = U[..., :, R - r :]
        else:
            Q = np.asarray(draw)[:, :r]
            Ut = np.einsum("...ci,ij->...cj", U, Q)
        P = np.einsum("...ci,...di->...cd", np.conj(Ut), Ut)
        K2 = tf.einsum("bncd,bndkj->bnckj", tf.constant(P, dtype=K.dtype), K)
    elif variant in ("norm", "diagp"):
        if variant == "norm":
            w = tf.math.real(tf.linalg.diag_part(gram(K)))
        else:
            T = tf.einsum("bnckj,bnjm->bnckm", K, rho)
            w = tf.math.real(tf.einsum("bnckm,bnckm->bnc", T, tf.math.conj(K)))
        idx = tf.math.top_k(w, k=r).indices
        mask = tf.reduce_sum(tf.one_hot(idx, R, dtype=tf.float32), axis=-2)
        K2 = K * tf.cast(mask, K.dtype)[..., None, None]
    elif variant == "subset":
        mask = np.zeros(R, dtype=np.float32)
        mask[np.asarray(draw)[:r]] = 1.0
        K2 = K * tf.cast(tf.constant(mask), K.dtype)[None, None, :, None, None]
    else:
        raise ValueError(variant)
    if renorm == "tp":
        A = tf.einsum("bnckj,bnckm->bnjm", tf.math.conj(K2), K2)
        K2 = tf.einsum("bnckj,bnjm->bnckm", K2, inv_sqrt_psd_ns(A, ns_iters, ns_eps))
    elif renorm != "trace":
        raise ValueError(renorm)
    return K2


def _entropy(p, axis=-1):
    p = tf.maximum(p, 0.0)
    p = p / tf.maximum(tf.reduce_sum(p, axis=axis, keepdims=True), 1e-30)
    return -tf.reduce_sum(tf.where(p > 1e-12, p * tf.math.log(tf.maximum(p, 1e-30)), 0.0), axis=axis)


def channel_stats(K, rho, chi, K0=None):
    """Per (sample, vertex) statistics of the channels K (B, n, R, chi, d) acting on rho (B, n, d, d).

    Returns a dict of tensors of shape (B, n) except ``spec_sv`` and ``spec_choi`` (B, n, R, descending).
    """
    G = gram(K)
    lam = tf.maximum(tf.constant(np.linalg.eigvalsh(G.numpy()))[..., ::-1], 0.0)  # descending (CPU LAPACK, R x R)
    sv = tf.sqrt(lam)
    sv_n = sv / tf.reduce_sum(sv, axis=-1, keepdims=True)
    lam_n = lam / tf.reduce_sum(lam, axis=-1, keepdims=True)
    T = tf.einsum("bnckj,bnjm->bnckm", K, rho)
    sig_env = tf.einsum("bnckm,bnekm->bnce", T, tf.math.conj(K))  # reduced state of the discarded register
    ev = tf.constant(np.linalg.eigvalsh(sig_env.numpy()))
    out = dict(
        spec_sv=sv_n, spec_choi=lam_n,
        eff_rank_sv=tf.exp(_entropy(sv_n)), eff_rank_choi=tf.exp(_entropy(lam_n)),
        env_entropy=_entropy(tf.maximum(ev, 0.0)),
        choi_tail=1.0 - tf.reduce_sum(lam_n[..., :chi], axis=-1),
    )
    if K0 is not None:
        G0 = gram(K0)
        X = tf.einsum("...ckj,...ekj->...ce", tf.math.conj(K), K0)
        n2 = tf.reduce_sum(tf.abs(G) ** 2, axis=[-2, -1])
        n02 = tf.reduce_sum(tf.abs(G0) ** 2, axis=[-2, -1])
        x2 = tf.reduce_sum(tf.abs(X) ** 2, axis=[-2, -1])
        out["dist_init"] = tf.sqrt(tf.maximum(n2 + n02 - 2.0 * x2, 0.0)) / tf.sqrt(n02)
    return out


def forward_nodes(layer, inputs, trunc=None, collect=False, init_layer=None, ns_iters=40, info=None):
    """Replica of ``CPTPTNLayer.call`` (modes baseline and B, shared or per-vertex, no leaf dephasing) with two hooks.

    trunc : None or dict(r=[r_0..r_{L-2}] (None = untouched, root never truncated), variant, renorm).
    collect : also return per-layer ``channel_stats`` of the (untruncated) channels.
    init_layer : a second ``CPTPTNLayer`` (the seed-matched initialization) giving K0 for ``dist_init``.
    info : optional dict; ``info['fallback'][l]`` accumulates the number of (sample, vertex) pairs of truncated layer ``l`` whose
        output trace fell below 1e-6 (input in the kernel of the truncated channel); those outputs are replaced by the
        maximally mixed state with uniform feature values.
    """
    chi, d = layer.chi, layer.d
    if layer.leaf_dephase is not None or 0 in layer.ff_layers:
        raise ValueError("unsupported option for forward_nodes")
    msgs = tf.transpose(inputs, [0, 3, 1, 2])
    ff_input = None
    stats = []
    out = None
    B = tf.shape(msgs)[0]
    for l in range(layer.nlayers):
        n = layer.n_vertices[l]
        R = layer.R if l < layer.nlayers - 1 else layer.Rroot
        mL, mR = msgs[:, 0::2], msgs[:, 1::2]
        rho = tf.reshape(tf.einsum("nvim,nvjo->nvijmo", mL, mR), [-1, n, d, d])
        V = layer.isometry(l, ff_input)
        if V.shape.rank == 3:
            V = V[tf.newaxis]
        if l == layer.nlayers - 1:
            if trunc is not None and trunc.get("root_r") is not None:  # W49: truncate the root's environment (R = chi)
                K = tf.broadcast_to(kraus_from_isometry(V, chi, R), [B, n, R, chi, d])
                K = truncate_kraus(K, int(trunc["root_r"]), trunc.get("variant", "choi"), trunc.get("renorm", "tp"), ns_iters,
                                   rho=rho, draw=trunc.get("draw"))
                T = tf.einsum("bnckj,bnjm->bnckm", K, rho)
                pck = tf.math.real(tf.einsum("bnckm,bnckm->bnck", T, tf.math.conj(K)))[:, 0]  # (B, R, chi)
                pck = pck / tf.maximum(tf.reduce_sum(pck, axis=[1, 2], keepdims=True), 1e-12)
                out = tf.reshape(tf.transpose(pck, [0, 2, 1]), [-1, chi * R])
                break
            sigma = tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))
            diag = tf.math.real(tf.linalg.diag_part(sigma))
            out = tf.reshape(tf.transpose(tf.reshape(diag, [-1, R, chi]), [0, 2, 1]), [-1, chi * R])
            if collect:
                Kr = tf.broadcast_to(kraus_from_isometry(V, chi, R), [B, n, R, chi, d])
                stats.append(channel_stats(Kr, rho, chi, None))
            break
        Kn = kraus_from_isometry(V, chi, R)  # native shape (1|B, 1|n, R, chi, d); (1, 1) for the fixed shared channel of layer 0
        K = tf.broadcast_to(Kn, [B, n, R, chi, d])
        if collect:
            K0 = None
            if init_layer is not None:
                V0 = init_layer.isometry(l, None)
                if V0.shape.rank == 3:
                    V0 = V0[tf.newaxis]
                K0 = kraus_from_isometry(V0, chi, R)
            stats.append(channel_stats(K, rho, chi, K0))
        if trunc is not None and trunc["r"][l] is not None:
            if Kn.shape[0] == 1 and trunc.get("variant", "choi") != "diagp" and trunc.get("fast", True):
                # W49: a sample-independent channel is truncated once at its native shape (identical result, far fewer matrices)
                K = tf.broadcast_to(truncate_kraus(Kn, int(trunc["r"][l]), trunc.get("variant", "choi"), trunc.get("renorm", "tp"),
                                                   ns_iters, draw=trunc.get("draw")), [B, n, R, chi, d])
            else:
                K = truncate_kraus(K, int(trunc["r"][l]), trunc.get("variant", "choi"), trunc.get("renorm", "tp"), ns_iters,
                                   rho=rho, draw=trunc.get("draw"))
        T = tf.einsum("bnckj,bnjm->bnckm", K, rho)
        rho_c = tf.einsum("bnckm,bnclm->bnckl", T, tf.math.conj(K))  # (B, n, R, chi, chi)
        p = tf.math.real(tf.einsum("bnckk->bnc", rho_c))
        tr = tf.reduce_sum(p, axis=-1)
        if trunc is not None and trunc["r"][l] is not None:
            bad = tr < 1e-6
            if info is not None:
                info.setdefault("fallback", {}).setdefault(l, 0)
                info["fallback"][l] += int(tf.reduce_sum(tf.cast(bad, tf.int32)))
            trs = tf.where(bad, 1.0, tr)
            p = p / trs[..., None]
            rho_c = rho_c / tf.cast(trs, rho_c.dtype)[..., None, None, None]
            p = tf.where(bad[..., None], 1.0 / R, p)
            rho_c = tf.where(bad[..., None, None, None], tf.eye(chi, dtype=rho_c.dtype) / (R * chi), rho_c)
        msgs = tf.reduce_sum(rho_c, axis=2)
        if layer.mode == "B":
            ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)
    return (out, stats) if collect else out
