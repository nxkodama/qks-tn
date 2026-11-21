"""CPTP tree tensor network: every non-root vertex is a channel of Kraus rank ``R``.

Vertex map.  Two child registers (dimension ``chi`` each, joint dimension ``d = chi**2``)
enter a vertex.  The vertex applies a Stinespring isometry ``V : C^d -> C^chi (x) C^R`` and
traces out the environment ``C^R``, so the channel has Kraus rank at most ``R``.  A Kraus
rank below ``chi`` cannot be trace preserving (the output space has dimension ``chi < d``).

``R = chi`` is the dissipative unitary TTN of ``qks_tn_ff.FeedforwardTNLayer``: ``V`` is then
a ``d x d`` unitary and the environment is the discarded register.  ``R > chi`` is a strict
generalization.  Extreme points of the channels ``C^d -> C^chi`` have Kraus rank at most
``d``, so ``R = d = chi**2`` already contains every extreme channel; larger ``R`` adds no
new channels.

Parametrization.  ``V`` is the first ``d`` rows of ``expm(i H)`` with ``H`` a traceless
Hermitian matrix of size ``chi * R``.  Rows of a unitary are orthonormal, so ``V V^dagger = I``
holds exactly at every parameter value, and gradients flow through ``expm``.

Initialization.  The default embeds the baseline unitary: ``H = H0 (+) 0`` gives
``V = [U, 0]``, which reproduces the unitary model exactly (plus ``init_noise`` on every
entry that is zero in the embedding).  A paired comparison across ``R`` then differs only by
the freedom to leave the unitary manifold.

The root vertex is a unitary (``root_rank = chi``) by default, which keeps ``d = chi**2`` readout
outcomes. ``root_rank = Rr > chi`` makes the root a rank-``Rr`` isometry whose ``chi * Rr`` outcome
probabilities all feed the readout, so the readout widens with the root's rank (the unitary model
cannot do this). The runner then builds the fixed readout for ``chi * Rr`` outcomes.

Modes.  ``baseline`` has no feedforward.  ``B`` adds the multishot feedforward of
``qks_tn_ff``: the generator of a vertex is ``H0 + sum_k p[k] H_k`` where ``p`` holds the
environment-outcome probabilities of the two child vertices (``2R`` features).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import tensorflow as tf

from qks_tn_ff import (
    _expm_i,
    hermitian_basis,
    params_to_hermitian,
    params_to_hermitian_scatter,
    random_hermitian_params,
)


@functools.lru_cache(maxsize=None)
def monomial_indices(n_base: int, n_total: int):
    """Index tuples of the extra polynomial features that extend ``n_base`` features to ``n_total``.

    Degree-2 monomials ``f_i f_j`` (``i <= j``) come first, then degree 3, in lexicographic order.
    """
    extra = n_total - n_base
    out = []
    for deg in (2, 3):
        for c in itertools.combinations_with_replacement(range(n_base), deg):
            if len(out) == extra:
                return tuple(out)
            out.append(c)
    if len(out) < extra:
        raise ValueError(f"cannot extend {n_base} features to {n_total} with monomials of degree <= 3")
    return tuple(out)


def monomial_features(f: tf.Tensor, n_total: int) -> tf.Tensor:
    """Extra features (..., n_total - F) of ``f`` (..., F): the monomials of :func:`monomial_indices`, one gather."""
    n_base = f.shape[-1]
    idx = np.full((n_total - n_base, 3), n_base, dtype=np.int32)  # index n_base points at the appended 1
    for j, c in enumerate(monomial_indices(n_base, n_total)):
        idx[j, : len(c)] = c
    f1 = tf.concat([f, tf.ones_like(f[..., :1])], axis=-1)
    return tf.reduce_prod(tf.gather(f1, idx, axis=-1), axis=-1)


def hermitian_from_params_np(params: np.ndarray, d: int) -> np.ndarray:
    """Numpy replica of the traceless-Hermitian parametrization: (K,) -> (d, d)."""
    n = d * (d - 1) // 2
    rows, cols = np.triu_indices(d, 1)
    H = np.zeros((d, d), dtype=np.complex128)
    H[rows, cols] = params[:n] + 1j * params[n : 2 * n]
    H = H + H.conj().T
    diag = np.zeros(d)
    diag[: d - 1] = params[2 * n :]
    diag[d - 1] = -params[2 * n :].sum()
    H[np.diag_indices(d)] = diag
    return H


def params_from_hermitian_np(H: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`hermitian_from_params_np` for a traceless Hermitian ``H``."""
    ind = np.triu_indices(d, 1)
    return np.concatenate([H.real[ind], H.imag[ind], np.diag(H.real)[: d - 1]])


class CPTPTNLayer(tf.keras.layers.Layer):
    """Dissipative TTN whose non-root vertices have Kraus rank ``kraus_rank``.

    Parameters
    ----------
    chi, nlayers, mode, share_per_layer : as in ``FeedforwardTNLayer`` (``mode`` is
        ``'baseline'`` or ``'B'``).
    kraus_rank : environment dimension ``R >= chi`` of every non-root vertex.
    root_rank : environment dimension of the root vertex (default ``chi``, a unitary root).
    root_fold : with ``root_rank > chi``, fold the ``chi * root_rank`` root outcomes back to ``chi**2``
        readout outcomes by summing outcomes that share ``(e mod chi, k)``. The root keeps the parameters of a
        rank-``root_rank`` isometry while the readout keeps its original width, which separates readout width
        from root parameter count. With the embedded start it reproduces the unitary readout exactly.
    init : ``'embed'`` (baseline unitary embedded; exact baseline at ``init_noise = 0``) or
        ``'random'`` (each non-root vertex is the first ``d`` rows of an independent Haar-random
        unitary on ``chi * R`` dimensions, so the environment is entangled with the data at
        initialization; the root stays embedded).
    init_noise : standard deviation of the noise on parameters that are zero in the embedding.
    ff_dim, root_ff_dim : number of feedforward features of the non-root layers / the root (default ``2R``,
        the environment probabilities of the two children). A larger value appends monomials of the
        environment probabilities (degree 2, then 3), which adds feedforward parameters without adding
        environment dimension. With ``kraus_rank = chi`` this is a parameter-matched unitary control.
    ff_map : extra-feature allocation: ``'monomial'`` (default), ``'hidden'`` (trainable tanh
        projection), or ``'random'`` (fixed random tanh projection). The first ``2R`` features
        remain the child probabilities; layer 0 uses the two leaf-register diagonals.
    ff_width : optional trainable linear projection of the ``2R`` child probabilities to
        this fixed width before the feedforward generator. ``None`` keeps the original path.
        At ``ff_width = 2R`` the projection starts as the identity.
    meas_bases : ``K`` measurement bases of the discarded (environment) register of every non-root vertex
        (default ``None`` = ``1`` = unchanged). Basis 0 is the computational basis (the existing ``R``
        outcome probabilities). Each of the ``K - 1`` extra bases is a trainable ``R x R`` unitary ``U_j``
        (``expm(iH_j)``, ``H_j`` traceless Hermitian, Haar-random start, one per layer, or one per vertex
        without sharing) and gives the probabilities ``diag(U_j sigma_E U_j^dagger)``, with ``sigma_E`` the
        reduced state of the discarded register. Each child then passes ``K * R`` features to its parent
        (``2 K R`` per vertex, so ``ff_dim`` defaults to ``2 K R``). With ``kraus_rank = chi`` this is a
        unitary vertex whose feedforward carries more measurement information. Needs ``mode = 'B'`` and
        none of ``ff_dim``, ``root_ff_dim``, ``ff_width``.
    leaf_dephase : ``p`` in [0, 1] (default ``None`` = ``0`` = unchanged). Each leaf register ``rho`` is replaced by
        ``(1 - p) rho + p diag(rho)`` before layer 0 (partial dephasing in the computational basis). Layer 0 already
        takes ``chi x chi`` density matrices, so no separate density-matrix path is needed. ``p = 1`` is the fully
        dephased input of the classical twins.
    ff_stop_grad : (W27) apply ``tf.stop_gradient`` to the feedforward input of every vertex. The forward pass is
        unchanged; the children receive gradient only through the kept state, while the parent's feedforward weights
        still train.
    ff_null : (W27) ``None`` | ``'shuffle'``: permute the feedforward features across the samples of the batch (one
        permutation per layer, applied at train and test time), which keeps each vertex's marginal feature
        statistics and removes the per-sample information. Parameter count unchanged.
    ff_noise : (W33) training-only Gaussian noise: standard deviation added to every feedforward feature (the
        environment probabilities, absolute units, no clipping). Default ``0`` = off. Evaluation (``training`` false) is clean.
    ff_dropout : (W33) training-only dropout rate on the feedforward features (inverted scaling ``1 / (1 - rate)``).
        Default ``0`` = off.
    ff_mix : (W33) training-only feature-wise mixing: each feature of a sample is replaced, with this probability, by the
        same feature of a randomly chosen sample of the batch (one permutation per layer call). Negative control. Default ``0`` = off.
    seed : numpy seed for the base unitaries (same seed -> same base unitaries as
        ``FeedforwardTNLayer``) and for the noise.
    """

    def __init__(
        self,
        chi: int = 4,
        nlayers: int = 6,
        mode: str = "baseline",
        share_per_layer: bool = True,
        kraus_rank: int | None = None,
        root_rank: int | None = None,
        root_fold: bool = False,
        init: str = "embed",
        init_noise: float = 1e-2,
        ff_layers=None,
        ff_dim: int | None = None,
        root_ff_dim: int | None = None,
        ff_map: str = "monomial",
        ff_width: int | None = None,
        meas_bases: int | None = None,
        leaf_dephase: float | None = None,
        ff_stop_grad: bool = False,
        ff_null: str | None = None,
        ff_noise: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mix: float = 0.0,
        seed=None,
        init_params0=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("baseline", "B"):
            raise ValueError(f"CPTPTNLayer supports modes 'baseline' and 'B', got {mode!r}")
        self.chi = int(chi)
        self.d = self.chi**2
        self.R = int(kraus_rank or chi)
        if self.R < self.chi:
            raise ValueError(f"Kraus rank {self.R} < chi={self.chi} cannot be trace preserving")
        self.Rroot = int(root_rank or chi)
        if self.Rroot < self.chi:
            raise ValueError(f"root rank {self.Rroot} < chi={self.chi} cannot be trace preserving")
        if init not in ("embed", "random"):
            raise ValueError(f"unknown init {init!r}")
        self.root_fold = bool(root_fold)
        if self.root_fold and self.Rroot % self.chi:
            raise ValueError("root_fold needs root_rank to be a multiple of chi")
        self.n_outcomes = self.d if self.root_fold else self.chi * self.Rroot
        self.nlayers = int(nlayers)
        self.mode = mode
        self.share_per_layer = bool(share_per_layer)
        self.init_kind = init
        self.init_noise = float(init_noise)
        if ff_map not in ("monomial", "hidden", "random"):
            raise ValueError(f"unknown ff_map {ff_map!r}")
        self.ff_map = ff_map
        self.n_vertices = [2 ** (self.nlayers - 1 - l) for l in range(self.nlayers)]
        self.leaf_dephase = None if not leaf_dephase else float(leaf_dephase)
        self.ff_stop_grad = bool(ff_stop_grad)
        if ff_null in ("none", ""):
            ff_null = None
        if ff_null not in (None, "shuffle"):
            raise ValueError(f"CPTPTNLayer supports ff_null None or 'shuffle', got {ff_null!r}")
        self.ff_null = ff_null
        self.ff_noise = float(ff_noise or 0.0)
        self.ff_dropout = float(ff_dropout or 0.0)
        self.ff_mix = float(ff_mix or 0.0)
        if self.ff_noise < 0.0:
            raise ValueError("ff_noise must be >= 0")
        if not 0.0 <= self.ff_dropout < 1.0:
            raise ValueError("ff_dropout must be in [0, 1)")
        if not 0.0 <= self.ff_mix <= 1.0:
            raise ValueError("ff_mix must be in [0, 1]")
        if self.leaf_dephase is not None and not 0.0 < self.leaf_dephase <= 1.0:
            raise ValueError("leaf_dephase must be in [0, 1]")
        self.K = int(meas_bases or 1)
        if self.K < 1:
            raise ValueError("meas_bases must be >= 1")
        if self.K > 1 and (mode != "B" or ff_width is not None or ff_dim is not None or root_ff_dim is not None):
            raise ValueError("meas_bases > 1 needs mode 'B' and none of ff_dim, root_ff_dim, ff_width")
        base_f = 2 * self.R * self.K
        self.ff_width = None if ff_width is None else int(ff_width)
        if self.ff_width is not None:
            if self.ff_width < 1 or self.ff_width > base_f:
                raise ValueError(f"ff_width must be in [1, {base_f}]")
            if ff_dim is not None or root_ff_dim is not None:
                raise ValueError("ff_width cannot be combined with ff_dim or root_ff_dim")
        self.ff_dims = [int(self.ff_width or ff_dim or base_f) for _ in range(self.nlayers)]
        self.ff_dims[-1] = int(self.ff_width or root_ff_dim or ff_dim or base_f)
        for fd in self.ff_dims:
            if self.ff_width is None and fd < base_f:
                raise ValueError(f"ff_dim {fd} < 2R = {base_f}")
        for fd in set(self.ff_dims):  # validates that the extension is possible
            if fd > base_f and self.ff_map == "monomial":
                monomial_indices(base_f, fd)
        if ff_layers is None:
            ff_layers = list(range(1, self.nlayers))
        self.ff_layers = sorted(int(l) for l in ff_layers) if mode == "B" else []
        if 0 in self.ff_layers and self.R != self.chi:
            raise ValueError("layer-0 feedforward requires kraus_rank = chi")

        # one Hermitian block size per layer: chi*R below the root, d at the root
        self.dim = [self.chi * self.R if l < self.nlayers - 1 else self.chi * self.Rroot for l in range(self.nlayers)]
        self._basis = {}
        for dm in set(self.dim):
            if dm <= 64:
                self._basis[dm] = tf.constant(hermitian_basis(dm))

        rng = np.random.default_rng(seed)
        n_units = self.nlayers if self.share_per_layer else 2**self.nlayers - 1
        if init_params0 is None:
            init_params0 = random_hermitian_params(self.d, n_units, np.random.default_rng(seed))
        init_params0 = np.asarray(init_params0, dtype=np.float32)
        offsets = np.concatenate([[0], np.cumsum(self.n_vertices)[:-1]]).astype(int)

        self.p0, self.pff, self.ff_proj, self.ff_width_proj = [], [], [], []
        self.pmb = [None] * self.nlayers  # extra measurement-basis unitaries of layer l (feed layer l + 1)
        self._mb_basis = tf.constant(hermitian_basis(self.R)) if (self.K > 1 and self.R <= 64) else None
        for l in range(self.nlayers):
            nu = 1 if self.share_per_layer else self.n_vertices[l]
            cols = [l] if self.share_per_layer else list(range(offsets[l], offsets[l] + nu))
            base = init_params0[:, cols]  # (K_d, nu)
            dm = self.dim[l]
            K = dm * dm - 1
            init_l = np.zeros((K, nu), dtype=np.float32)
            for u in range(nu):
                H0 = hermitian_from_params_np(base[:, u], self.d)
                if self.init_kind == "random" and l < self.nlayers - 1:
                    pv = random_hermitian_params(dm, 1, rng)[:, 0]
                else:
                    Hb = np.zeros((dm, dm), dtype=np.complex128)
                    Hb[: self.d, : self.d] = H0
                    pv = params_from_hermitian_np(Hb, dm)
                    if dm > self.d and self.init_noise > 0:
                        # fresh noise only where the embedding has zeros
                        mask = np.abs(pv) < 1e-12
                        pv = pv + mask * rng.normal(0.0, self.init_noise, size=K)
                init_l[:, u] = pv
            w = self.add_weight(
                name=f"params0_l{l}", shape=(K, nu), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(init_l), trainable=True,
            )
            self.p0.append(w)
            if self.K > 1 and l < self.nlayers - 1 and (l + 1) in self.ff_layers:
                mb_rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), l, 917]))
                init_mb = random_hermitian_params(self.R, nu * (self.K - 1), mb_rng)  # (R*R-1, nu*(K-1))
                init_mb = init_mb.reshape(self.R * self.R - 1, nu, self.K - 1).astype(np.float32)
                self.pmb[l] = self.add_weight(
                    name=f"params_mb_l{l}", shape=init_mb.shape, dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(init_mb), trainable=True,
                )
            if self.mode == "B" and l in self.ff_layers:
                wf = self.add_weight(
                    name=f"params_ff_l{l}", shape=(K, nu, self.ff_dims[l]), dtype=tf.float32,
                    initializer="zeros", trainable=True,
                )
                self.pff.append(wf)
                if self.ff_width is not None:
                    projection = np.eye(base_f, self.ff_width, dtype=np.float32)
                    self.ff_width_proj.append(self.add_weight(
                        name=f"ff_width_projection_l{l}", shape=(base_f, self.ff_width), dtype=tf.float32,
                        initializer=tf.keras.initializers.Constant(projection), trainable=True))
                else:
                    self.ff_width_proj.append(None)
                extra = self.ff_dims[l] - base_f if self.ff_width is None else 0
                if extra and self.ff_map in ("hidden", "random"):
                    proj_rng = np.random.default_rng(np.random.SeedSequence([int(seed or 0), l, 731]))
                    projection = proj_rng.normal(0, 1 / np.sqrt(base_f), size=(base_f, extra)).astype(np.float32)
                    if self.ff_map == "hidden":
                        wp = self.add_weight(name=f"ff_projection_l{l}", shape=(base_f, extra), dtype=tf.float32,
                                             initializer=tf.keras.initializers.Constant(projection), trainable=True)
                        bp = self.add_weight(name=f"ff_projection_bias_l{l}", shape=(extra,), dtype=tf.float32,
                                             initializer="zeros", trainable=True)
                        self.ff_proj.append((wp, bp))
                    else:
                        self.ff_proj.append((tf.constant(projection), None))
                else:
                    self.ff_proj.append(None)
            else:
                self.pff.append(None)
                self.ff_proj.append(None)
                self.ff_width_proj.append(None)
        self.built = True

    def _herm(self, params, dm):
        if dm > 64:
            return params_to_hermitian_scatter(params, dm)
        return params_to_hermitian(params, self._basis[dm])

    def meas_unitaries(self, l: int) -> tf.Tensor:
        """Extra-basis unitaries of layer ``l``: (nu, K - 1, R, R)."""
        if self._mb_basis is None:
            H = params_to_hermitian_scatter(self.pmb[l], self.R)
        else:
            H = params_to_hermitian(self.pmb[l], self._mb_basis)
        return _expm_i(H)

    def meas_probs(self, l: int, sigma_env: tf.Tensor) -> tf.Tensor:
        """Outcome probabilities of the discarded register of layer ``l`` in ``K`` bases: (B, n, K * R).

        ``sigma_env`` is its reduced state (B, n, R, R). Basis 0 is the computational basis.
        """
        p0 = tf.math.real(tf.linalg.diag_part(sigma_env))  # (B, n, R)
        if self.pmb[l] is None:
            return p0
        U = self.meas_unitaries(l)  # (nu, K-1, R, R)
        if self.share_per_layer:
            pj = tf.math.real(tf.einsum("jab,Bnbc,jac->Bnja", U[0], sigma_env, tf.math.conj(U[0])))
        else:
            pj = tf.math.real(tf.einsum("njab,Bnbc,njac->Bnja", U, sigma_env, tf.math.conj(U)))
        b, n = tf.shape(p0)[0], p0.shape[1]
        return tf.reshape(tf.concat([p0[:, :, tf.newaxis], pj], axis=2), [b, n, self.K * self.R])

    def _regularize_ff(self, f: tf.Tensor) -> tf.Tensor:
        """W33 training-only regularizers on the child probabilities (B, n, F): mixing, noise, dropout."""
        if self.ff_mix > 0.0:
            other = tf.gather(f, tf.random.shuffle(tf.range(tf.shape(f)[0])), axis=0)
            take = tf.random.uniform(tf.shape(f)) < self.ff_mix
            f = tf.where(take, other, f)
        if self.ff_noise > 0.0:
            f = f + tf.random.normal(tf.shape(f), stddev=self.ff_noise, dtype=f.dtype)
        if self.ff_dropout > 0.0:
            f = tf.nn.dropout(f, rate=self.ff_dropout)
        return f

    def isometry(self, l: int, ff_input=None, training=None):
        """Isometry rows ``V`` of layer ``l``: (1|n, d, dm) or (B, n, d, dm) with feedforward."""
        dm = self.dim[l]
        H0 = self._herm(self.p0[l], dm)  # (nu, dm, dm)
        if self.pff[l] is not None and ff_input is not None:
            if self.ff_null == "shuffle":
                ff_input = tf.gather(ff_input, tf.random.shuffle(tf.range(tf.shape(ff_input)[0])), axis=0)
            if self.ff_stop_grad:
                ff_input = tf.stop_gradient(ff_input)
            if training is True or (training is not None and not isinstance(training, bool) and bool(training)):
                ff_input = self._regularize_ff(ff_input)
            if self.ff_width_proj[l] is not None:
                ff_input = tf.linalg.matmul(ff_input, self.ff_width_proj[l])
            fd = self.ff_dims[l]
            if fd > ff_input.shape[-1]:
                if self.ff_map == "monomial":
                    extra = monomial_features(ff_input, fd)
                else:
                    wp, bp = self.ff_proj[l]
                    extra = tf.linalg.matmul(ff_input, wp)
                    if bp is not None:
                        extra = extra + bp
                    extra = tf.math.tanh(extra)
                ff_input = tf.concat([ff_input, extra], axis=-1)
            f = tf.complex(ff_input, tf.zeros_like(ff_input))  # (B, n, ff_dim)
            Hk = self._herm(self.pff[l], dm)  # (nu, 2R, dm, dm)
            if self.share_per_layer:
                H = H0[tf.newaxis] + tf.einsum("nvk,kab->nvab", f, Hk[0])
            else:
                H = H0[tf.newaxis] + tf.einsum("nvk,vkab->nvab", f, Hk)
        else:
            H = H0
        return _expm_i(H)[..., : self.d, :]

    def call(self, inputs, training=None):
        chi, d = self.chi, self.d
        x = tf.transpose(inputs, [0, 3, 1, 2])  # (B, 2**nlayers, chi, chi)
        if self.leaf_dephase is not None:  # rho -> (1 - p) rho + p diag(rho)
            eye = tf.eye(chi, dtype=x.dtype)
            x = (1.0 - self.leaf_dephase) * x + self.leaf_dephase * x * eye
        msgs = x
        ff_input = None
        if 0 in self.ff_layers:
            leaf_p = tf.math.real(tf.linalg.diag_part(x))
            ff_input = tf.concat([leaf_p[:, 0::2], leaf_p[:, 1::2]], axis=-1)
        out = None
        for l in range(self.nlayers):
            n = self.n_vertices[l]
            R = self.R if l < self.nlayers - 1 else self.Rroot
            mL, mR = msgs[:, 0::2], msgs[:, 1::2]
            rho = tf.einsum("nvim,nvjo->nvijmo", mL, mR)
            rho = tf.reshape(rho, [-1, n, d, d])
            V = self.isometry(l, ff_input, training=training)
            sigma = tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))  # (B, n, chi R, chi R)
            diag = tf.math.real(tf.linalg.diag_part(sigma))  # index e*chi + k, e = environment
            if l < self.nlayers - 1:
                s6 = tf.reshape(sigma, [-1, n, R, chi, R, chi])
                msgs = tf.einsum("nvcdCe,cC->nvde", s6, tf.eye(R, dtype=sigma.dtype))
                if self.mode == "B":
                    if self.pmb[l] is not None:
                        sigma_env = tf.linalg.trace(tf.transpose(s6, [0, 1, 2, 4, 3, 5]))  # (B, n, R, R)
                        p = self.meas_probs(l, sigma_env)  # (B, n, K R)
                    else:
                        p = tf.reduce_sum(tf.reshape(diag, [-1, n, R, chi]), axis=-1)  # (B, n, R)
                    ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)
            else:
                if self.root_fold:
                    # environment e = a * chi + b; sum over a, keep (b, k), keep-major order k * chi + b
                    g = tf.reduce_sum(tf.reshape(diag, [-1, self.Rroot // chi, chi, chi]), axis=1)  # (B, b, k)
                    out = tf.reshape(tf.transpose(g, [0, 2, 1]), [-1, self.n_outcomes])
                else:
                    # keep-major readout order: index k * Rroot + e (the original order when Rroot = chi)
                    out = tf.reshape(tf.transpose(tf.reshape(diag, [-1, self.Rroot, chi]), [0, 2, 1]), [-1, self.n_outcomes])
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(chi=self.chi, nlayers=self.nlayers, mode=self.mode,
                        share_per_layer=self.share_per_layer, kraus_rank=self.R, root_rank=self.Rroot, root_fold=self.root_fold,
                        init=self.init_kind, init_noise=self.init_noise, ff_layers=self.ff_layers,
                        ff_dim=None if (self.ff_width is not None or self.K > 1) else self.ff_dims[0],
                        root_ff_dim=None if (self.ff_width is not None or self.K > 1) else self.ff_dims[-1],
                        ff_map=self.ff_map, ff_width=self.ff_width,
                        meas_bases=None if self.K == 1 else self.K, leaf_dephase=self.leaf_dephase,
                        ff_stop_grad=self.ff_stop_grad, ff_null=self.ff_null,
                        ff_noise=self.ff_noise, ff_dropout=self.ff_dropout, ff_mix=self.ff_mix))
        return cfg
