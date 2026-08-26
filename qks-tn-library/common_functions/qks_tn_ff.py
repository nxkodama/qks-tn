"""Feedforward-measurement variants of the QKS+TTN model.

The original ``qks_tn.TNLayer`` contracts a *dissipative* tree tensor network: at every
vertex a d x d unitary (d = chi**2) acts on two chi-dimensional registers, one output
register is traced out and the other is passed to the parent.  This module re-implements
that contraction as an explicit batched bottom-up sweep (no ``tf.vectorized_map``, no
``tensornetwork``) and adds two ways of *using* the discarded register instead of
tracing it, following the design note "Feedforward measurement in QKS+TTN: single-shot
vs multishot":

``mode='baseline'``
    Exactly the existing model (verified against ``TNLayer`` in ``tests/``).
``mode='A'`` (single-shot feedforward)
    Each vertex measures its discarded register; the parent applies a unitary
    ``U(b_L, b_R)`` conditioned on the two children's outcomes.  Averaged over outcomes
    this is a linear CPTP map, so the sweep carries the chi unnormalised
    post-measurement states per vertex and sums over the chi**2 outcome pairs at the
    parent.
``mode='B'`` (multishot statistics feedforward)
    Each vertex's outcome distribution ``p = diag(Tr_keep sigma)`` (chi numbers) is fed
    to the parent, whose Hermitian generator is affine in the children's statistics:
    ``H = H0 + sum_k p_L[k] H_k^L + sum_k p_R[k] H_k^R``.  Nonlinear in the input state,
    computed by the same bottom-up sweep with example-dependent unitaries.  Optional
    finite-shot noise (``shots=M``) replaces ``p`` by a Multinomial(M, p)/M estimate
    with a straight-through gradient.

Conventions (all checked numerically against ``construct_dttn`` + ``greedy``):

* ``tf.reshape(U, [chi]*4)`` puts input legs (0, 1) on the *row* index of the d x d
  matrix, so the vertex map is ``sigma = U^T rho conj(U)`` (not ``U rho U^dagger``).
* In the reshaped output ``[c, d, c', d']`` leg ``c`` (index 2 of the 4-leg tensor) is
  the discarded register and ``d`` (index 3) is the kept register.
* The root's readout is indexed ``keep * chi + discard``.
* All vertices in a layer share one unitary when ``share_per_layer=True`` (as in the
  original code, where only ``nlayers`` of the ``2**nlayers - 1`` parameter columns are
  live).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.stats import unitary_group

# Re-exported so callers only need this module.
from qks_tn import ConstMul, ConstMulBinary, FeatureEncodingLayer  # noqa: F401

__all__ = [
    "angles_to_registers",
    "fixed_readout_matrix",
    "ConstMulFixed",
    "hermitian_basis",
    "random_hermitian_params",
    "params_to_hermitian",
    "sample_shots",
    "FastFeatureEncodingLayer",
    "FeedforwardTNLayer",
    "ConstMul",
    "ConstMulBinary",
    "FeatureEncodingLayer",
]


# --------------------------------------------------------------------------------------
# Hermitian parametrisation (identical ordering to qks_tn.TNLayer)
# --------------------------------------------------------------------------------------
def hermitian_basis(d: int) -> np.ndarray:
    """Basis of traceless Hermitian d x d matrices, shape (d*d-1, d, d) complex64.

    Index order matches the parameter vector used by ``qks_tn.TNLayer``:
    ``[0, n)``   -> ``E_ij + E_ji`` for ``i < j`` in ``np.triu_indices(d, 1)`` order,
    ``[n, 2n)``  -> ``i (E_ij - E_ji)``,
    ``[2n, ..)`` -> ``E_kk - E_{d-1,d-1}`` for ``k = 0 .. d-2``,
    with ``n = d (d-1) / 2``.
    """
    n = d * (d - 1) // 2
    rows, cols = np.triu_indices(d, 1)
    basis = np.zeros((d * d - 1, d, d), dtype=np.complex64)
    for k, (i, j) in enumerate(zip(rows, cols)):
        basis[k, i, j] = 1.0
        basis[k, j, i] = 1.0
        basis[n + k, i, j] = 1j
        basis[n + k, j, i] = -1j
    for k in range(d - 1):
        basis[2 * n + k, k, k] = 1.0
        basis[2 * n + k, d - 1, d - 1] = -1.0
    return basis


def random_hermitian_params(d: int, n_units: int, rng) -> np.ndarray:
    """Replica of ``qks_tn.TNLayer.__init__``'s initialisation, shape (d*d-1, n_units).

    For every unit a Haar-random unitary ``u`` is drawn, ``H = -i log(u)`` is formed via
    its eigendecomposition and the real upper triangle, imaginary upper triangle and the
    first ``d-1`` real diagonal entries are stored.  (The last diagonal entry is dropped,
    i.e. the resulting unitary is ``u`` with one diagonal generator entry replaced so
    that ``H`` is traceless -- exactly what the original code does.)
    """
    rng = np.random.default_rng(rng)
    ind = np.triu_indices(d, 1)
    ind_diag = np.diag_indices(d - 1)
    params = np.zeros((d * d - 1, n_units), dtype=np.float64)
    for m in range(n_units):
        u = unitary_group.rvs(d, random_state=rng)
        _, V = np.linalg.eig(u)
        Vinv = np.linalg.inv(V)
        H = V @ np.diag(np.log(np.diag(Vinv @ u @ V))) @ Vinv / 1j
        params[:, m] = np.concatenate((H.real[ind], H.imag[ind], H.real[ind_diag]))
    return params.astype(np.float32)


def params_to_hermitian(params: tf.Tensor, basis: tf.Tensor) -> tf.Tensor:
    """``params`` (K, ...) real -> Hermitian matrices (..., d, d) complex64."""
    # tf.complex (not tf.cast) so the backward pass takes the real part silently
    params_c = tf.complex(params, tf.zeros_like(params))
    return tf.tensordot(params_c, basis, axes=[[0], [0]])


def _expm_i(H: tf.Tensor) -> tf.Tensor:
    """``expm(i H)`` for a batch (..., d, d) of complex matrices (reshaped to 3-D)."""
    shape = tf.shape(H)
    d = H.shape[-1]
    iH = tf.complex(-tf.math.imag(H), tf.math.real(H))
    U = tf.linalg.expm(tf.reshape(iH, [-1, d, d]))
    return tf.reshape(U, shape)


# --------------------------------------------------------------------------------------
# Finite-shot statistics
# --------------------------------------------------------------------------------------
def sample_shots(p: tf.Tensor, shots: int) -> tf.Tensor:
    """Replace the outcome distribution ``p`` (..., K) by a Multinomial(M, p)/M estimate.

    The returned tensor has the sampled *value* but the *gradient* of ``p``
    (straight-through), so the infinite-shot gradient is used for training.
    """
    K = p.shape[-1]
    shape = tf.shape(p)
    q = tf.maximum(p, 1e-12)
    q = q / tf.reduce_sum(q, axis=-1, keepdims=True)
    logits = tf.math.log(tf.reshape(q, [-1, K]))
    draws = tf.random.categorical(logits, shots)  # (N, M) int64
    counts = tf.reduce_sum(tf.one_hot(draws, K, dtype=p.dtype), axis=1) / float(shots)
    counts = tf.reshape(counts, shape)
    return p + tf.stop_gradient(counts - p)


# --------------------------------------------------------------------------------------
# Angles -> registers, for any register size chi = 2**k
# --------------------------------------------------------------------------------------
def angles_to_registers(theta: tf.Tensor) -> tf.Tensor:
    """``theta`` (B, L, k) real -> (B, chi, chi, L) complex64 with ``chi = 2**k``.

    Each angle becomes a single-qubit density matrix (the quantum-kitchen-sink state) and
    the k qubits of a register are combined by a k-fold Kronecker product, in order.
    For k = 2 this reproduces ``FeatureEncodingLayer``'s pairing exactly.
    """
    k = theta.shape[-1]
    L = theta.shape[1]
    zero = tf.zeros_like(theta)
    r00 = tf.complex(tf.math.cos(theta / 2) ** 2, zero)
    r01 = tf.complex(zero, tf.math.sin(theta) / 2)
    r11 = tf.complex(tf.math.sin(theta / 2) ** 2, zero)
    rho1 = tf.stack([tf.stack([r00, r01], axis=-1), tf.stack([-r01, r11], axis=-1)], axis=-2)  # (B, L, k, 2, 2)
    acc = rho1[:, :, 0]
    dim = 2
    for i in range(1, k):
        acc = tf.einsum("nlab,nlcd->nlacbd", acc, rho1[:, :, i])
        dim *= 2
        acc = tf.reshape(acc, [-1, L, dim, dim])
    return tf.transpose(acc, [0, 2, 3, 1])


def fixed_readout_matrix(n_outcomes: int, n_classes: int) -> np.ndarray:
    """Block 0/1 assignment of readout outcomes to classes, (n_classes, n_outcomes).

    Outcomes are handed out in order, the first ``n_outcomes % n_classes`` classes getting
    one extra.  For (16, 10) this is exactly ``qks_tn.ConstMul``; for (16, 2) exactly
    ``qks_tn.ConstMulBinary``; (64, 10) is the chi=8 analogue.
    """
    if n_outcomes < n_classes:
        raise ValueError(f"cannot spread {n_outcomes} outcomes over {n_classes} classes")
    q, r = divmod(n_outcomes, n_classes)
    M = np.zeros((n_classes, n_outcomes), dtype="float32")
    j = 0
    for c in range(n_classes):
        take = q + (1 if c < r else 0)
        M[c, j : j + take] = 1.0
        j += take
    return M


class ConstMulFixed(tf.keras.layers.Layer):
    """Non-trainable readout: sums blocks of outcome probabilities into class scores."""

    def __init__(self, n_outcomes: int, n_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.n_outcomes, self.n_classes = int(n_outcomes), int(n_classes)
        self.const = tf.constant(fixed_readout_matrix(n_outcomes, n_classes))
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.linalg.matvec(self.const, inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(n_outcomes=self.n_outcomes, n_classes=self.n_classes))
        return cfg


# --------------------------------------------------------------------------------------
# Batched feature encoding (same maths as qks_tn.FeatureEncodingLayer)
# --------------------------------------------------------------------------------------
class FastFeatureEncodingLayer(tf.keras.layers.Layer):
    """Quantum-kitchen-sink feature map, fully batched.

    Same parameters, initialisation order and output ``(batch, chi, chi, units/k)`` as
    ``qks_tn.FeatureEncodingLayer`` (so with the same global seed the weights coincide),
    but without ``tf.vectorized_map`` and the per-episode scatter loop.  Unlike the
    original this supports any ``chi = 2**k``: k consecutive episodes form one register.
    """

    def __init__(self, units, bond_dim, input_dim, sigma, **kwargs):
        super().__init__(**kwargs)
        k = int(np.log2(bond_dim))
        if 2**k != bond_dim or k < 1:
            raise ValueError(f"bond_dim must be a power of two, got {bond_dim}")
        if units % k:
            raise ValueError(f"units ({units}) must be divisible by log2(chi) = {k}")
        self.units = units
        self.chi = bond_dim
        self.qubits = k
        self.sigma = sigma
        self.W = tf.Variable(
            tf.random.normal(shape=(units, input_dim), stddev=sigma), name="W", trainable=True
        )
        self.b = tf.Variable(2 * np.pi * tf.random.uniform(shape=(units,)), name="b", trainable=True)

    def call(self, inputs):
        theta = tf.matmul(inputs, self.W, transpose_b=True) + self.b  # (B, units)
        theta = tf.reshape(theta, [-1, self.units // self.qubits, self.qubits])
        return angles_to_registers(theta)  # (B, chi, chi, units/k)


# --------------------------------------------------------------------------------------
# The tree sweep
# --------------------------------------------------------------------------------------
class FeedforwardTNLayer(tf.keras.layers.Layer):
    """Dissipative TTN with optional feedforward of the discarded register.

    Parameters
    ----------
    chi : register dimension (4 in the repo).
    nlayers : tree depth; the layer expects ``2**nlayers`` input registers.
    mode : ``'baseline'`` | ``'A'`` | ``'B'`` (see module docstring).
    share_per_layer : one unitary per layer (original behaviour) or one per vertex.
    shots : ``None`` (exact statistics) or ``M`` (Multinomial shot noise, mode B, training only).
    ff_layers : layers whose unitary receives feedforward from their children
        (default ``1 .. nlayers-1``; layer 0 has no measured children).
    ff_features : ``'raw'`` (``p``) or ``'centered'`` (``p - 1/chi``) for mode B.
    ff_null : ``None`` | ``'constant'`` (``p`` replaced by the uniform distribution) |
        ``'shuffle'`` (``p`` permuted across the batch) -- information-free controls for mode B.
    init_params0 : optional (d*d-1, n_units) array for the base generators.
    seed : numpy seed for the base generators when ``init_params0`` is not given.
    """

    def __init__(
        self,
        chi: int = 4,
        nlayers: int = 6,
        mode: str = "baseline",
        share_per_layer: bool = True,
        shots: int | None = None,
        ff_layers=None,
        ff_features: str = "raw",
        ff_null: str | None = None,
        init_params0=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("baseline", "A", "B"):
            raise ValueError(f"unknown mode {mode!r}")
        if ff_features not in ("raw", "centered"):
            raise ValueError(f"unknown ff_features {ff_features!r}")
        if ff_null not in (None, "none", "constant", "shuffle"):
            raise ValueError(f"unknown ff_null {ff_null!r}")
        self.chi = int(chi)
        self.d = self.chi**2
        self.K = self.d**2 - 1
        self.nlayers = int(nlayers)
        self.mode = mode
        self.share_per_layer = bool(share_per_layer)
        self.shots = None if not shots else int(shots)
        self.ff_features = ff_features
        self.ff_null = None if ff_null in (None, "none") else ff_null
        # vertices per layer and layer-major slot offsets
        self.n_vertices = [2 ** (self.nlayers - 1 - l) for l in range(self.nlayers)]
        self.layer_offsets = np.concatenate([[0], np.cumsum(self.n_vertices)[:-1]]).astype(int)
        if ff_layers is None:
            ff_layers = list(range(1, self.nlayers))
        self.ff_layers = sorted(int(l) for l in ff_layers)
        if any(l < 1 or l >= self.nlayers for l in self.ff_layers):
            raise ValueError("ff_layers must lie in 1 .. nlayers-1")
        self._ff_offsets = {}
        off = 0
        for l in self.ff_layers:
            self._ff_offsets[l] = off
            off += 1 if self.share_per_layer else self.n_vertices[l]
        self._n_ff_units = off

        self.basis = tf.constant(hermitian_basis(self.d))

        n_units = self.nlayers if self.share_per_layer else 2**self.nlayers - 1
        if init_params0 is None:
            init_params0 = random_hermitian_params(self.d, n_units, np.random.default_rng(seed))
        init_params0 = np.asarray(init_params0, dtype=np.float32)
        if init_params0.shape != (self.K, n_units):
            raise ValueError(f"init_params0 must have shape {(self.K, n_units)}, got {init_params0.shape}")
        self.params0 = self.add_weight(
            name="params0",
            shape=(self.K, n_units),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(init_params0),
            trainable=True,
        )
        self.params_ff = None
        self.params_ffA = None
        if self.mode == "B" and self._n_ff_units:
            self.params_ff = self.add_weight(
                name="params_ff",
                shape=(self.K, self._n_ff_units, 2 * self.chi),
                dtype=tf.float32,
                initializer="zeros",
                trainable=True,
            )
        if self.mode == "A" and self._n_ff_units:
            self.params_ffA = self.add_weight(
                name="params_ffA",
                shape=(self.K, self._n_ff_units, self.chi**2),
                dtype=tf.float32,
                initializer="zeros",
                trainable=True,
            )
        # all weights are created here, so mark the layer built (enables count_params)
        self.built = True

    # ---- parameter slices -----------------------------------------------------------
    def _unit_slice(self, l: int) -> slice:
        if self.share_per_layer:
            return slice(l, l + 1)
        return slice(int(self.layer_offsets[l]), int(self.layer_offsets[l]) + self.n_vertices[l])

    def _ff_slice(self, l: int) -> slice:
        off = self._ff_offsets[l]
        return slice(off, off + (1 if self.share_per_layer else self.n_vertices[l]))

    # ---- unitaries --------------------------------------------------------------------
    def build_unitaries(self, l: int, ff_input=None, training=None) -> tf.Tensor:
        """Unitaries of layer ``l``.

        Returns ``(1|n, d, d)`` for baseline / unconditioned layers, ``(B, n, d, d)`` in
        mode B when ``ff_input`` (B, n, 2 chi) is given, and ``(1|n, chi**2, d, d)`` in
        mode A for conditioned layers.
        """
        H0 = params_to_hermitian(self.params0[:, self._unit_slice(l)], self.basis)  # (1|n, d, d)
        if self.mode == "B" and l in self.ff_layers and ff_input is not None:
            f = ff_input
            if self.ff_null == "constant":
                f = tf.ones_like(f) / self.chi
            elif self.ff_null == "shuffle":
                perm = tf.random.shuffle(tf.range(tf.shape(f)[0]))
                f = tf.gather(f, perm, axis=0)
            if self.ff_features == "centered":
                f = f - 1.0 / self.chi
            f = tf.complex(f, tf.zeros_like(f))
            Hk = params_to_hermitian(self.params_ff[:, self._ff_slice(l), :], self.basis)  # (1|n, 2chi, d, d)
            if self.share_per_layer:
                H = H0[tf.newaxis] + tf.einsum("nvk,kab->nvab", f, Hk[0])
            else:
                H = H0[tf.newaxis] + tf.einsum("nvk,vkab->nvab", f, Hk)
            return _expm_i(H)  # (B, n, d, d)
        if self.mode == "A" and l in self.ff_layers:
            dH = params_to_hermitian(self.params_ffA[:, self._ff_slice(l), :], self.basis)  # (1|n, chi², d, d)
            return _expm_i(H0[:, tf.newaxis] + dH)
        return _expm_i(H0)

    # ---- forward ----------------------------------------------------------------------
    def call(self, inputs, training=None):
        chi, d = self.chi, self.d
        eye = tf.eye(chi, dtype=inputs.dtype)
        x = tf.transpose(inputs, [0, 3, 1, 2])  # (B, 2**nlayers, chi, chi)
        if self.mode == "A":
            msgs = x[:, :, tf.newaxis]  # (B, L, 1, chi, chi): one outcome branch at the leaves
        else:
            msgs = x
        ff_input = None
        out = None
        for l in range(self.nlayers):
            n = self.n_vertices[l]
            mL, mR = msgs[:, 0::2], msgs[:, 1::2]
            if self.mode == "A":
                kk = mL.shape[2] * mR.shape[2]
                joint = tf.einsum("nvxim,nvyjo->nvxyijmo", mL, mR)
                joint = tf.reshape(joint, [-1, n, kk, d, d])
                U = self.build_unitaries(l)
                if U.shape.rank == 3:
                    U = U[:, tf.newaxis]  # broadcast over outcome pairs
                sig = tf.matmul(tf.matmul(U, joint, transpose_a=True), tf.math.conj(U))
                sigma = tf.reduce_sum(sig, axis=2)  # (B, n, d, d)
            else:
                rho = tf.einsum("nvim,nvjo->nvijmo", mL, mR)
                rho = tf.reshape(rho, [-1, n, d, d])
                U = self.build_unitaries(l, ff_input, training)
                sigma = tf.matmul(tf.matmul(U, rho, transpose_a=True), tf.math.conj(U))
            s6 = tf.reshape(sigma, [-1, n, chi, chi, chi, chi])  # [c=discard, d=keep, c', d']
            diag = tf.math.real(tf.linalg.diag_part(sigma))  # (B, n, d), index c*chi + d
            if l < self.nlayers - 1:
                if self.mode == "A":
                    # unnormalised post-measurement states of the kept register, per outcome c
                    msgs = tf.einsum("nvcdCe,cC->nvcde", s6, eye)
                else:
                    msgs = tf.einsum("nvcdCe,cC->nvde", s6, eye)  # partial trace over c
                    if self.mode == "B":
                        p = tf.reduce_sum(tf.reshape(diag, [-1, n, chi, chi]), axis=-1)  # (B, n, chi)
                        if self.shots and training:
                            p = sample_shots(p, self.shots)
                        ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)  # (B, n/2, 2chi)
            else:
                # original readout order: keep-major
                out = tf.reshape(tf.transpose(tf.reshape(diag, [-1, chi, chi]), [0, 2, 1]), [-1, d])
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                chi=self.chi,
                nlayers=self.nlayers,
                mode=self.mode,
                share_per_layer=self.share_per_layer,
                shots=self.shots,
                ff_layers=self.ff_layers,
                ff_features=self.ff_features,
                ff_null=self.ff_null,
            )
        )
        return cfg
