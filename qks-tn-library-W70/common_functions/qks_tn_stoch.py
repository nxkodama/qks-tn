"""Classical-stochastic tree: the classical twin of the CPTP tree ``qks_tn_cptp.CPTPTNLayer``.

Node.  Two children pass probability vectors ``pL, pR`` over ``chi`` outcomes.  The node forms the
product distribution ``r = pL (x) pR`` (dimension ``d = chi**2``, index ``i * chi + j``) and applies a
column-stochastic map ``T`` from ``d`` input states to ``chi * R`` output states, index
``e * chi + k`` (environment ``e`` in ``0..R-1``, kept register ``k`` in ``0..chi-1``, the same index as
the diagonal of the CPTP vertex output).  ``T[x, a] = P(a | x)``, so each row of the array
``(d, chi * R)`` sums to one (each column of the transposed map is a distribution).  The kept
marginal goes to the parent and the environment marginal (``R`` numbers) feeds forward.

Parametrization (as in the fit-line study's ``classical_vertices='amplitude'``):
``T = W**2 / sum_a W**2`` with free real ``W``.  With feedforward, ``W = W0 + sum_k f_k W_k``, where
``f`` holds the environment probabilities of the two children (``2R`` features by default, the same count
as ``CPTPTNLayer``); ``W_k`` starts at zero.

Initialization.  ``W0[x, a] = |U0[x, a]|`` for ``a < d``, where ``U0 = expm(i s H0)`` with ``H0`` the seeded
base generator of ``FeedforwardTNLayer`` (``s = init_scale``, default 0.3, the fit-line value; a Haar-scale
dephased tree passes almost no signal).  At ``R = chi`` this is the dephased (unistochastic) unitary vertex.
Entries for the extra environment states (``a >= d``) are ``|N(0, init_noise)|``.

Variants.  ``coherent_layer0=False``: the leaf registers are dephased (only their diagonals enter), every
layer is classical.  ``coherent_layer0=True``: layer 0 is a CPTP vertex of Kraus rank ``R`` acting on the
quantum leaf states (embedded-unitary start as in ``CPTPTNLayer``); the kept register it passes up is
dephased, and layers >= 1 are classical.  ``coherent_layers=k``: layers ``0..k-1`` are CPTP vertices (``k=1``
is ``coherent_layer0=True``); layers ``1..k-1`` receive the quantum kept register of the layer below and, in mode
B, the feedforward Hamiltonian of ``CPTPTNLayer``; layer ``k-1`` passes only the diagonal of its kept register
up (the coherent-to-classical interface of ``k=1``), and layers ``k..`` are classical.  ``k < nlayers``.  The root has environment dimension ``chi`` (``chi**2``
readout outcomes) as in ``CPTPTNLayer``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.linalg import expm

from qks_tn_cptp import hermitian_from_params_np, monomial_features, params_from_hermitian_np
from qks_tn_ff import _expm_i, hermitian_basis, params_to_hermitian, params_to_hermitian_scatter, random_hermitian_params


class StochTNLayer(tf.keras.layers.Layer):
    """Tree whose non-leaf nodes are learned classical stochastic maps with an ``R``-state environment.

    Parameters
    ----------
    chi, nlayers, mode, share_per_layer : as in ``CPTPTNLayer`` (``mode`` is ``'baseline'`` or ``'B'``).
    kraus_rank : environment size ``R >= chi`` of every non-root node (named as in the CPTP layer so the
        two models pair).
    ff_dim, root_ff_dim : feedforward features per non-root node / at the root (default ``2R``); more
        than ``2R`` appends monomials of the environment probabilities (as in ``CPTPTNLayer``).
    coherent_layer0 : layer 0 is a quantum CPTP vertex of Kraus rank ``R`` (see module docstring).
    coherent_layers : number ``k`` of coherent CPTP layers from the leaves (overrides ``coherent_layer0``).
    init_scale, init_noise, seed : see module docstring.
    """

    def __init__(
        self,
        chi: int = 8,
        nlayers: int = 6,
        mode: str = "B",
        share_per_layer: bool = True,
        kraus_rank: int | None = None,
        ff_dim: int | None = None,
        root_ff_dim: int | None = None,
        ff_layers=None,
        coherent_layer0: bool = False,
        coherent_layers: int | None = None,
        init_scale: float = 0.3,
        init_noise: float = 1e-2,
        seed=None,
        ff_noise: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mode not in ("baseline", "B"):
            raise ValueError(f"StochTNLayer supports modes 'baseline' and 'B', got {mode!r}")
        self.chi = int(chi)
        self.d = self.chi**2
        self.R = int(kraus_rank or chi)
        if self.R < self.chi:
            raise ValueError(f"environment size {self.R} < chi={self.chi}")
        self.nlayers = int(nlayers)
        self.mode = mode
        self.share_per_layer = bool(share_per_layer)
        self.n_coh = int(coherent_layers) if coherent_layers is not None else int(bool(coherent_layer0))
        if not 0 <= self.n_coh < int(nlayers):
            raise ValueError(f"coherent_layers must be in [0, nlayers) = [0, {nlayers}), got {self.n_coh}")
        self.coherent_layer0 = self.n_coh >= 1
        self.init_scale = float(init_scale)
        # W60: training-only Gaussian noise (std, absolute units) on the 2R environment probabilities fed forward,
        # added before the monomial feature map, once per layer that uses feedforward; same rule as CPTPTNLayer.
        self.ff_noise = float(ff_noise or 0.0)
        if self.ff_noise < 0.0:
            raise ValueError("ff_noise must be >= 0")
        self.init_noise = float(init_noise)
        self.n_vertices = [2 ** (self.nlayers - 1 - l) for l in range(self.nlayers)]
        base_f = 2 * self.R
        self.ff_dims = [int(ff_dim or base_f) for _ in range(self.nlayers)]
        self.ff_dims[-1] = int(root_ff_dim or ff_dim or base_f)
        for fd in self.ff_dims:
            if fd < base_f:
                raise ValueError(f"ff_dim {fd} < 2R = {base_f}")
        if ff_layers is None:
            ff_layers = list(range(1, self.nlayers))
        self.ff_layers = sorted(int(l) for l in ff_layers) if mode == "B" else []
        if 0 in self.ff_layers:
            raise ValueError("layer-0 feedforward is not defined for the stochastic tree (no leaf environment)")
        # environment size and output states per layer: the root has environment chi (chi**2 readout outcomes)
        self.env = [self.R if l < self.nlayers - 1 else self.chi for l in range(self.nlayers)]
        self.n_out = [self.chi * e for e in self.env]
        self.cls_layers = [l for l in range(self.nlayers) if l >= self.n_coh]

        rng = np.random.default_rng(seed)
        n_units = self.nlayers if self.share_per_layer else 2**self.nlayers - 1
        base = random_hermitian_params(self.d, n_units, np.random.default_rng(seed)).astype(np.float32) * np.float32(self.init_scale)
        offsets = np.concatenate([[0], np.cumsum(self.n_vertices)[:-1]]).astype(int)

        self.W0, self.Wk, self.p0 = [None] * self.nlayers, [None] * self.nlayers, None
        self.pcoh, self.pcoh_ff = [None] * self.nlayers, [None] * self.nlayers  # coherent layers >= 1 (layer 0 is self.p0)
        for l in range(self.nlayers):
            nu = 1 if self.share_per_layer else self.n_vertices[l]
            cols = [l] if self.share_per_layer else list(range(offsets[l], offsets[l] + nu))
            if l in self.cls_layers:
                init = np.zeros((nu, self.d, self.n_out[l]), dtype=np.float32)
                for u in range(nu):
                    U0 = expm(1j * hermitian_from_params_np(base[:, cols[u]].astype(np.float64), self.d))
                    init[u, :, : self.d] = np.abs(U0)
                    if self.n_out[l] > self.d:
                        init[u, :, self.d :] = np.abs(rng.normal(0.0, self.init_noise, size=(self.d, self.n_out[l] - self.d)))
                self.W0[l] = self.add_weight(
                    name=f"W0_l{l}", shape=init.shape, dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(init), trainable=True,
                )
                if l in self.ff_layers:
                    self.Wk[l] = self.add_weight(
                        name=f"Wk_l{l}", shape=(nu, self.ff_dims[l], self.d, self.n_out[l]), dtype=tf.float32,
                        initializer="zeros", trainable=True,
                    )
            else:  # coherent layer: CPTP vertex, embedded-unitary start (as CPTPTNLayer)
                dm = self.chi * self.env[l]
                K = dm * dm - 1
                self._basis0 = tf.constant(hermitian_basis(dm)) if dm <= 64 else None
                init = np.zeros((K, nu), dtype=np.float32)
                for u in range(nu):
                    Hb = np.zeros((dm, dm), dtype=np.complex128)
                    Hb[: self.d, : self.d] = hermitian_from_params_np(base[:, cols[u]].astype(np.float64), self.d)
                    pv = params_from_hermitian_np(Hb, dm)
                    if dm > self.d and self.init_noise > 0:
                        pv = pv + (np.abs(pv) < 1e-12) * rng.normal(0.0, self.init_noise, size=K)
                    init[:, u] = pv
                w = self.add_weight(
                    name=f"params0_l{l}", shape=(K, nu), dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(init), trainable=True,
                )
                if l == 0:
                    self.p0 = w
                else:
                    self.pcoh[l] = w
                    if l in self.ff_layers:
                        self.pcoh_ff[l] = self.add_weight(
                            name=f"params_ff_l{l}", shape=(K, nu, self.ff_dims[l]), dtype=tf.float32,
                            initializer="zeros", trainable=True,
                        )
        self.built = True

    # ---- nodes ------------------------------------------------------------------------------
    def _features(self, l, ff_input):
        fd = self.ff_dims[l]
        if fd > ff_input.shape[-1]:
            ff_input = tf.concat([ff_input, monomial_features(ff_input, fd)], axis=-1)
        return ff_input

    def transition_matrices(self, l: int, ff_input=None) -> tf.Tensor:
        """Stochastic maps of classical layer ``l``: ``(B|1, n|1, d, n_out)``, each row (fixed input) sums to one."""
        L = self.W0[l][tf.newaxis]  # (1, nu, d, out)
        if self.Wk[l] is not None and ff_input is not None:
            f = self._features(l, ff_input)
            Wk = self.Wk[l]
            if self.share_per_layer:
                L = L + tf.einsum("nvk,kxa->nvxa", f, Wk[0])
            else:
                L = L + tf.einsum("nvk,vkxa->nvxa", f, Wk)
        W2 = tf.square(L)
        return W2 / (tf.reduce_sum(W2, axis=-1, keepdims=True) + 1e-12)

    def _coherent_sigma(self, l, msgs, ff_input=None):
        """Coherent layer ``l`` as a CPTP vertex of Kraus rank R: output state (B, n, chi R, chi R), index e*chi + k.

        ``msgs`` are the quantum child registers (B, 2n, chi, chi).  Layer 0 has no feedforward; layers >= 1 add
        the feedforward Hamiltonian ``sum_k f_k H_k`` as in ``CPTPTNLayer.isometry``.
        """
        chi, d = self.chi, self.d
        n = self.n_vertices[l]
        dm = chi * self.env[l]
        mL, mR = msgs[:, 0::2], msgs[:, 1::2]
        rho = tf.reshape(tf.einsum("nvim,nvjo->nvijmo", mL, mR), [-1, n, d, d])
        p = self.p0 if l == 0 else self.pcoh[l]
        H = params_to_hermitian(p, self._basis0) if dm <= 64 else params_to_hermitian_scatter(p, dm)  # (nu, dm, dm)
        pff = None if l == 0 else self.pcoh_ff[l]
        if pff is not None and ff_input is not None:
            f = self._features(l, ff_input)
            f = tf.complex(f, tf.zeros_like(f))  # (B, n, ff_dim)
            Hk = params_to_hermitian(pff, self._basis0) if dm <= 64 else params_to_hermitian_scatter(pff, dm)  # (nu, ff, dm, dm)
            if self.share_per_layer:
                H = H[tf.newaxis] + tf.einsum("nvk,kab->nvab", f, Hk[0])
            else:
                H = H[tf.newaxis] + tf.einsum("nvk,vkab->nvab", f, Hk)
        V = _expm_i(H)[..., :d, :]  # (1|n or B, n, d, dm)
        return tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))

    def _coherent_diag0(self, x):
        """Layer 0 as a CPTP vertex of Kraus rank R on the quantum leaf states: diagonal (B, n, chi R), index e*chi + k."""
        return tf.math.real(tf.linalg.diag_part(self._coherent_sigma(0, x)))

    def call(self, inputs, training=None):
        chi, d = self.chi, self.d
        x = tf.transpose(inputs, [0, 3, 1, 2])  # (B, 2**nlayers, chi, chi)
        probs = tf.math.real(tf.linalg.diag_part(x))  # (B, L, chi): leaf registers dephased
        msgs = x
        ff_input = None
        out = None
        training_on = training is True or (training is not None and not isinstance(training, bool) and bool(training))
        for l in range(self.nlayers):
            n = self.n_vertices[l]
            R = self.env[l]
            if self.ff_noise > 0.0 and training_on and ff_input is not None and l in self.ff_layers:
                ff_input = ff_input + tf.random.normal(tf.shape(ff_input), stddev=self.ff_noise, dtype=ff_input.dtype)
            if l in self.cls_layers:
                pL, pR = probs[:, 0::2], probs[:, 1::2]
                r = tf.reshape(pL[..., :, tf.newaxis] * pR[..., tf.newaxis, :], [-1, n, d])  # index i*chi + j
                T = self.transition_matrices(l, ff_input)
                diag = tf.matmul(r[:, :, tf.newaxis, :], T)[:, :, 0, :]  # (B, n, chi R), index e*chi + k
            else:
                sigma = self._coherent_sigma(l, msgs, ff_input)
                diag = tf.math.real(tf.linalg.diag_part(sigma))
                if l < self.n_coh - 1:  # the next layer is coherent: pass the quantum kept register up
                    s6 = tf.reshape(sigma, [-1, n, R, chi, R, chi])
                    msgs = tf.einsum("nvcdCe,cC->nvde", s6, tf.eye(R, dtype=sigma.dtype))
            if l < self.nlayers - 1:
                dr = tf.reshape(diag, [-1, n, R, chi])
                probs = tf.reduce_sum(dr, axis=2)  # kept marginal (B, n, chi)
                if self.mode == "B":
                    p = tf.reduce_sum(dr, axis=-1)  # environment probabilities (B, n, R)
                    ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)  # (B, n/2, 2R)
            else:
                out = tf.reshape(tf.transpose(tf.reshape(diag, [-1, R, chi]), [0, 2, 1]), [-1, chi * R])
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(chi=self.chi, nlayers=self.nlayers, mode=self.mode, share_per_layer=self.share_per_layer,
                        kraus_rank=self.R, ff_dim=self.ff_dims[0], root_ff_dim=self.ff_dims[-1],
                        ff_layers=self.ff_layers, coherent_layer0=self.coherent_layer0, coherent_layers=self.n_coh,
                        init_scale=self.init_scale, init_noise=self.init_noise, ff_noise=self.ff_noise))
        return cfg
