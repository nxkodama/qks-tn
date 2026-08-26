"""LeNet-style building blocks for the QKS+TTN classifier.

LeNet's ingredients are local receptive fields, weight sharing across positions, pooling,
several feature maps and fully connected layers.  Mapped onto the primitives of this repo:

* **Patch-local, position-shared kitchen sinks** (`PatchQKSEncodingLayer`): the dense
  QKS weight matrix ``W (units, H*W*C)`` is replaced by ``G`` pairs of small filters applied
  to every ``patch x patch`` window of the image (a convolution with stride ``stride``).
  The two angles of a pair become the two qubits of one chi=4 leaf register, exactly as
  in ``qks_tn_ff.FastFeatureEncodingLayer`` (same single-qubit states, same Kronecker
  order, same ``(batch, 4, 4, leaves)`` output), so ``FeedforwardTNLayer`` plugs in
  unchanged.  ``G`` is the number of feature maps ("channel groups") per position.
* **2-D locality in the tree** (`morton_leaf_order`): ``FeedforwardTNLayer`` merges
  consecutive leaves ``(2j, 2j+1)``.  Ordering the positions along a Z-order (Morton)
  curve makes successive merges alternate x and y, so every two tree layers contract one
  2x2 block -- a quadtree of "conv + pool" stages.  Leaves are position-major and
  group-minor, so with ``G > 1`` the first ``log2(G)`` layers mix channels at the same
  location (a 1x1 convolution) before spatial merging starts.
* Pooling is the dissipative trace at every vertex, the nonlinearity between stages is
  feedforward mode B, and the FC layer is the trainable linear head -- all existing.

`share_positions=False` gives the *locally connected* control (a different filter at every
position): same receptive fields, no weight sharing.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

__all__ = [
    "IMAGE_SHAPES",
    "PATCH_DEFAULTS",
    "morton_index",
    "morton_inverse",
    "morton_leaf_order",
    "angles_to_registers",
    "PatchQKSEncodingLayer",
]

IMAGE_SHAPES = {"binary": (28, 28, 1), "multiclass": (28, 28, 1), "cifar10": (32, 32, 3), "imagenet32": (32, 32, 3)}
# (patch, stride, pad) giving an 8x8 grid of positions (64 leaves per channel group)
PATCH_DEFAULTS = {(28, 28): dict(patch=7, stride=3, pad=0), (32, 32): dict(patch=4, stride=4, pad=0)}


# --------------------------------------------------------------------------------------
# Z-order (Morton) leaf ordering
# --------------------------------------------------------------------------------------
def morton_index(y: int, x: int, bits: int) -> int:
    """Interleave bits: bit 2i of the result is x_i, bit 2i+1 is y_i."""
    k = 0
    for i in range(bits):
        k |= ((x >> i) & 1) << (2 * i)
        k |= ((y >> i) & 1) << (2 * i + 1)
    return k


def morton_inverse(k: int, bits: int) -> tuple[int, int]:
    """Inverse of :func:`morton_index`; returns ``(y, x)``."""
    x = y = 0
    for i in range(bits):
        x |= ((k >> (2 * i)) & 1) << i
        y |= ((k >> (2 * i + 1)) & 1) << i
    return y, x


def morton_leaf_order(grid_h: int, grid_w: int, groups: int = 1, order: str = "morton") -> np.ndarray:
    """Permutation ``perm`` with ``perm[leaf_slot] = register_index``.

    Registers are indexed ``r = (y * grid_w + x) * groups + g`` (raster position, then
    channel group).  Leaf slot ``l = slot(y, x) * groups + g`` with ``slot`` the Morton
    index (``order='morton'``) or the raster index (``order='raster'``, identity).
    Morton order requires a square power-of-two grid; ``groups`` must be a power of two.
    """
    if order not in ("morton", "raster"):
        raise ValueError(f"unknown leaf order {order!r}")
    if groups < 1 or groups & (groups - 1):
        raise ValueError("groups must be a power of two")
    L = grid_h * grid_w * groups
    if L & (L - 1):
        raise ValueError(f"number of leaves {L} is not a power of two")
    if order == "morton":
        if grid_h != grid_w or grid_h & (grid_h - 1):
            raise ValueError("Morton order needs a square power-of-two grid of positions")
        bits = int(np.log2(grid_h))
    perm = np.empty(L, dtype=np.int64)
    for y in range(grid_h):
        for x in range(grid_w):
            pos_r = y * grid_w + x
            slot = morton_index(y, x, bits) if order == "morton" else pos_r
            for g in range(groups):
                perm[slot * groups + g] = pos_r * groups + g
    return perm


# --------------------------------------------------------------------------------------
# angles -> chi=4 registers (identical maths to FastFeatureEncodingLayer)
# --------------------------------------------------------------------------------------
def angles_to_registers(theta: tf.Tensor) -> tf.Tensor:
    """``theta`` (B, L, 2) real -> (B, 4, 4, L) complex64 leaf density matrices.

    ``theta[..., 0]`` is the first (high) qubit of each register, ``theta[..., 1]`` the
    second -- the same pairing as consecutive episodes in the dense encoder.
    """
    zero = tf.zeros_like(theta)
    r00 = tf.complex(tf.math.cos(theta / 2) ** 2, zero)
    r01 = tf.complex(zero, tf.math.sin(theta) / 2)
    r11 = tf.complex(tf.math.sin(theta / 2) ** 2, zero)
    rho1 = tf.stack([tf.stack([r00, r01], axis=-1), tf.stack([-r01, r11], axis=-1)], axis=-2)  # (B, L, 2, 2, 2)
    a, b = rho1[:, :, 0], rho1[:, :, 1]
    L = theta.shape[1]
    rho2 = tf.einsum("nkab,nkcd->nkacbd", a, b)  # kron(a, b)
    rho2 = tf.reshape(rho2, [-1, L, 4, 4])
    return tf.transpose(rho2, [0, 2, 3, 1])


# --------------------------------------------------------------------------------------
# The encoder
# --------------------------------------------------------------------------------------
class PatchQKSEncodingLayer(tf.keras.layers.Layer):
    """Patch-local (convolutional) quantum kitchen sinks.

    Parameters
    ----------
    image_shape : (H, W, C); inputs arrive flattened as (B, H*W*C), channels-last.
    patch, stride, pad : receptive field, stride and symmetric zero padding; the grid of
        positions is ``(H + 2 pad - patch) / stride + 1`` per axis (must divide exactly).
    sigma : std of the filter weights (choose ~0.125 * sqrt(784 / (patch*patch*C)) to match
        the dense encoder's angle spread).
    groups : channel groups (feature maps) per position; one chi=4 register each.
    share_positions : True = one filter bank for all positions (convolution);
        False = a separate filter bank per position (locally connected control).
    leaf_order : 'morton' (quadtree merges) or 'raster'.
    """

    def __init__(
        self,
        image_shape,
        patch: int,
        stride: int,
        sigma: float,
        groups: int = 1,
        pad: int = 0,
        share_positions: bool = True,
        leaf_order: str = "morton",
        filters_per_register: int = 2,
        bond_dim: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if filters_per_register != 2 or bond_dim != 4:
            raise NotImplementedError("only chi=4 registers built from 2 filters are supported")
        H, W, C = (int(v) for v in image_shape)
        if (H + 2 * pad - patch) % stride or (W + 2 * pad - patch) % stride:
            raise ValueError(f"patch {patch}/stride {stride}/pad {pad} does not tile a {H}x{W} image exactly")
        self.image_shape = (H, W, C)
        self.patch, self.stride, self.pad = int(patch), int(stride), int(pad)
        self.groups = int(groups)
        self.share_positions = bool(share_positions)
        self.leaf_order = leaf_order
        self.sigma = float(sigma)
        self.grid = ((H + 2 * pad - patch) // stride + 1, (W + 2 * pad - patch) // stride + 1)
        self.n_pos = self.grid[0] * self.grid[1]
        self.n_leaves = self.n_pos * self.groups
        perm = morton_leaf_order(self.grid[0], self.grid[1], self.groups, leaf_order)
        self.perm = tf.constant(perm, dtype=tf.int64)
        self.nlayers = int(np.log2(self.n_leaves))
        n_out = 2 * self.groups
        if self.share_positions:
            self.kernel = tf.Variable(
                tf.random.normal(shape=(self.patch, self.patch, C, n_out), stddev=self.sigma), name="kernel", trainable=True
            )
            self.bias = tf.Variable(2 * np.pi * tf.random.uniform(shape=(n_out,)), name="bias", trainable=True)
        else:
            k = self.patch * self.patch * C
            self.kernel = tf.Variable(
                tf.random.normal(shape=(self.n_pos, k, n_out), stddev=self.sigma), name="kernel", trainable=True
            )
            self.bias = tf.Variable(2 * np.pi * tf.random.uniform(shape=(self.n_pos, n_out)), name="bias", trainable=True)
        self.built = True

    def angles(self, inputs):
        """Encoding angles (B, n_leaves, 2) in leaf-slot order (before the register map)."""
        H, W, C = self.image_shape
        x = tf.reshape(inputs, [-1, H, W, C])
        if self.pad:
            x = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        if self.share_positions:
            theta = tf.nn.conv2d(x, self.kernel, strides=self.stride, padding="VALID") + self.bias  # (B, gh, gw, 2G)
        else:
            P = tf.image.extract_patches(
                x,
                sizes=[1, self.patch, self.patch, 1],
                strides=[1, self.stride, self.stride, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )  # (B, gh, gw, patch*patch*C), inner order (py, px, c) = the kernel's flatten order
            P = tf.reshape(P, [-1, self.n_pos, self.patch * self.patch * C])
            theta = tf.einsum("bnk,nko->bno", P, self.kernel) + self.bias  # (B, n_pos, 2G)
        theta = tf.reshape(theta, [-1, self.n_leaves, 2])  # register r = pos*G + g, last axis f
        return tf.gather(theta, self.perm, axis=1)  # leaf slot l <- register perm[l]

    def call(self, inputs):
        return angles_to_registers(self.angles(inputs))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                image_shape=self.image_shape,
                patch=self.patch,
                stride=self.stride,
                pad=self.pad,
                sigma=self.sigma,
                groups=self.groups,
                share_positions=self.share_positions,
                leaf_order=self.leaf_order,
            )
        )
        return cfg
