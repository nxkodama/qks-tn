"""Two-stage "deep kitchen sinks": conv -> pool -> **conv** -> pool.

The single-tree LeNet analogue (`qks_tn_lenet`) has one encoding stage: the image is turned
into leaf registers once and everything after that is quantum contraction.  LeNet-5 has two
convolutional stages, the second of which sees the *outputs* of the first.  The quantum
analogue of "seeing the outputs" is the design note's multishot idea: measure, and re-encode
the resulting statistics.

Pipeline (chi = 4, 8x8 positions, Morton leaf order):

    image (B, H*W*C)
      -> PatchQKSEncodingLayer                       (B, chi, chi, 64)   64 leaf registers
      -> fold_blocks(4)                              (B*16, chi, chi, 4) one 2x2 spatial block per row
      -> FeedforwardTNLayer(nlayers=2)               (B*16, chi**2)      block outcome distribution
      -> unfold_blocks + Morton-inverse scatter      (B, 4, 4, chi**2)   a classical 4x4 feature map
      -> centre/scale, PatchQKSEncodingLayer(3x3)    (B, chi, chi, 16*G2)
      -> FeedforwardTNLayer(nlayers=log2(16*G2))     (B, chi**2)
      -> head

Because the fold puts the blocks in the *batch* dimension, the stage-1 unitaries are shared
across all 16 blocks -- exactly the weight sharing a convolution has, and the reason this is
implemented by folding rather than by stopping the existing tree part-way.

The classical hand-off between the stages is what makes this strictly more than a deeper
tree: stage 2's encoding angles are a *learned nonlinear function of measured statistics*.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from qks_tn_ff import FeedforwardTNLayer
from qks_tn_lenet import PatchQKSEncodingLayer, morton_inverse

__all__ = ["fold_blocks", "unfold_blocks", "block_scatter_indices", "TwoStageQKSTTN"]


def fold_blocks(rho: tf.Tensor, block: int = 4) -> tf.Tensor:
    """(B, chi, chi, L) -> (B * L/block, chi, chi, block), consecutive leaf slots per row.

    With Morton-ordered leaves, ``block=4`` groups each 2x2 spatial block together.
    """
    chi = rho.shape[1]
    x = tf.transpose(rho, [0, 3, 1, 2])  # (B, L, chi, chi)
    x = tf.reshape(x, [-1, block, chi, chi])  # (B*n_blocks, block, chi, chi)
    return tf.transpose(x, [0, 2, 3, 1])  # (B*n_blocks, chi, chi, block)


def unfold_blocks(out: tf.Tensor, n_blocks: int) -> tf.Tensor:
    """(B * n_blocks, d) -> (B, n_blocks, d)."""
    return tf.reshape(out, [-1, n_blocks, out.shape[-1]])


def block_scatter_indices(n_blocks: int) -> np.ndarray:
    """Permutation mapping Morton block index -> raster position on the sqrt(n) x sqrt(n) grid.

    ``perm[raster] = morton_block`` so ``tf.gather(blocks, perm, axis=1)`` lays the blocks out
    as an image again.
    """
    side = int(round(np.sqrt(n_blocks)))
    if side * side != n_blocks or side & (side - 1):
        raise ValueError(f"{n_blocks} blocks do not form a square power-of-two grid")
    bits = int(np.log2(side))
    perm = np.empty(n_blocks, dtype=np.int64)
    for k in range(n_blocks):
        y, x = morton_inverse(k, bits)
        perm[y * side + x] = k
    return perm


class TwoStageQKSTTN(tf.keras.layers.Layer):
    """Encoder -> 2-layer tree -> classical statistics -> encoder -> tree.

    Parameters
    ----------
    image_shape, patch, stride, pad, groups, sigma : stage-1 patch encoder (see
        :class:`qks_tn_lenet.PatchQKSEncodingLayer`); the leaf order is always Morton so that
        folding by 4 gives 2x2 spatial blocks.
    chi : register dimension for both stages.
    mode, share_per_layer : passed to both trees (``FeedforwardTNLayer``).
    groups2, patch2, sigma2_scale : stage-2 encoder (3x3, stride 1, pad 1 by default, so the
        4x4 grid of blocks is preserved).
    """

    def __init__(
        self,
        image_shape,
        patch: int,
        stride: int,
        sigma: float,
        chi: int = 4,
        pad: int = 0,
        groups: int = 1,
        mode: str = "B",
        share_per_layer: bool = False,
        groups2: int = 1,
        patch2: int = 3,
        sigma2_scale: float = 1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chi = int(chi)
        self.enc1 = PatchQKSEncodingLayer(
            image_shape, patch, stride, sigma, groups=groups, pad=pad, leaf_order="morton", bond_dim=chi
        )
        if self.enc1.nlayers < 2:
            raise ValueError("stage 1 needs at least 4 leaves")
        self.n_blocks = self.enc1.n_leaves // 4
        self.tn1 = FeedforwardTNLayer(chi, 2, mode=mode, share_per_layer=share_per_layer, seed=seed)
        self.perm2 = tf.constant(block_scatter_indices(self.n_blocks))
        side = int(round(np.sqrt(self.n_blocks)))
        d = self.chi**2
        # stage-2 "image": side x side positions, d channels (the block outcome distributions)
        sigma2 = float(sigma) * float(np.sqrt((patch**2 * image_shape[2]) / (patch2**2 * d))) * float(sigma2_scale)
        self.enc2 = PatchQKSEncodingLayer(
            (side, side, d), patch2, 1, sigma2, groups=groups2, pad=(patch2 - 1) // 2, leaf_order="morton", bond_dim=chi
        )
        self.tn2 = FeedforwardTNLayer(
            chi, self.enc2.nlayers, mode=mode, share_per_layer=share_per_layer, seed=(None if seed is None else seed + 1)
        )
        self.d = d
        self.built = True

    def call(self, inputs, training=None):
        rho = self.enc1(inputs)  # (B, chi, chi, L)
        folded = fold_blocks(rho, 4)  # (B*n_blocks, chi, chi, 4)
        p1 = self.tn1(folded, training=training)  # (B*n_blocks, d) probabilities
        blocks = unfold_blocks(p1, self.n_blocks)  # (B, n_blocks, d)
        blocks = tf.gather(blocks, self.perm2, axis=1)  # Morton -> raster
        # centre and scale: the d outcome probabilities sum to 1, so d*p - 1 has zero mean and O(1) spread
        field = tf.reshape(blocks * float(self.d) - 1.0, [-1, self.n_blocks * self.d])
        rho2 = self.enc2(field)
        return self.tn2(rho2, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(chi=self.chi, n_blocks=self.n_blocks, d=self.d))
        return cfg
