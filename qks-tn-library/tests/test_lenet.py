"""Tests for the LeNet-style pieces (``qks_tn_lenet``) and their wiring.

Run with the project venv::

    .venv/bin/python qks-tn-library/tests/test_lenet.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common_functions"))
sys.path.insert(0, os.path.join(HERE, ".."))

import qks_tn_ff as ff  # noqa: E402
import qks_tn_lenet as ln  # noqa: E402
import run_ff_experiment as rfe  # noqa: E402


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------------------
def test_a_angles_to_registers_matches_fast_encoder():
    """Identity-weight dense encoder on the angles == angles_to_registers."""
    tf.keras.utils.set_random_seed(0)
    L = 64
    theta = tf.random.uniform((8, L, 2), 0, 2 * np.pi)
    fast = ff.FastFeatureEncodingLayer(2 * L, 4, 2 * L, 0.1)
    fast.W.assign(np.eye(2 * L, dtype="float32"))
    fast.b.assign(np.zeros(2 * L, dtype="float32"))
    ref = fast(tf.reshape(theta, [8, 2 * L])).numpy()  # unit 2l <-> theta[l,0], unit 2l+1 <-> theta[l,1]
    out = ln.angles_to_registers(theta).numpy()
    assert out.shape == (8, 4, 4, L)
    assert _maxabs(ref, out) < 1e-6


def test_b_patch_encoder_is_a_patch_sparse_dense_encoder():
    """The patch encoder equals FastFeatureEncodingLayer with a patch-sparse, position-shared W."""
    for (H, W, C), geo in [((28, 28, 1), dict(patch=7, stride=3, pad=0)), ((32, 32, 3), dict(patch=4, stride=4, pad=0)), ((28, 28, 1), dict(patch=9, stride=3, pad=1))]:
        for groups in (1, 2):
            for order in ("raster", "morton"):
                for shared in (True, False):
                    tf.keras.utils.set_random_seed(1)
                    enc = ln.PatchQKSEncodingLayer((H, W, C), sigma=0.3, groups=groups, share_positions=shared, leaf_order=order, **geo)
                    gh, gw = enc.grid
                    L = enc.n_leaves
                    P, S, pad = geo["patch"], geo["stride"], geo["pad"]
                    perm = enc.perm.numpy()
                    kernel, bias = enc.kernel.numpy(), enc.bias.numpy()
                    Wd = np.zeros((2 * L, H * W * C), dtype="float32")
                    bd = np.zeros((2 * L,), dtype="float32")
                    for l in range(L):
                        r = int(perm[l])
                        pos, g = divmod(r, groups)
                        y, x = divmod(pos, gw)
                        for f in (0, 1):
                            o = 2 * g + f
                            for py in range(P):
                                for px in range(P):
                                    iy, ix = y * S - pad + py, x * S - pad + px
                                    if 0 <= iy < H and 0 <= ix < W:
                                        for c in range(C):
                                            w = kernel[py, px, c, o] if shared else kernel[pos, (py * P + px) * C + c, o]
                                            Wd[2 * l + f, (iy * W + ix) * C + c] = w
                            bd[2 * l + f] = bias[o] if shared else bias[pos, o]
                    dense = ff.FastFeatureEncodingLayer(2 * L, 4, H * W * C, 0.3)
                    dense.W.assign(Wd)
                    dense.b.assign(bd)
                    x = tf.random.uniform((4, H * W * C))
                    diff = _maxabs(dense(x).numpy(), enc(x).numpy())
                    assert diff < 2e-5, ((H, W, C), geo, groups, order, shared, diff)
                    # and the tree consumes it unchanged
                    tn = ff.FeedforwardTNLayer(4, enc.nlayers, mode="baseline", seed=0)
                    assert _maxabs(tn(dense(x)).numpy(), tn(enc(x)).numpy()) < 1e-6


def test_c_morton_permutation_and_quadtree_structure():
    for bits in (2, 3):
        n = 2**bits
        for k in range(n * n):
            y, x = ln.morton_inverse(k, bits)
            assert ln.morton_index(y, x, bits) == k
        perm = ln.morton_leaf_order(n, n, 1, "morton")
        assert sorted(perm.tolist()) == list(range(n * n))
        pos = [divmod(int(r), n) for r in perm]  # (y, x) per leaf slot
        for j in range(n * n // 2):  # layer-0 pairs: x-neighbours, same y
            (y0, x0), (y1, x1) = pos[2 * j], pos[2 * j + 1]
            assert y0 == y1 and abs(x0 - x1) == 1
        for j in range(n * n // 4):  # two layers: a 2x2 block
            ys = {p[0] for p in pos[4 * j : 4 * j + 4]}
            xs = {p[1] for p in pos[4 * j : 4 * j + 4]}
            assert len(ys) == 2 and len(xs) == 2 and max(ys) - min(ys) == 1 and max(xs) - min(xs) == 1
        if bits == 3:
            for j in range(4):  # four layers: a 4x4 block
                ys = {p[0] for p in pos[16 * j : 16 * j + 16]}
                xs = {p[1] for p in pos[16 * j : 16 * j + 16]}
                assert len(ys) == 4 and len(xs) == 4
    perm2 = ln.morton_leaf_order(8, 8, 2, "morton")
    assert all(perm2[2 * j] // 2 == perm2[2 * j + 1] // 2 for j in range(64))  # G=2: first merge = same position
    assert np.array_equal(ln.morton_leaf_order(8, 8, 1, "raster"), np.arange(64))


def test_d_grid_validation_and_leaf_counts():
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 3, 0.1).n_leaves == 64
    assert ln.PatchQKSEncodingLayer((32, 32, 3), 4, 4, 0.1).n_leaves == 64
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 7, 0.1).nlayers == 4
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 3, 0.1, groups=2).nlayers == 7
    for bad in (dict(patch=5, stride=3), dict(patch=7, stride=3, groups=3)):
        try:
            ln.PatchQKSEncodingLayer((28, 28, 1), sigma=0.1, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass


def test_e_param_counts():
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 3, 0.1).count_params() == 100
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 3, 0.1, groups=2).count_params() == 200
    assert ln.PatchQKSEncodingLayer((28, 28, 1), 7, 3, 0.1, share_positions=False).count_params() == 6400
    assert ln.PatchQKSEncodingLayer((32, 32, 3), 4, 4, 0.1).count_params() == 98
    assert ln.PatchQKSEncodingLayer((32, 32, 3), 11, 3, 0.1).count_params() == 728
    assert ln.PatchQKSEncodingLayer((32, 32, 3), 4, 4, 0.1, share_positions=False).count_params() == 6272


def test_f_models_build_and_train():
    rng = np.random.default_rng(0)
    for task, p in (("multiclass", 784), ("cifar10", 3072)):
        X = rng.uniform(size=(16, p)).astype("float32")
        y = (np.arange(16) % 10).astype("float32")
        for name in ("L0", "L4", "R4", "U4"):
            for head in ("const", "linear"):
                cfg = rfe.make_cfg(**dict(rfe.CONFIGS[name], task=task, seed=0, head=head))
                model, qks, tn, counts = rfe.build_model(cfg, steps_per_epoch=2, p=p, n_classes=10)
                assert counts["encoder"] == "patch" and counts["leaves"] == 64 and counts["nlayers"] == 6, counts
                h = model.fit(X, y, batch_size=8, epochs=2, verbose=0)
                assert np.all(np.isfinite(h.history["loss"])), (task, name, head)
    cfg = rfe.make_cfg(**dict(rfe.CONFIGS["L4"], task="multiclass", seed=0, freeze_qks=True))
    model, qks, tn, counts = rfe.build_model(cfg, steps_per_epoch=2, p=784, n_classes=10)
    assert counts["trainable"] == counts["tn"] + counts["head"]
    cfg = rfe.make_cfg(**dict(rfe.CONFIGS["L4"], task="multiclass", seed=0, groups=2))
    model, qks, tn, counts = rfe.build_model(cfg, steps_per_epoch=2, p=784, n_classes=10)
    assert counts["leaves"] == 128 and counts["nlayers"] == 7


def test_g_seed_reproducibility_and_raster_vs_morton_pairing():
    def build(order):
        cfg = rfe.make_cfg(**dict(rfe.CONFIGS["L4"], task="multiclass", seed=3, leaf_order=order))
        return rfe.build_model(cfg, steps_per_epoch=2, p=784, n_classes=10)

    m1, q1, t1, _ = build("morton")
    m2, q2, t2, _ = build("morton")
    assert _maxabs(q1.kernel.numpy(), q2.kernel.numpy()) == 0 and _maxabs(t1.params0.numpy(), t2.params0.numpy()) == 0
    m3, q3, t3, _ = build("raster")
    assert _maxabs(q1.kernel.numpy(), q3.kernel.numpy()) == 0 and _maxabs(q1.bias.numpy(), q3.bias.numpy()) == 0
    assert _maxabs(t1.params0.numpy(), t3.params0.numpy()) == 0
    x = tf.random.uniform((4, 784))
    assert _maxabs(m1(x).numpy(), m3(x).numpy()) > 1e-4  # same weights, different leaf order
    # dense model with the same seed shares the TN init (numpy-seeded)
    cfg = rfe.make_cfg(**dict(rfe.CONFIGS["C4"], task="multiclass", seed=3))
    _, _, t4, _ = rfe.build_model(cfg, steps_per_epoch=2, p=784, n_classes=10)
    assert _maxabs(t1.params0.numpy(), t4.params0.numpy()) == 0


def test_h_make_dataset_epoch_length_and_schedule_alignment():
    X = np.zeros((100, 4), dtype="float32")
    y = np.zeros(100, dtype="float32")
    assert len(list(rfe.make_dataset(X, y, 32, shuffle=True, seed=0))) == 3  # training: ragged batch dropped
    assert len(list(rfe.make_dataset(X, y, 32, shuffle=False, seed=0))) == 4  # evaluation keeps it


def test_i_config_names_are_distinct():
    names = {rfe.make_cfg(**dict(rfe.CONFIGS[c], task="cifar10", seed=0))["name"] for c in ("C4", "L4", "R4", "U4", "L0", "L1", "L2", "R1", "U1")}
    assert len(names) == 9
    assert "patch4s4_g1_morton" in rfe.make_cfg(**dict(rfe.CONFIGS["L4"], task="cifar10", seed=0))["name"]
    assert "patch7s3_g2_morton" in rfe.make_cfg(**dict(rfe.CONFIGS["L4"], task="multiclass", seed=0, groups=2))["name"]


def test_j_chi8_generalisation():
    """chi = 8: 3 filters per register, 64-outcome readout; chi = 4 stays bit-compatible."""
    import qks_tn as legacy

    tf.keras.utils.set_random_seed(0)
    X = tf.random.uniform((4, 784))
    # chi = 4 backwards compatibility
    tf.keras.utils.set_random_seed(0)
    ref = legacy.FeatureEncodingLayer(128, 4, 784, 0.125)
    tf.keras.utils.set_random_seed(0)
    new = ff.FastFeatureEncodingLayer(128, 4, 784, 0.125)
    assert _maxabs(ref.call(X).numpy(), new(X).numpy()) < 1e-6
    assert np.array_equal(ff.fixed_readout_matrix(16, 10), legacy.ConstMul().const)
    assert np.array_equal(ff.fixed_readout_matrix(16, 2), legacy.ConstMulBinary().const)
    # chi = 8 dense encoder: 3 angles per register, unit-trace registers
    enc = ff.FastFeatureEncodingLayer(192, 8, 784, 0.125)
    rho = enc(X)
    assert rho.shape == (4, 8, 8, 64)
    tr = np.trace(rho.numpy(), axis1=1, axis2=2)
    assert np.max(np.abs(tr - 1)) < 1e-5
    # tree + readout
    for mode in ("baseline", "B"):
        tn8 = ff.FeedforwardTNLayer(8, 6, mode=mode, share_per_layer=False, seed=0)
        out = tn8(rho).numpy()
        assert out.shape == (4, 64) and np.max(np.abs(out.sum(-1) - 1)) < 1e-5 and out.min() > -1e-6
    assert ff.FeedforwardTNLayer(8, 6, mode="baseline", share_per_layer=False).count_params() == 63 * 4095
    # zero-coupling mode B reduces to the baseline at chi = 8 too
    p0 = ff.random_hermitian_params(64, 6, np.random.default_rng(0))
    b8 = ff.FeedforwardTNLayer(8, 6, mode="baseline", init_params0=p0)
    m8 = ff.FeedforwardTNLayer(8, 6, mode="B", init_params0=p0)
    assert _maxabs(b8(rho).numpy(), m8(rho).numpy()) < 1e-5
    # chi = 8 patch encoder and the generalised head
    pe = ln.PatchQKSEncodingLayer((32, 32, 3), 4, 4, 0.5, bond_dim=8)
    assert pe(tf.random.uniform((4, 3072))).shape == (4, 8, 8, 64)
    assert pe.count_params() == 4 * 4 * 3 * 3 + 3  # 3 filters per register
    head = ff.ConstMulFixed(64, 10)
    assert head.const.shape == (10, 64) and head.const.numpy().sum() == 64


def test_k_chi8_model_builds_and_trains():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 3072)).astype("float32")
    y = (np.arange(8) % 10).astype("float32")
    for name in ("L1", "L4"):
        cfg = rfe.make_cfg(**dict(rfe.CONFIGS[name], task="cifar10", seed=0, chi=8))
        assert "chi8" in cfg["name"]
        model, qks, tn, counts = rfe.build_model(cfg, steps_per_epoch=2, p=3072, n_classes=10)
        assert counts["chi"] == 8 and counts["n_outcomes"] == 64 and counts["leaves"] == 64
        h = model.fit(X, y, batch_size=4, epochs=1, verbose=0)
        assert np.all(np.isfinite(h.history["loss"]))


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
