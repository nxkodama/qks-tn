"""Train / evaluate QKS+TTN models (baseline and feedforward variants) on MNIST.

Reproduces the data pipeline, optimiser and schedule of the repo notebooks
(``train_binary_classifier.ipynb`` / ``train_multiclass_classifier.ipynb``) and logs
per-epoch training and validation (= test set) metrics to JSON.

Usage (local)::

    .venv/bin/python qks-tn-library/run_ff_experiment.py --task binary --mode B --epochs 15 \
        --seed 0 --out qks-tn-library/runs/binary_B_s0.json

``train(cfg, on_epoch=...)`` is importable (used by ``modal_ff_sweep.py``).

Checkpointing: a ``tf.train.Checkpoint`` (weights, AdamW moments, optimizer step counter)
is written to ``cfg['ckpt_dir']`` after every epoch together with the metric history.
Re-running with a larger ``--epochs`` and ``--resume`` continues training from the last
epoch with the learning-rate / weight-decay schedules exactly where they left off.
The schedule's warm-restart cycles are 1, 2, 4, 8, 16, ... epochs, so cycles end at epochs
1, 3, 7, 15, 31, 63; the state at each of these is kept permanently in
``<ckpt_dir>/cycle-<epoch>/`` (restore one with ``--resume-from``), while ``<ckpt_dir>/ckpt-*``
is the rolling latest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common_functions"))

DEFAULTS = dict(
    task="multiclass",  # binary | multiclass | cifar10 | imagenet32
    mode="baseline",  # baseline | A | B
    impl="new",  # new (qks_tn_ff sweep) | old (original TNLayer; baseline only)
    epochs=15,
    batch_size=32,
    shots=None,
    share_per_layer=True,
    seed=0,
    ff_layers=None,  # e.g. "1,2,3,4,5"
    ff_features="raw",
    ff_null=None,
    freeze_qks=False,
    fast_qks=True,
    limit_train=None,
    eval_every_epoch=True,
    intra_threads=0,
    inter_threads=0,
    cache_dir=os.path.join(HERE, "data_cache"),
    out=None,
    ckpt_dir=None,  # directory for per-epoch tf.train.Checkpoint (None = no checkpointing)
    resume=False,  # resume from the latest checkpoint in ckpt_dir if one exists
    resume_from=None,  # explicit checkpoint prefix to restore instead (e.g. <ckpt_dir>/cycle-015/ckpt)
    name=None,
    nepisodes=128,
    chi=4,
    sigma=0.125,  # QKS weight std for 784-dim inputs; scaled by sqrt(784/p) for other input sizes
    head="auto",  # const (paper's fixed 0/1 readout, <=10 classes) | linear (trainable logits) | auto
    head_scale=None,  # W17: linear head only; multiply the readout probabilities by this constant before the Dense layer (None = no scaling layer, old behavior)
    # LeNet-style encoder (qks_tn_lenet.PatchQKSEncodingLayer); leaves = positions x groups
    encoder="dense",  # dense (paper's QKS) | patch (local filters) | twostage (patch -> tree -> re-encode -> tree)
    patch=None,  # receptive field (default per image size: 7 for 28x28, 4 for 32x32)
    stride=None,  # default 3 for 28x28, 4 for 32x32 -> 8x8 positions
    pad=None,  # symmetric zero padding (default 0)
    groups=1,  # channel groups (chi=4 registers) per position
    leaf_order="morton",  # morton (quadtree merges) | raster
    share_positions=True,  # False = locally connected control (separate filters per position)
    sigma_scale=1.0,  # multiplier on the dimensional sigma rule for the patch encoder
    shuffle_buffer=None,  # index-shuffle buffer; None = full permutation every epoch (indices only, cheap)
    lr=1e-3,
    wd=4e-4,
    kraus_rank=None,  # None = unitary TTN; R >= chi = CPTP vertices of Kraus rank R (qks_tn_cptp)
    root_rank=None,  # CPTP root vertex environment dimension (None = chi, unitary root); readout has chi*root_rank outcomes
    root_fold=False,  # fold the chi*root_rank root outcomes back to chi**2 readout outcomes (parameters without width)
    cptp_init="embed",  # embed (unitary embedded in the R-dim dilation) | random
    cptp_init_noise=1e-2,
    ff_dim=None,  # CPTP layer: feedforward features per non-root vertex (None = 2R); larger appends monomials of the environment probabilities
    root_ff_dim=None,  # same for the root vertex (None = ff_dim)
    ff_map="monomial",  # extra features: monomial, trainable hidden projection, or fixed random projection
    ff_width=None,  # trainable linear compression of 2R child probabilities to a fixed width
    meas_bases=None,  # CPTP layer: K measurement bases of each discarded register (None = computational basis only); each child passes K*R features
    stoch=None,  # classical-stochastic twin of the CPTP tree (qks_tn_stoch): "full" (leaves dephased) | "c0" (layer 0 coherent CPTP) | "c2", "c3" (W16: first 2 / 3 layers coherent CPTP)
    stoch_init_scale=0.3,  # W12: init scale of the classical node (StochTNLayer.init_scale); 0.3 = old default
    leaf_dephase=None,  # W20: p in (0, 1]: leaf registers rho -> (1 - p) rho + p diag(rho) before layer 0 (CPTP layer); None = unchanged
    ff_stop_grad=False,  # W27: tf.stop_gradient on the CPTP layer's feedforward features (forward unchanged; children get gradient only through the kept state)
    schedule="restarts",  # W18: restarts = CosineDecayRestarts (cycles 1,2,4,.. epochs, no warmup) | cosine1 = one cosine decay to 0 over all epochs
    ff_noise=None,  # W33/W44: training-only Gaussian noise std on the CPTP layer's feedforward features (None = off)
    ff_dropout=None,  # W33: training-only dropout rate on the feedforward features (None = off)
    ff_mix=None,  # W33: training-only feature-wise mixing rate (negative control; None = off)
    augment=None,  # W34: training-only image augmentation applied before the encoder, "+"-joined tokens from flip, crop (4-px zero pad), cutout (8x8, p = 0.5); None = off (old behavior)
)

CONFIGS = {
    # name: overrides on top of DEFAULTS (task/seed/epochs supplied separately)
    "C0": dict(mode="baseline", share_per_layer=True),
    "C1": dict(mode="baseline", share_per_layer=False),
    "C2": dict(mode="B", share_per_layer=True),
    "C3": dict(mode="B", share_per_layer=True, ff_null="constant"),
    "C4": dict(mode="B", share_per_layer=False),
    "C5": dict(mode="A", share_per_layer=True),
    "C6": dict(mode="A", share_per_layer=False),
    "F0": dict(mode="baseline", share_per_layer=True, freeze_qks=True),
    "F2": dict(mode="B", share_per_layer=True, freeze_qks=True),
    "F5": dict(mode="A", share_per_layer=True, freeze_qks=True),
    # phase-2 ablations
    "C2s16": dict(mode="B", share_per_layer=True, shots=16),
    "C2s128": dict(mode="B", share_per_layer=True, shots=128),
    "C2s1024": dict(mode="B", share_per_layer=True, shots=1024),
    "C2root": dict(mode="B", share_per_layer=True, ff_layers="5"),
    "C2top2": dict(mode="B", share_per_layer=True, ff_layers="4,5"),
    "C2c": dict(mode="B", share_per_layer=True, ff_features="centered"),
    "C0old": dict(mode="baseline", share_per_layer=True, impl="old", fast_qks=False),
    # LeNet-style: patch-local position-shared kitchen sinks, Morton (quadtree) leaf order
    "L0": dict(encoder="patch", leaf_order="morton", mode="baseline", share_per_layer=True),
    "L1": dict(encoder="patch", leaf_order="morton", mode="baseline", share_per_layer=False),
    "L2": dict(encoder="patch", leaf_order="morton", mode="B", share_per_layer=True),
    "L4": dict(encoder="patch", leaf_order="morton", mode="B", share_per_layer=False),
    # controls: raster leaf order (same weights, no 2-D locality in the tree) ...
    "R1": dict(encoder="patch", leaf_order="raster", mode="baseline", share_per_layer=False),
    "R4": dict(encoder="patch", leaf_order="raster", mode="B", share_per_layer=False),
    # ... and locally connected filters (same receptive fields, no weight sharing)
    "U1": dict(encoder="patch", leaf_order="morton", share_positions=False, mode="baseline", share_per_layer=False),
    "U4": dict(encoder="patch", leaf_order="morton", share_positions=False, mode="B", share_per_layer=False),
    # two-stage "deep kitchen sinks": conv -> pool -> re-encode measured statistics -> conv -> pool
    "T1": dict(encoder="twostage", mode="baseline", share_per_layer=False),
    "T4": dict(encoder="twostage", mode="B", share_per_layer=False),
    # trainable-logits head variants (the paper's fixed readout is the default)
    "L4lin": dict(encoder="patch", leaf_order="morton", mode="B", share_per_layer=False, head="linear"),
    "C4lin": dict(mode="B", share_per_layer=False, head="linear"),
    # CPTP vertices (autograd on Stinespring isometries), Kraus rank R; chi = 4 -> R = 4 is the unitary TTN
    **{f"K{R}{suf}": dict(kraus_rank=R, mode=m, share_per_layer=sh)
       for R in (4, 8, 16, 32)
       for suf, m, sh in (("", "baseline", True), ("p", "baseline", False), ("B", "B", True), ("pB", "B", False))},
    # CPTP with the QKS encoder frozen (TN-only learning), to compare with F0 / F2
    **{f"FK{R}{suf}": dict(kraus_rank=R, mode=m, share_per_layer=True, freeze_qks=True)
       for R in (8, 16) for suf, m in (("", "baseline"), ("B", "B"))},
    # parameter-matched frozen controls: unitary per-node + feedforward, and per-node rank 8 + feedforward
    "F4": dict(mode="B", share_per_layer=False, freeze_qks=True),
    "FK8pB": dict(kraus_rank=8, mode="B", share_per_layer=False, freeze_qks=True),
    # Haar-random initialization of the dilation (instead of the embedded unitary)
    **{f"RK{R}{suf}": dict(kraus_rank=R, mode=m, share_per_layer=True, cptp_init="random")
       for R in (8, 16) for suf, m in (("", "baseline"), ("B", "B"))},
    # wide root: the root vertex has Kraus rank W and the readout has chi*W outcomes
    **{f"K{R}{suf}W{W}": dict(kraus_rank=R, root_rank=W, mode=m, share_per_layer=True)
       for R in (8,) for W in (8, 16) for suf, m in (("", "baseline"), ("B", "B"))},
    # CPTP with the LeNet-style patch encoder (CIFAR-10 / chi = 8): controls are L0, L1, L2, L4
    **{f"LK{R}{suf}": dict(encoder="patch", leaf_order="morton", kraus_rank=R, mode=m, share_per_layer=sh)
       for R in (4, 8, 16, 32)
       for suf, m, sh in (("", "baseline", True), ("p", "baseline", False), ("B", "B", True), ("pB", "B", False))},
    **{f"LK{R}B": dict(encoder="patch", leaf_order="morton", kraus_rank=R, mode="B", share_per_layer=True)
       for R in (12, 24)},
    **{f"LK{R}Bfw16": dict(encoder="patch", leaf_order="morton", kraus_rank=R,
                            mode="B", share_per_layer=True, ff_width=16)
       for R in (8, 16, 32)},
    "LRK16B": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, cptp_init="random"),
    **{f"LK8BW{W}": dict(encoder="patch", leaf_order="morton", kraus_rank=8, root_rank=W, mode="B", share_per_layer=True)
       for W in (8, 16)},
    # root parameters without readout width: rank-W root, outcomes folded back to chi**2
    **{f"LK8BW{W}F": dict(encoder="patch", leaf_order="morton", kraus_rank=8, root_rank=W, root_fold=True, mode="B", share_per_layer=True)
       for W in (8, 16)},
    # chi = 8: rank 32 (4 chi) with feedforward is LK32B; baseline controls L0 / L2 / L4
    # parameter-matched unitary controls (kraus_rank = chi, so no environment): feedforward features extended by
    # monomials of the discarded-register probabilities
    "LU8Bf112": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=112),  # vs LK16B (chi = 8)
    "LU8pBf16R22": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=False, root_ff_dim=22),
    "LU8Bf187L012": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=187, ff_layers="0,1,2"),
    "LU8Bhidden111": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=111, ff_map="hidden"),
    "LU8BmbK4": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, meas_bases=4),  # W7: unitary node, 4 measurement bases (vs LK16B, LU8Bf112)
    "LU8BmbK2": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, meas_bases=2),
    "LU8BmbK7": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, meas_bases=7),  # W10: 56 features per child, 112 per vertex (same vertex feature count as LU8Bf112)
    "LU8BmbK8": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, meas_bases=8),  # W10 secondary
    "LU8Brandom112": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=112, ff_map="random"),
    # same parameters, different allocation: feedforward only at the top three layers, 187 features each
    "LU8Bf187L345": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=187, ff_layers="3,4,5"),
    "LU8Bf847": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=847),  # vs LK32B (chi = 8)
    # W14 (timn100): unitary control for LK16B with a rank-16 root (540,639 root params): 131 root features -> 540,540 root params; 132 features at layers 1-4 -> total within 0.2% of LK16B + root_rank 16
    "LU8Bf132R131": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=132, root_ff_dim=131),
    "LK16BW16": dict(encoder="patch", leaf_order="morton", kraus_rank=16, root_rank=16, mode="B", share_per_layer=True),  # W14: rank-16 root, chi*16 = 128 readout outcomes (use --head const for the fixed readout)
    "LK16BW16F": dict(encoder="patch", leaf_order="morton", kraus_rank=16, root_rank=16, root_fold=True, mode="B", share_per_layer=True),  # W14: same root parameters, outcomes folded to 64
    "LK8BRf272": dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, root_ff_dim=272),  # unitary root vs LK8BW16F (chi = 4)
    # classical-stochastic twins of LK16B (chi = 8, R = 16, shared, mode B): W8
    "LS16B": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, stoch="full"),
    "LS16Bc0": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, stoch="c0"),
    "LS16Bf65": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, stoch="full", ff_dim=65, root_ff_dim=32),
    # W16: first k layers coherent rank-16 CPTP vertices, layers k..5 classical (k = 1 is LS16Bc0)
    "LS16Bck2": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, stoch="c2"),
    "LS16Bck3": dict(encoder="patch", leaf_order="morton", kraus_rank=16, mode="B", share_per_layer=True, stoch="c3"),
}
# W20: partial leaf dephasing p, W5-selected lr/wd per model (LK16B lr 0.002 wd 4e-4; LU8Bf112 lr 0.001 wd 4e-4); p = 0 is the base config
for _p, _t in ((0.25, "025"), (0.5, "05"), (0.75, "075"), (1.0, "1")):
    CONFIGS[f"LK16Bd{_t}"] = dict(CONFIGS["LK16B"], lr=2e-3, leaf_dephase=_p)
    CONFIGS[f"LU8Bf112d{_t}"] = dict(CONFIGS["LU8Bf112"], lr=1e-3, leaf_dephase=_p)

# W12: classical arms with init scale 1.0 (the LK16B CPTP layer's scale); the init scale enters the run name only when it differs from 0.3
for _b in ("LS16B", "LS16Bc0", "LS16Bf65"):
    CONFIGS[_b + "i1"] = dict(CONFIGS[_b], stoch_init_scale=1.0)
# W17 (timn100): classical-root width matched to the unitary control (root_ff_dim 131 -> 132 * 4096 = 540,672 root params)
CONFIGS["LS16Bc0R131"] = dict(CONFIGS["LS16Bc0"], root_ff_dim=131)
# W26: LS16Bc0 with ff_dim 65 at layers 1-4, root 32: 2,314,239 TN params (LK16B 2,314,074, +0.007%)
CONFIGS["LS16Bc0f65"] = dict(CONFIGS["LS16Bc0"], ff_dim=65, root_ff_dim=32)
# W17: shared linear readout (64 outcomes -> 100 logits) with a constant gain on the probabilities; name suffix H<gain>
for _b in ("LK16BW16F", "LU8Bf132R131", "LS16Bc0R131"):
    for _h in (1, 4, 16, 64, 256, 1024):
        CONFIGS[f"{_b}H{_h}"] = dict(CONFIGS[_b], head="linear", head_scale=float(_h))

# W19 (rank at fixed ~2.3M TN parameters, chi 8, shared, mode B): LK12Bf55 2,302,695; LK24Bfw14 2,313,428; LK32Bfw7 2,197,655 (-4.45%); R=8 is LU8Bf112, R=16 is LK16B
CONFIGS["LK12Bf55"] = dict(encoder="patch", leaf_order="morton", kraus_rank=12, mode="B", share_per_layer=True, ff_dim=55)
CONFIGS["LK24Bfw14"] = dict(encoder="patch", leaf_order="morton", kraus_rank=24, mode="B", share_per_layer=True, ff_width=14)
CONFIGS["LK32Bfw7"] = dict(encoder="patch", leaf_order="morton", kraus_rank=32, mode="B", share_per_layer=True, ff_width=7)

# W24 (feedforward width 32 at layers 1-4 for every R; root width sets the parameters): R=8 2,301,390; R=12 2,298,485; R=16 is LK16B (2,314,074); R=24 5,045,594 (+119%, cannot fit at F=32)
CONFIGS["LK8Bf32R428"] = dict(encoder="patch", leaf_order="morton", kraus_rank=8, mode="B", share_per_layer=True, ff_dim=32, root_ff_dim=428)
CONFIGS["LK12Bf32R261"] = dict(encoder="patch", leaf_order="morton", kraus_rank=12, mode="B", share_per_layer=True, ff_dim=32, root_ff_dim=261)
CONFIGS["LK24Bfw32"] = dict(encoder="patch", leaf_order="morton", kraus_rank=24, mode="B", share_per_layer=True, ff_width=32)

# W27 (feedforward channel ablations of LK16B at its W5 setting lr 0.002, wd 4e-4): stop-gradient on the features; features shuffled across the batch
CONFIGS["LK16Bsg"] = dict(CONFIGS["LK16B"], lr=2e-3, ff_stop_grad=True)
CONFIGS["LK16Bshuf"] = dict(CONFIGS["LK16B"], lr=2e-3, ff_null="shuffle")

# W34 (training-only augmentation before the encoder; CIFAR-10): af = flip, afc = flip + crop (4-px pad), afco = flip + crop + cutout 8x8. lr/wd come from the command line.
# W36 adds the classical hybrid LS16Bc0f65 to the family (LS16Bc0f65af, LS16Bc0f65afc, LS16Bc0f65afco).
for _base in ("LK16B", "LU8Bf112", "LS16Bc0f65"):
    for _sfx, _aug in (("af", "flip"), ("afc", "flip+crop"), ("afco", "flip+crop+cutout")):
        CONFIGS[f"{_base}{_sfx}"] = dict(CONFIGS[_base], augment=_aug)

# W33 (training-only feedforward-channel regularizers, ported in W44; lr/wd come from the command line)
CONFIGS["LK16Bn01"] = dict(CONFIGS["LK16B"], ff_noise=0.01)
CONFIGS["LK16Bn03"] = dict(CONFIGS["LK16B"], ff_noise=0.03)
CONFIGS["LK16Bdo10"] = dict(CONFIGS["LK16B"], ff_dropout=0.1)
CONFIGS["LK16Bdo25"] = dict(CONFIGS["LK16B"], ff_dropout=0.25)
CONFIGS["LK16Bmx10"] = dict(CONFIGS["LK16B"], ff_mix=0.1)

# W44: flip augmentation with training-only Gaussian noise on the feedforward features (std 0.03; lr/wd come from the command line)
CONFIGS["LK16Bn03af"] = dict(CONFIGS["LK16B"], ff_noise=0.03, augment="flip")

# W55: flip augmentation with ff noise std 0.01
CONFIGS["LK16Bn01af"] = dict(CONFIGS["LK16B"], ff_noise=0.01, augment="flip")

# W60: the unitary control and the classical hybrid with the same flip + ff noise std 0.01 (lr/wd come from the command line)
CONFIGS["LU8Bf112n01af"] = dict(CONFIGS["LU8Bf112"], ff_noise=0.01, augment="flip")
CONFIGS["LS16Bc0f65n01af"] = dict(CONFIGS["LS16Bc0f65"], ff_noise=0.01, augment="flip")

# Registered A2 grid. Each name fixes one base model and one (lr, wd) factor pair.
for _base in ("LK16B", "L4", "LU8Brandom112"):
    for _lf, _lr in (("05", 5e-4), ("1", 1e-3), ("2", 2e-3)):
        for _wf, _wd in (("0", 0.0), ("1", 4e-4), ("3", 1.2e-3)):
            CONFIGS[f"A2{_base}l{_lf}w{_wf}"] = dict(CONFIGS[_base], lr=_lr, wd=_wd)


def make_cfg(**overrides) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg["name"] is None:
        parts = [cfg["task"], cfg["mode"], "shared" if cfg["share_per_layer"] else "pervertex"]
        if cfg.get("holdout_5000"):
            parts.append("holdout5000")
        if cfg["kraus_rank"]:
            parts.append(f"kraus{cfg['kraus_rank']}")
        if cfg["root_rank"]:
            parts.append(f"root{cfg['root_rank']}" + ("fold" if cfg["root_fold"] else ""))
        if cfg["ff_dim"]:
            parts.append(f"ffd{cfg['ff_dim']}")
        if cfg["root_ff_dim"]:
            parts.append(f"rffd{cfg['root_ff_dim']}")
        if cfg["ff_width"]:
            parts.append(f"ffw{cfg['ff_width']}")
        if cfg.get("meas_bases"):
            parts.append(f"mb{cfg['meas_bases']}")
        if cfg.get("leaf_dephase"):
            parts.append(f"dph{cfg['leaf_dephase']}")
        if cfg.get("ff_stop_grad"):
            parts.append("sg")
        if cfg.get("stoch"):
            parts.append(f"stoch-{cfg['stoch']}")
            if float(cfg["stoch_init_scale"]) != 0.3:
                parts.append(f"is{cfg['stoch_init_scale']}")
        if cfg["ff_map"] != "monomial":
            parts.append(f"ffmap-{cfg['ff_map']}")
        if cfg.get("schedule", "restarts") != "restarts":
            parts.append(f"sched-{cfg['schedule']}")
        if cfg.get("ff_noise"):
            parts.append(f"ffn{cfg['ff_noise']}")
        if cfg.get("ff_dropout"):
            parts.append(f"ffdo{cfg['ff_dropout']}")
        if cfg.get("ff_mix"):
            parts.append(f"ffmx{cfg['ff_mix']}")
        if cfg.get("augment"):
            parts.append(f"aug-{cfg['augment']}")
        if float(cfg["lr"]) != float(DEFAULTS["lr"]):
            parts.append(f"lr{cfg['lr']}")
        if float(cfg["wd"]) != float(DEFAULTS["wd"]):
            parts.append(f"wd{cfg['wd']}")
        if cfg["kraus_rank"] and cfg["cptp_init"] != "embed":
            parts.append(f"init-{cfg['cptp_init']}")
        if cfg["ff_null"]:
            parts.append(f"null-{cfg['ff_null']}")
        if cfg["shots"]:
            parts.append(f"shots{cfg['shots']}")
        if cfg["ff_layers"]:
            parts.append("ffl" + str(cfg["ff_layers"]).replace(",", ""))
        if cfg["ff_features"] != "raw":
            parts.append(cfg["ff_features"])
        if cfg["freeze_qks"]:
            parts.append("frozenqks")
        if cfg["impl"] != "new":
            parts.append(cfg["impl"])
        if cfg["encoder"] in ("patch", "twostage"):
            if cfg["encoder"] == "twostage":
                parts.append("twostage")
            geo = patch_geometry(cfg)
            parts.append(f"patch{geo['patch']}s{geo['stride']}" + (f"p{geo['pad']}" if geo["pad"] else ""))
            parts.append(f"g{cfg['groups']}")
            parts.append(cfg["leaf_order"])
            if not cfg["share_positions"]:
                parts.append("unshared")
            if float(cfg["sigma_scale"]) != 1.0:
                parts.append(f"sig{cfg['sigma_scale']}")
        elif int(cfg["nepisodes"]) != 128:
            parts.append(f"n{cfg['nepisodes']}")  # 128 episodes = 6 tree layers, 256 = 7, 512 = 8
        if int(cfg["chi"]) != 4:
            parts.append(f"chi{cfg['chi']}")
        if cfg.get("head") not in (None, "auto"):
            parts.append(f"head-{cfg['head']}")
        if cfg.get("head_scale"):
            parts.append(f"hs{cfg['head_scale']}")
        parts.append(f"s{cfg['seed']}")
        cfg["name"] = "_".join(parts)
    return cfg


def _parse_layers(v):
    if v is None or v == "" or v == "all":
        return None
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(",") if x.strip()]


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
N_CLASSES = {"binary": 2, "multiclass": 10, "cifar10": 10, "imagenet32": 1000, "timn100": 100, "svhn": 10}
IMAGE_SHAPES = {"binary": (28, 28, 1), "multiclass": (28, 28, 1), "cifar10": (32, 32, 3), "imagenet32": (32, 32, 3), "timn100": (32, 32, 3), "svhn": (32, 32, 3)}
PATCH_DEFAULTS = {(28, 28): dict(patch=7, stride=3, pad=0), (32, 32): dict(patch=4, stride=4, pad=0)}


def patch_geometry(cfg: dict) -> dict:
    """Resolved (patch, stride, pad) for the patch encoder on cfg['task']."""
    H, W, C = IMAGE_SHAPES[cfg["task"]]
    geo = dict(PATCH_DEFAULTS[(H, W)])
    for k in ("patch", "stride", "pad"):
        if cfg.get(k) is not None:
            geo[k] = int(cfg[k])
    return geo
HF_IMAGENET32 = "benjamin-paine/imagenet-1k-32x32"  # ImageNet-1k, centre-crop + Lanczos to 32x32
SVHN_URLS = {"train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"}  # official files; the 'extra' split is not used


def _svhn_mat_to_uint8(path: str):
    """Official SVHN cropped-digit .mat (X: (32, 32, 3, n) uint8, y: (n, 1) in 1..10 with 10 = digit 0)
    -> X uint8 (n, 3072) channels-last row-major (same layout as CIFAR-10), y int16 in 0..9."""
    import numpy as np
    from scipy.io import loadmat

    m = loadmat(path)
    X = np.asarray(m["X"], dtype=np.uint8)
    y = np.asarray(m["y"]).reshape(-1).astype(np.int16)
    if X.ndim != 4 or X.shape[:3] != (32, 32, 3) or X.shape[3] != len(y):
        raise ValueError(f"unexpected SVHN array shapes {X.shape}, {y.shape}")
    y = np.where(y == 10, 0, y).astype(np.int16)
    if y.min() < 0 or y.max() > 9:
        raise ValueError("SVHN labels outside 1..10")
    return np.ascontiguousarray(X.transpose(3, 0, 1, 2)).reshape(len(y), -1), y


HF_SVHN = "ufldl-stanford/svhn"  # ungated HF mirror of the official cropped-digit data (config cropped_digits)


def _svhn_split_hf(split: str, log=print):
    """Fallback when the official .mat host is unreachable: the ungated HF mirror's standard parquet file for
    ``split`` (the 'extra' files are never fetched). Labels are already 0..9. Returns X uint8 (n, 3072), y int16."""
    import io

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    fn = hf_hub_download(HF_SVHN, f"cropped_digits/{split}-00000-of-00001.parquet", repo_type="dataset")
    tab = pq.read_table(fn).to_pydict()
    n = len(tab["label"])
    X = np.empty((n, 3072), dtype=np.uint8)
    for i, im in enumerate(tab["image"]):
        X[i] = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"), dtype=np.uint8).reshape(-1)
    return X, np.asarray(tab["label"], dtype=np.int16)


def _svhn_split(split: str, raw_dir: str, log=print):
    """Official .mat (30 s connect timeout), else the HF mirror; checks the official counts and label range."""
    import shutil
    import urllib.request

    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{split}_32x32.mat")
    X = None
    try:
        if not os.path.exists(path):
            log(f"downloading {SVHN_URLS[split]}")
            with urllib.request.urlopen(SVHN_URLS[split], timeout=30) as r, open(path + ".tmp", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(path + ".tmp", path)
        X, y = _svhn_mat_to_uint8(path)
        source = "official .mat"
    except OSError as exc:  # URLError, timeouts and refused connections are OSError subclasses
        log(f"official SVHN host unreachable ({exc!r}); using the HF mirror {HF_SVHN}")
    if X is None:
        X, y = _svhn_split_hf(split, log)
        source = f"HF mirror {HF_SVHN}"
    expect = {"train": 73257, "test": 26032}[split]
    if len(y) != expect or X.shape != (expect, 3072) or y.min() < 0 or y.max() > 9:
        raise ValueError(f"SVHN {split}: got X {X.shape}, y [{y.min()}, {y.max()}] n={len(y)}; expected {expect} images, labels 0..9")
    log(f"svhn {split}: {len(y)} images from {source}")
    return X, y


HF_TIMN = "zh-plus/tiny-imagenet"  # ungated Tiny-ImageNet: 200 classes, 64x64, 500 train + 50 validation images per class


def _timn100_split(split: str, log=print):
    """First 100 classes (sorted wnid order = HF label ids 0..99) of a Tiny-ImageNet split, 64x64 -> 32x32
    by 2x2 area averaging (rounded to uint8). Returns X uint8 (n, 3072), y int16."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(HF_TIMN, split=split)
    names = ds.features["label"].names
    if list(names) != sorted(names):
        raise RuntimeError("Tiny-ImageNet label names are not in sorted wnid order; 'first 100 by sorted label' is ambiguous")
    labels = np.asarray(ds["label"], dtype=np.int16)
    idx = np.flatnonzero(labels < 100)
    X = np.empty((len(idx), 32 * 32 * 3), dtype=np.uint8)
    for j, i in enumerate(idx):
        a = np.asarray(ds[int(i)]["image"].convert("RGB"), dtype=np.float32)  # (64, 64, 3)
        a = a.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))
        X[j] = np.rint(a).astype(np.uint8).reshape(-1)
        if j % 10_000 == 0 and j:
            log(f"  timn100 {split}: {j}/{len(idx)}")
    log(f"timn100 {split}: {len(idx)} images, classes {names[0]} .. {names[99]}")
    return X, labels[idx]


def _hf_split_to_uint8(repo: str, split: str, log=print):
    """Decode an image-classification HF split to (X uint8 (n, 3072), y int16)."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    label_col = "class_label" if "class_label" in ds.column_names else "label"
    n = len(ds)
    X = np.empty((n, 32 * 32 * 3), dtype=np.uint8)
    y = np.asarray(ds[label_col], dtype=np.int16)
    t0 = time.time()
    for i, img in enumerate(ds["image"] if n < 200_000 else (ex["image"] for ex in ds)):
        X[i] = np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1)
        if i % 200_000 == 0 and i:
            log(f"  decoded {i}/{n} ({time.time() - t0:.0f}s)")
    return X, y


def load_data(task: str, cache_dir: str, log=print):
    """Data pipeline per task.

    binary / multiclass : the notebooks' MNIST pipeline (/255, float32, deskew); returns float32.
    cifar10             : tf.keras CIFAR-10, (n, 3072) uint8 channels-last (scaled to [0,1] in the
                          tf.data pipeline); 10 classes, fixed readout as for MNIST.
    imagenet32          : HF ``benjamin-paine/imagenet-1k-32x32`` train (1,281,167) / validation
                          (50,000), (n, 3072) uint8; 1000 classes -> trainable linear head.
    """
    import numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    if task in ("cifar10", "imagenet32", "timn100", "svhn"):
        d = os.path.join(cache_dir, task)
        files = [os.path.join(d, f) for f in ("X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy")]
        if not all(os.path.exists(f) for f in files):
            os.makedirs(d, exist_ok=True)
            if task == "cifar10":
                from tensorflow.keras.datasets import cifar10

                (Xtr, ytr), (Xte, yte) = cifar10.load_data()
                Xtr = Xtr.reshape(len(Xtr), -1).astype(np.uint8)
                Xte = Xte.reshape(len(Xte), -1).astype(np.uint8)
                ytr, yte = ytr.reshape(-1).astype(np.int16), yte.reshape(-1).astype(np.int16)
            elif task == "svhn":
                Xtr, ytr = _svhn_split("train", os.path.join(d, "raw"), log)
                Xte, yte = _svhn_split("test", os.path.join(d, "raw"), log)
            elif task == "timn100":
                log(f"downloading + decoding {HF_TIMN} (train, first 100 classes)")
                Xtr, ytr = _timn100_split("train", log)
                log(f"downloading + decoding {HF_TIMN} (valid, first 100 classes; used as the test set)")
                Xte, yte = _timn100_split("valid", log)
            else:
                log(f"downloading + decoding {HF_IMAGENET32} (train)")
                Xtr, ytr = _hf_split_to_uint8(HF_IMAGENET32, "train", log)
                log(f"downloading + decoding {HF_IMAGENET32} (validation)")
                Xte, yte = _hf_split_to_uint8(HF_IMAGENET32, "validation", log)
                keep = yte < 1000  # label 1000 = "none" (only in the unlabeled test split)
                Xte, yte = Xte[keep], yte[keep]
            for f, a in zip(files, (Xtr, ytr, Xte, yte)):
                np.save(f + ".tmp.npy", a)
                os.replace(f + ".tmp.npy", f)
        X_train, y_train, X_test, y_test = (np.load(f, mmap_mode="r") for f in files)
        return np.ascontiguousarray(X_train), np.asarray(y_train).astype("float32"), np.ascontiguousarray(X_test), np.asarray(y_test).astype("float32")

    cache = os.path.join(cache_dir, f"mnist_deskew_{task}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X_train"], z["y_train"], z["X_test"], z["y_test"]
    from tensorflow.keras.datasets import mnist

    from general_functions import deskewAll

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if task == "binary":
        mtr = (y_train == 3) | (y_train == 5)
        mte = (y_test == 3) | (y_test == 5)
        X_train, y_train, X_test, y_test = X_train[mtr], y_train[mtr], X_test[mte], y_test[mte]
        y_train = np.where(y_train == 3, 0, 1)
        y_test = np.where(y_test == 3, 0, 1)
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255
    X_train = deskewAll(X_train.astype("float32")).astype("float32")
    X_test = deskewAll(X_test.astype("float32")).astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    tmp = cache + ".tmp.npz"
    np.savez(tmp, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    os.replace(tmp, cache)
    return X_train, y_train, X_test, y_test


def split_cifar10_holdout(X_train, y_train):
    """W4's fixed stratified CIFAR-10 split: 500 images per class, seed 20261102."""
    import numpy as np

    if len(y_train) != 50000 or len(X_train) != 50000:
        raise ValueError("--holdout-5000 requires the full CIFAR-10 training set")
    rng = np.random.default_rng(20261102)
    val_parts = [rng.permutation(np.flatnonzero(y_train == c))[:500] for c in range(10)]
    val_idx = np.sort(np.concatenate(val_parts))
    keep = np.ones(len(y_train), dtype=bool)
    keep[val_idx] = False
    return X_train[keep], y_train[keep], X_train[val_idx], y_train[val_idx], val_idx


def split_svhn_holdout(X_train, y_train):
    """Stratified SVHN selection holdout: 500 of the standard-train images per class (5,000 total), seed 20261102.
    The remaining 68,257 images are the holdout-protocol training set."""
    import numpy as np

    if len(y_train) != 73257 or len(X_train) != 73257:
        raise ValueError("--holdout-5000 requires the full SVHN training set (73,257 images)")
    rng = np.random.default_rng(20261102)
    val_parts = [rng.permutation(np.flatnonzero(y_train == c))[:500] for c in range(10)]
    val_idx = np.sort(np.concatenate(val_parts))
    keep = np.ones(len(y_train), dtype=bool)
    keep[val_idx] = False
    return X_train[keep], y_train[keep], X_train[val_idx], y_train[val_idx], val_idx


def split_timn100_holdout(X_train, y_train):
    """Stratified timn100 selection holdout: 50 of the 500 training images per class (5,000 total), seed 20261102."""
    import numpy as np

    if len(y_train) != 50000 or len(X_train) != 50000:
        raise ValueError("--holdout-5000 requires the full timn100 training set")
    rng = np.random.default_rng(20261102)
    val_parts = [rng.permutation(np.flatnonzero(y_train == c))[:50] for c in range(100)]
    val_idx = np.sort(np.concatenate(val_parts))
    keep = np.ones(len(y_train), dtype=bool)
    keep[val_idx] = False
    return X_train[keep], y_train[keep], X_train[val_idx], y_train[val_idx], val_idx


AUGMENT_TOKENS = ("flip", "crop", "cutout")
AUGMENT_TASKS = {"cifar10": (32, 32, 3)}  # image shape (H, W, C) of the flat (n, H*W*C) rows; channels-last


def parse_augment(spec):
    """"flip+crop" -> ("flip", "crop"); None or "" or "none" -> (). Unknown tokens raise."""
    if not spec or spec == "none":
        return ()
    toks = tuple(t for t in str(spec).split("+") if t)
    bad = [t for t in toks if t not in AUGMENT_TOKENS]
    if bad:
        raise ValueError(f"unknown augment token(s) {bad}; allowed {AUGMENT_TOKENS}")
    return toks


def augment_images(x, spec, shape=(32, 32, 3), pad=4, cutout_size=8, cutout_p=0.5):
    """Per-sample random augmentation of a float batch x of shape (B, H*W*C) in [0, 1]; returns the same shape.

    flip   : horizontal flip with probability 0.5 per image.
    crop   : zero-pad by ``pad`` pixels on every side, then take a random HxW window (independent offsets per image).
    cutout : with probability ``cutout_p`` per image, set a ``cutout_size`` x ``cutout_size`` square (random center, clipped
             at the border) to 0.
    Order: flip, crop, cutout. Randomness comes from TensorFlow's stateful ops, so every call draws fresh values."""
    import tensorflow as tf

    toks = parse_augment(spec)
    if not toks:
        return x
    H, W, C = shape
    B = tf.shape(x)[0]
    x = tf.reshape(x, [B, H, W, C])
    if "flip" in toks:
        m = tf.random.uniform([B]) < 0.5
        x = tf.where(m[:, None, None, None], x[:, :, ::-1, :], x)
    if "crop" in toks:
        xp = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        oy = tf.random.uniform([B], 0, 2 * pad + 1, dtype=tf.int32)
        ox = tf.random.uniform([B], 0, 2 * pad + 1, dtype=tf.int32)
        rows = oy[:, None] + tf.range(H)[None, :]
        cols = ox[:, None] + tf.range(W)[None, :]
        xp = tf.gather(xp, rows, axis=1, batch_dims=1)
        x = tf.gather(xp, cols, axis=2, batch_dims=1)
    if "cutout" in toks:
        on = tf.random.uniform([B]) < cutout_p
        cy = tf.random.uniform([B], 0, H, dtype=tf.int32)
        cx = tf.random.uniform([B], 0, W, dtype=tf.int32)
        h = cutout_size // 2
        ys = tf.range(H)[None, :, None]
        xs = tf.range(W)[None, None, :]
        inside = (ys >= (cy - h)[:, None, None]) & (ys < (cy - h + cutout_size)[:, None, None])
        inside &= (xs >= (cx - h)[:, None, None]) & (xs < (cx - h + cutout_size)[:, None, None])
        x = tf.where((inside & on[:, None, None])[..., None], tf.zeros_like(x), x)
    return tf.reshape(x, [B, H * W * C])


def make_dataset(X, y, batch_size: int, shuffle: bool, seed: int, shuffle_buffer=None, augment=None, image_shape=None):
    """tf.data pipeline backed by a CPU tf.Variable (sidesteps the 2 GB graph-constant limit).

    Indices are shuffled -- by default a *full* permutation every epoch (the buffer holds int64
    indices, so this is cheap even for 1.28M rows; a windowed shuffle would be catastrophic on
    class-sorted datasets such as the HF ImageNet-32 train split) -- and batches are gathered from
    the variable; uint8 images are scaled to [0, 1] on the fly.
    """
    import numpy as np
    import tensorflow as tf

    n = len(y)
    with tf.device("/CPU:0"):
        X_var = tf.Variable(X, trainable=False, name="X_data")
        y_var = tf.Variable(np.asarray(y, dtype="float32"), trainable=False, name="y_data")
    scale = X_var.dtype == tf.uint8

    def fetch(idx):
        xb = tf.gather(X_var, idx)
        if scale:
            xb = tf.cast(xb, tf.float32) / 255.0
        return xb, tf.gather(y_var, idx)

    ds = tf.data.Dataset.range(n)
    if shuffle:
        buf = n if not shuffle_buffer else min(n, int(shuffle_buffer))
        ds = ds.shuffle(buf, seed=seed, reshuffle_each_iteration=True)
    # training: drop the ragged last batch so steps/epoch == n // batch_size exactly -- the
    # CosineDecayRestarts cycles are defined in steps and would otherwise drift by one step per
    # epoch (15 epochs => the last 15 steps run at the restarted peak learning rate)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.map(fetch, num_parallel_calls=tf.data.AUTOTUNE)
    if parse_augment(augment):  # W34: training-only augmentation, after the gather and /255, before the encoder
        ds = ds.map(lambda xb, yb: (augment_images(xb, augment, image_shape), yb), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _make_schedule(cfg: dict, key: str, steps_per_epoch: int):
    """Learning-rate / weight-decay schedule for cfg[key] ("lr" or "wd").

    restarts (default): CosineDecayRestarts, first cycle = 1 epoch, cycle length doubling (t_mul 2), no warmup.
    cosine1: one cosine decay from cfg[key] to 0 over cfg["epochs"] * steps_per_epoch steps, no warmup, no restarts."""
    import tensorflow as tf

    sched = cfg.get("schedule", "restarts")
    if sched == "restarts":
        return tf.keras.optimizers.schedules.CosineDecayRestarts(cfg[key], steps_per_epoch)
    if sched == "cosine1":
        return tf.keras.optimizers.schedules.CosineDecay(cfg[key], int(cfg["epochs"]) * steps_per_epoch, alpha=0.0)
    raise ValueError(f"unknown schedule {sched!r}")


def build_model(cfg: dict, steps_per_epoch: int, p: int = 784, n_classes: int | None = None):
    import numpy as np
    import tensorflow as tf
    from tensorflow_addons.optimizers import AdamW

    import qks_tn_ff as ff

    n_classes = n_classes or N_CLASSES[cfg["task"]]
    sigma = float(cfg["sigma"]) * float(np.sqrt(784.0 / p))  # keep the kitchen-sink angle scale
    head_kind = cfg.get("head", "auto")
    if head_kind == "auto":
        head_kind = "const" if n_classes <= 10 else "linear"
    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    encoder_info = {}
    if cfg.get("encoder", "dense") == "twostage":
        import qks_tn_twostage as ts

        H, W, C = IMAGE_SHAPES[cfg["task"]]
        if p != H * W * C:
            raise ValueError(f"input dim {p} != {H}x{W}x{C}")
        geo = patch_geometry(cfg)
        sigma = float(cfg["sigma"]) * float(np.sqrt(784.0 / (geo["patch"] ** 2 * C))) * float(cfg["sigma_scale"])
        stage = ts.TwoStageQKSTTN(
            (H, W, C), sigma=sigma, chi=int(cfg["chi"]), groups=int(cfg["groups"]),
            mode=cfg["mode"], share_per_layer=cfg["share_per_layer"], seed=int(cfg["seed"]), **geo,
        )
        qks, tn_layer = stage.enc1, stage.tn2
        nlayers = stage.enc2.nlayers
        encoder_info = dict(encoder="twostage", blocks=stage.n_blocks, stage2_grid=list(stage.enc2.grid),
                            leaves=stage.enc1.n_leaves, **geo, groups=int(cfg["groups"]))
        n_outcomes = int(cfg["chi"]) ** 2
        if head_kind == "const":
            head = ff.ConstMulFixed(n_outcomes, n_classes)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            head = tf.keras.layers.Dense(n_classes, name="linear_head")
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(p,)), stage, head])
        if cfg["freeze_qks"]:
            stage.enc1.trainable = False
            stage.enc2.trainable = False
        lr_schedule = _make_schedule(cfg, "lr", steps_per_epoch)
        wd_schedule = _make_schedule(cfg, "wd", steps_per_epoch)
        model.compile(loss=loss, optimizer=AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule), metrics=["accuracy"])
        counts = dict(
            total=int(model.count_params()),
            qks=int(stage.enc1.count_params() + stage.enc2.count_params()),
            tn=int(stage.tn1.count_params() + stage.tn2.count_params()),
            tn_effective=int(stage.tn1.count_params() + stage.tn2.count_params()),
            head=int(head.count_params()),
            trainable=int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
            input_dim=int(p), n_classes=int(n_classes), sigma=float(sigma), head_kind=head_kind,
            chi=int(cfg["chi"]), n_outcomes=n_outcomes, nlayers=int(nlayers), **encoder_info,
        )
        return model, qks, tn_layer, counts

    if cfg.get("encoder", "dense") == "patch":
        import qks_tn_lenet as ln

        H, W, C = IMAGE_SHAPES[cfg["task"]]
        if p != H * W * C:
            raise ValueError(f"input dim {p} != {H}x{W}x{C}")
        geo = patch_geometry(cfg)
        sigma = float(cfg["sigma"]) * float(np.sqrt(784.0 / (geo["patch"] ** 2 * C))) * float(cfg["sigma_scale"])
        qks = ln.PatchQKSEncodingLayer(
            (H, W, C),
            sigma=sigma,
            groups=int(cfg["groups"]),
            share_positions=bool(cfg["share_positions"]),
            leaf_order=cfg["leaf_order"],
            bond_dim=int(cfg["chi"]),
            **geo,
        )
        nlayers = qks.nlayers
        encoder_info = dict(encoder="patch", grid=list(qks.grid), leaves=qks.n_leaves, **geo, groups=int(cfg["groups"]),
                            leaf_order=cfg["leaf_order"], share_positions=bool(cfg["share_positions"]))
    else:
        nlayers = int(np.log2(cfg["nepisodes"] / np.log2(cfg["chi"])))
        if cfg["fast_qks"]:
            qks = ff.FastFeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, sigma)
        else:
            qks = ff.FeatureEncodingLayer(cfg["nepisodes"], cfg["chi"], p, sigma)
        encoder_info = dict(encoder="dense", leaves=int(cfg["nepisodes"] // np.log2(cfg["chi"])))

    if cfg["impl"] == "old":
        if cfg["mode"] != "baseline":
            raise ValueError("impl='old' supports mode='baseline' only")
        import tensornetwork as tn

        import qks_tn as old

        tn.set_default_backend("tensorflow")
        rho0 = qks(tf.zeros((1, p))).numpy()
        uni_array = tf.constant(np.zeros([cfg["chi"]] * 4 + [nlayers]), dtype="complex64")
        obs = [rho0[0, :, :, j] for j in range(cfg["nepisodes"] // 2)]
        nodes_set, edge_order = old.construct_dttn(cfg["chi"], uni_array, obs, cfg["nepisodes"])
        _, path = old.greedy(nodes_set, output_edge_order=edge_order)
        tn_layer = old.TNLayer(cfg["chi"], nlayers, path)
        tn_effective = (cfg["chi"] ** 4 - 1) * nlayers
    elif cfg.get("stoch"):
        import qks_tn_stoch as stoch

        tn_layer = stoch.StochTNLayer(
            chi=cfg["chi"], nlayers=nlayers, mode=cfg["mode"], share_per_layer=cfg["share_per_layer"],
            kraus_rank=int(cfg["kraus_rank"]), coherent_layers={"full": 0, "c0": 1, "c2": 2, "c3": 3}[cfg["stoch"]],
            ff_dim=(int(cfg["ff_dim"]) if cfg["ff_dim"] else None),
            root_ff_dim=(int(cfg["root_ff_dim"]) if cfg["root_ff_dim"] else None),
            init_noise=float(cfg["cptp_init_noise"]), seed=int(cfg["seed"]),
            init_scale=float(cfg.get("stoch_init_scale", 0.3)),
            ff_noise=float(cfg.get("ff_noise") or 0.0),
        )
        tn_effective = None
    elif cfg["kraus_rank"]:
        import qks_tn_cptp as cptp

        tn_layer = cptp.CPTPTNLayer(
            chi=cfg["chi"], nlayers=nlayers, mode=cfg["mode"], share_per_layer=cfg["share_per_layer"],
            kraus_rank=int(cfg["kraus_rank"]), root_rank=(int(cfg["root_rank"]) if cfg["root_rank"] else None),
            root_fold=bool(cfg["root_fold"]),
            ff_dim=(int(cfg["ff_dim"]) if cfg["ff_dim"] else None),
            root_ff_dim=(int(cfg["root_ff_dim"]) if cfg["root_ff_dim"] else None),
            ff_map=cfg["ff_map"],
            ff_width=(int(cfg["ff_width"]) if cfg["ff_width"] else None),
            meas_bases=(int(cfg["meas_bases"]) if cfg.get("meas_bases") else None),
            leaf_dephase=(float(cfg["leaf_dephase"]) if cfg.get("leaf_dephase") else None),
            ff_stop_grad=bool(cfg.get("ff_stop_grad")), ff_null=cfg.get("ff_null"),
            ff_noise=float(cfg.get("ff_noise") or 0.0), ff_dropout=float(cfg.get("ff_dropout") or 0.0),
            ff_mix=float(cfg.get("ff_mix") or 0.0),
            init=cfg["cptp_init"], init_noise=float(cfg["cptp_init_noise"]),
            ff_layers=_parse_layers(cfg["ff_layers"]), seed=int(cfg["seed"]),
        )
        tn_effective = None
    else:
        tn_layer = ff.FeedforwardTNLayer(
            chi=cfg["chi"],
            nlayers=nlayers,
            mode=cfg["mode"],
            share_per_layer=cfg["share_per_layer"],
            shots=cfg["shots"],
            ff_layers=_parse_layers(cfg["ff_layers"]),
            ff_features=cfg["ff_features"],
            ff_null=cfg["ff_null"],
            seed=int(cfg["seed"]),
        )
        tn_effective = None

    n_outcomes = int(cfg["chi"]) ** 2
    if cfg.get("kraus_rank") and cfg.get("root_rank") and not cfg.get("root_fold"):
        n_outcomes = int(cfg["chi"]) * int(cfg["root_rank"])
    if head_kind == "const":
        head = ff.ConstMulFixed(n_outcomes, n_classes)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        # trainable logits from the 16 readout probabilities (needed when classes > 16)
        head = tf.keras.layers.Dense(n_classes, name="linear_head")
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _readout = [head]
    if head_kind == "linear" and cfg.get("head_scale"):  # W17: constant gain on the readout probabilities (no parameters)
        _readout = [tf.keras.layers.Rescaling(float(cfg["head_scale"]), name="head_scale"), head]
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(p,)), qks, tn_layer, *_readout])
    if cfg["freeze_qks"]:
        qks.trainable = False

    lr_schedule = _make_schedule(cfg, "lr", steps_per_epoch)
    wd_schedule = _make_schedule(cfg, "wd", steps_per_epoch)
    model.compile(
        loss=loss,
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule),
        metrics=["accuracy"],
    )
    counts = dict(
        total=int(model.count_params()),
        qks=int(qks.count_params()),
        tn=int(tn_layer.count_params()),
        tn_effective=int(tn_effective if tn_effective is not None else tn_layer.count_params()),
        head=int(head.count_params()),
        trainable=int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
        input_dim=int(p),
        n_classes=int(n_classes),
        sigma=float(sigma),
        head_kind=head_kind,
        chi=int(cfg["chi"]),
        n_outcomes=n_outcomes,
        nlayers=int(nlayers),
        **encoder_info,
    )
    return model, qks, tn_layer, counts


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _eval_sampled(model, test_ds):
    """Evaluate with ``training=True`` forward passes (finite-shot statistics)."""
    import numpy as np
    import tensorflow as tf

    losses, correct, n = 0.0, 0, 0
    for xb, yb in test_ds:
        pr = model(xb, training=True)
        loss = model.loss(yb, pr)
        losses += float(loss) * int(xb.shape[0])
        correct += int(np.sum(np.argmax(pr.numpy(), -1) == yb.numpy()))
        n += int(xb.shape[0])
    return dict(loss=losses / n, acc=correct / n)


def _log(msg):
    print(msg, flush=True)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _newest_checkpoint(ckpt_dir, latest_rolling):
    """Pick the checkpoint with the most completed epochs among the rolling one and cycle-*/."""
    import glob as _glob

    cands = []
    if latest_rolling:
        hp = os.path.join(ckpt_dir, "history.json")
        try:
            with open(hp) as f:
                n = len(json.load(f)["epochs"])
            cands.append((n, latest_rolling, hp))
        except Exception:  # noqa: BLE001  (torn/absent history -> ignore the rolling checkpoint)
            pass
    for d in _glob.glob(os.path.join(ckpt_dir, "cycle-*")):
        hp = os.path.join(d, "history.json")
        try:
            with open(hp) as f:
                n = len(json.load(f)["epochs"])
            cands.append((n, os.path.join(d, "ckpt"), hp))
        except Exception:  # noqa: BLE001
            pass
    if not cands:
        return None, os.path.join(ckpt_dir, "history.json")
    n, path, hp = max(cands)
    return path, hp


def train(cfg: dict, on_epoch=None, log=_log) -> dict:
    """Run one experiment; returns the result dict (also written to ``cfg['out']`` if set)."""
    import numpy as np
    import tensorflow as tf

    if cfg.get("intra_threads"):
        tf.config.threading.set_intra_op_parallelism_threads(int(cfg["intra_threads"]))
    if cfg.get("inter_threads"):
        tf.config.threading.set_inter_op_parallelism_threads(int(cfg["inter_threads"]))
    tf.get_logger().setLevel("ERROR")

    t_start = time.time()
    X_train, y_train, X_test, y_test = load_data(cfg["task"], cfg["cache_dir"], log=log)
    holdout = bool(cfg.get("holdout_5000"))
    if holdout:
        if cfg["task"] not in ("cifar10", "timn100", "svhn"):
            raise ValueError("--holdout-5000 is only supported for CIFAR-10, timn100 and svhn")
        _split = {"timn100": split_timn100_holdout, "svhn": split_svhn_holdout}.get(cfg["task"], split_cifar10_holdout)
        X_train, y_train, X_val, y_val, val_idx = _split(X_train, y_train)
    if cfg.get("limit_train"):
        X_train, y_train = X_train[: int(cfg["limit_train"])], y_train[: int(cfg["limit_train"])]
    bs = int(cfg["batch_size"])
    steps_per_epoch = len(y_train) // bs
    p = int(X_train.shape[1])
    n_classes = N_CLASSES[cfg["task"]]

    model, qks, tn_layer, counts = build_model(cfg, steps_per_epoch, p=p, n_classes=n_classes)
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    log(f"[{cfg['name']}] devices={devices} params={counts} steps/epoch={steps_per_epoch} n_train={len(y_train)} n_test={len(y_test)}")

    tf.keras.utils.set_random_seed(int(cfg["seed"]))
    aug = cfg.get("augment")
    if parse_augment(aug) and cfg["task"] not in AUGMENT_TASKS:
        raise ValueError(f"--augment is only supported for {sorted(AUGMENT_TASKS)}")
    train_ds = make_dataset(X_train, y_train, bs, shuffle=True, seed=int(cfg["seed"]), shuffle_buffer=cfg.get("shuffle_buffer"),
                            augment=aug, image_shape=AUGMENT_TASKS.get(cfg["task"]))  # test and holdout datasets below are never augmented
    log(f"[{cfg['name']}] shuffle buffer = {cfg.get('shuffle_buffer') or len(y_train)} of {len(y_train)} (full permutation)" if not cfg.get("shuffle_buffer") else f"[{cfg['name']}] shuffle buffer = {cfg['shuffle_buffer']} of {len(y_train)} (WINDOWED)")
    test_ds = make_dataset(X_test, y_test, bs, shuffle=False, seed=int(cfg["seed"]))
    val_ds = make_dataset(X_val, y_val, bs, shuffle=False, seed=int(cfg["seed"])) if holdout else test_ds

    history = []
    initial_epoch = 0
    manager = None
    if cfg.get("ckpt_dir"):
        ckpt_dir = cfg["ckpt_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        # model weights + AdamW moments + the optimizer step counter (which drives the
        # CosineDecayRestarts schedules), so training continues exactly where it stopped
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
        hist_path = os.path.join(ckpt_dir, "history.json")
        restore_path = None
        if cfg.get("resume_from"):
            restore_path = cfg["resume_from"]
            hist_path = os.path.join(os.path.dirname(restore_path), "history.json")
        elif cfg.get("resume"):
            restore_path, hist_path = _newest_checkpoint(ckpt_dir, manager.latest_checkpoint)
        if restore_path is not None and os.path.exists(hist_path):
            with open(hist_path) as f:
                saved = json.load(f)
            history = saved["epochs"]
            initial_epoch = len(history)
            ckpt.restore(restore_path).expect_partial()
            log(f"[{cfg['name']}] resumed from {restore_path} at epoch {initial_epoch}")
        elif cfg.get("resume") or cfg.get("resume_from"):
            log(f"[{cfg['name']}] no checkpoint to resume from in {ckpt_dir}; starting fresh")
        with open(os.path.join(ckpt_dir, "cfg.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    if initial_epoch >= int(cfg["epochs"]):
        log(f"[{cfg['name']}] already trained {initial_epoch} >= {cfg['epochs']} epochs; evaluating only")

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t0 = time.time()
            self._t_val = None
            self._first_batch_loss = None

        def on_train_batch_end(self, batch, logs=None):
            if batch == 0:
                self._first_batch_loss = float((logs or {}).get("loss", np.nan))

        def on_test_begin(self, logs=None):
            if getattr(self, "_t0", None) is not None:
                self._t_val = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            now = time.time()
            train_time = (self._t_val or now) - self._t0
            try:
                lr_obj = self.model.optimizer.learning_rate
                lr_val = float(lr_obj(self.model.optimizer.iterations)) if callable(lr_obj) else float(lr_obj)
            except Exception:  # noqa: BLE001
                lr_val = None
            rec = dict(
                epoch=epoch + 1,
                time_s=now - self._t0,
                train_time_s=train_time,
                ms_per_step=1000.0 * train_time / max(steps_per_epoch, 1),
                loss=float(logs.get("loss", np.nan)),
                first_batch_loss=self._first_batch_loss,
                acc=float(logs.get("accuracy", np.nan)),
                val_loss=float(logs["val_loss"]) if "val_loss" in logs else None,
                val_acc=float(logs["val_accuracy"]) if "val_accuracy" in logs else None,
                lr=lr_val,
            )
            history.append(rec)
            if manager is not None:
                manager.save(checkpoint_number=rec["epoch"])  # rolling latest
                state = dict(cfg=cfg, params=counts, epochs=list(history))
                _atomic_json(os.path.join(cfg["ckpt_dir"], "history.json"), state)
                e = rec["epoch"]
                if (e + 1) & e == 0:
                    # e = 2^k - 1: end of a cosine-annealing cycle (1, 3, 7, 15, 31, 63, ...);
                    # keep this state permanently
                    cycle_dir = os.path.join(cfg["ckpt_dir"], f"cycle-{e:03d}")
                    os.makedirs(cycle_dir, exist_ok=True)
                    ckpt.write(os.path.join(cycle_dir, "ckpt"))
                    _atomic_json(os.path.join(cycle_dir, "history.json"), state)
                    rec["cycle_checkpoint"] = cycle_dir
            log(
                f"[{cfg['name']}] epoch {rec['epoch']}/{cfg['epochs']} "
                f"loss {rec['loss']:.4f} acc {rec['acc']:.4f} "
                + (f"val_loss {rec['val_loss']:.4f} val_acc {rec['val_acc']:.4f} " if rec["val_loss"] is not None else "")
                + f"({rec['time_s']:.0f}s, {rec['ms_per_step']:.0f} ms/step)"
            )
            if on_epoch is not None:
                on_epoch(dict(cfg=cfg, params=counts, epochs=list(history)))

    if initial_epoch < int(cfg["epochs"]):
        model.fit(
            train_ds,
            epochs=int(cfg["epochs"]),
            initial_epoch=initial_epoch,
            validation_data=val_ds if cfg["eval_every_epoch"] else None,
            verbose=0,
            callbacks=[EpochLogger()],
        )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, -1).astype(int)
    test_errors = np.nonzero(preds != y_test.astype(int))[0].astype(int).tolist()
    preds = preds.tolist()
    result = dict(
        cfg=cfg,
        git_commit=_git_commit(),
        versions=dict(tensorflow=tf.__version__, numpy=np.__version__, python=sys.version.split()[0]),
        devices=devices,
        params=counts,
        steps_per_epoch=steps_per_epoch,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        epochs=history,
        final_test=dict(loss=float(test_loss), acc=float(test_acc)),
        best_val=(
            max(((h["val_acc"], -h["val_loss"], h["epoch"]) for h in history if h["val_acc"] is not None), default=None)
        ),
        total_wall_s=time.time() - t_start,
        resumed_from_epoch=initial_epoch,
        checkpoint=(manager.latest_checkpoint if manager is not None else None),
        ms_per_step=float(np.median([h["ms_per_step"] for h in history[1:]] or [history[0]["ms_per_step"]])),
        test_predictions=preds,
        test_errors=test_errors,
    )
    if holdout:
        result["n_val"] = int(len(y_val))
        result["validation_split_seed"] = 20261102
        result["validation_indices"] = val_idx.astype(int).tolist()
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        result["final_val"] = dict(loss=float(val_loss), acc=float(val_acc))
    if result["best_val"] is not None:
        va, nvl, ep = result["best_val"]
        result["best_val"] = dict(val_acc=va, val_loss=-nvl, epoch=ep)
    if cfg.get("shots"):
        result["final_test_sampled"] = _eval_sampled(model, test_ds)
    log(
        f"[{cfg['name']}] DONE test_loss {test_loss:.4f} test_acc {test_acc:.4f} "
        f"wall {result['total_wall_s']/60:.1f} min ({result['ms_per_step']:.0f} ms/step)"
    )
    if cfg.get("out"):
        os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
        with open(cfg["out"], "w") as f:
            json.dump(result, f)
    return result


# --------------------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), help="named configuration (overrides mode/share/etc.)")
    ap.add_argument("--task", choices=["binary", "multiclass", "cifar10", "imagenet32", "timn100", "svhn"])
    ap.add_argument("--head", choices=["auto", "const", "linear"])
    ap.add_argument("--head-scale", dest="head_scale", type=float, help="linear head: constant gain on the readout probabilities (W17)")
    ap.add_argument("--mode", choices=["baseline", "A", "B"])
    ap.add_argument("--impl", choices=["new", "old"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--shots", type=int)
    ap.add_argument("--share-per-layer", dest="share_per_layer", action="store_true", default=None)
    ap.add_argument("--no-share-per-layer", dest="share_per_layer", action="store_false")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--nepisodes", type=int, help="QKS episodes = tree leaves x 2; 128 -> 6 layers, 256 -> 7, 512 -> 8")
    ap.add_argument("--encoder", choices=["dense", "patch", "twostage"])
    ap.add_argument("--patch", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--pad", type=int)
    ap.add_argument("--groups", type=int)
    ap.add_argument("--chi", type=int, help="register (bond) dimension, a power of two; 4 = 2 qubits, 8 = 3")
    ap.add_argument("--leaf-order", dest="leaf_order", choices=["morton", "raster"])
    ap.add_argument("--no-share-positions", dest="share_positions", action="store_false", default=None)
    ap.add_argument("--sigma-scale", dest="sigma_scale", type=float)
    ap.add_argument("--ff-layers", dest="ff_layers", help="comma-separated layers receiving feedforward, e.g. 4,5")
    ap.add_argument("--ff-features", dest="ff_features", choices=["raw", "centered"])
    ap.add_argument("--ff-null", dest="ff_null", choices=["none", "constant", "shuffle"])
    ap.add_argument("--kraus-rank", dest="kraus_rank", type=int, help="CPTP vertices of this Kraus rank (>= chi)")
    ap.add_argument("--root-rank", dest="root_rank", type=int, help="CPTP root vertex Kraus rank; the readout widens to chi*root_rank")
    ap.add_argument("--root-fold", dest="root_fold", action="store_true", default=None, help="fold root outcomes back to chi**2")
    ap.add_argument("--ff-dim", dest="ff_dim", type=int, help="CPTP layer: feedforward features per vertex (>= 2R; extra are monomials)")
    ap.add_argument("--root-ff-dim", dest="root_ff_dim", type=int, help="CPTP layer: feedforward features at the root")
    ap.add_argument("--ff-map", dest="ff_map", choices=["monomial", "hidden", "random"], help="CPTP extra feature allocation")
    ap.add_argument("--ff-width", dest="ff_width", type=int, help="CPTP trainable linear compression of 2R child probabilities")
    ap.add_argument("--meas-bases", dest="meas_bases", type=int, help="CPTP layer: K measurement bases of each discarded register (K*R features per child)")
    ap.add_argument("--leaf-dephase", dest="leaf_dephase", type=float, help="CPTP layer: leaf dephasing p, rho -> (1 - p) rho + p diag(rho)")
    ap.add_argument("--ff-stop-grad", dest="ff_stop_grad", action="store_const", const=True, default=None, help="CPTP layer: stop gradients through the feedforward features (W27)")
    ap.add_argument("--cptp-init", dest="cptp_init", choices=["embed", "random"])
    ap.add_argument("--cptp-init-noise", dest="cptp_init_noise", type=float)
    ap.add_argument("--stoch-init-scale", dest="stoch_init_scale", type=float, help="classical node (stoch configs): init scale, default 0.3")
    ap.add_argument("--schedule", choices=["restarts", "cosine1"], help="lr/wd schedule: restarts (default, CosineDecayRestarts) | cosine1 (single cosine decay to 0 over all epochs)")
    ap.add_argument("--ff-noise", dest="ff_noise", type=float, help="CPTP layer: training-only Gaussian noise std on the feedforward features (W33)")
    ap.add_argument("--ff-dropout", dest="ff_dropout", type=float, help="CPTP layer: training-only dropout rate on the feedforward features (W33)")
    ap.add_argument("--ff-mix", dest="ff_mix", type=float, help="CPTP layer: training-only feature-wise mixing rate, negative control (W33)")
    ap.add_argument("--augment", help="training-only augmentation, '+'-joined from flip, crop, cutout (CIFAR-10 only; default off)")
    ap.add_argument("--freeze-qks", dest="freeze_qks", action="store_true", default=None)
    ap.add_argument("--fast-qks", dest="fast_qks", action="store_true", default=None)
    ap.add_argument("--no-fast-qks", dest="fast_qks", action="store_false")
    ap.add_argument("--limit-train", dest="limit_train", type=int)
    ap.add_argument("--holdout-5000", dest="holdout_5000", action="store_true", default=None,
                    help="CIFAR-10: use W4's stratified 45k train / 5k validation split (seed 20261102)")
    ap.add_argument("--no-eval-every-epoch", dest="eval_every_epoch", action="store_false", default=None)
    ap.add_argument("--intra-threads", dest="intra_threads", type=int)
    ap.add_argument("--inter-threads", dest="inter_threads", type=int)
    ap.add_argument("--cache-dir", dest="cache_dir")
    ap.add_argument("--out")
    ap.add_argument("--name")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", help="checkpoint directory (default runs/ckpt/<name>)")
    ap.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true", help="disable checkpointing")
    ap.add_argument("--resume", action="store_true", default=None, help="continue from the latest checkpoint in --ckpt-dir")
    ap.add_argument("--resume-from", dest="resume_from", help="checkpoint prefix to restore, e.g. runs/ckpt/<name>/cycle-015/ckpt")
    return ap


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    config_name = args.pop("config")
    overrides = dict(CONFIGS.get(config_name or "", {}))
    overrides.update({k: v for k, v in args.items() if v is not None})
    if overrides.get("ff_null") == "none":
        overrides["ff_null"] = None
    no_ckpt = overrides.pop("no_checkpoint", False)
    cfg = make_cfg(**overrides)
    cfg["config"] = config_name
    if cfg["out"] is None:
        cfg["out"] = os.path.join(HERE, "runs", cfg["name"] + ".json")
    if no_ckpt:
        cfg["ckpt_dir"] = None
    elif cfg["ckpt_dir"] is None:
        cfg["ckpt_dir"] = os.path.join(HERE, "runs", "ckpt", cfg["name"])
    train(cfg)


if __name__ == "__main__":
    main()
