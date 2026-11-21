"""Summarise and plot feedforward-vs-baseline runs.

Reads result JSONs written by ``run_ff_experiment.py`` / ``modal_ff_sweep.py`` and produces

* a markdown table per configuration (mean +/- s.e.m. over seeds of final test acc/CE,
  best validation acc, TN parameter count, ms/step),
* paired per-seed comparisons against the pre-registered controls (same seed = same
  QKS init, TN init and data order) with a McNemar exact test on the test-set errors,
* training / validation loss and accuracy curves (one figure per facet: trainable QKS,
  frozen QKS), thin lines per seed and a thick mean per configuration.

Usage::

    .venv/bin/python qks-tn-library/analyze_ff_runs.py --sweep ff-v1          # results/ff-v1/*.json
    .venv/bin/python qks-tn-library/analyze_ff_runs.py --dir qks-tn-library/runs --tag binary-local
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_ff_experiment as rfe  # noqa: E402

# Controls: (candidate, control, what the comparison isolates)
COMPARISONS = [
    ("C1", "C0", "per-vertex unitaries vs shared (more parameters, no feedforward)"),
    ("C2", "C0", "multishot feedforward (B) vs baseline"),
    ("C3", "C0", "information-free B (constant p) vs baseline: reparametrisation only"),
    ("C2", "C3", "B vs its information-free twin: value of the fed-forward information"),
    ("C4", "C1", "per-vertex B vs per-vertex baseline"),
    ("C5", "C0", "single-shot feedforward (A) vs baseline"),
    ("C5", "C1", "A vs parameter-matched per-vertex baseline"),
    ("C6", "C1", "per-vertex A vs per-vertex baseline"),
    ("C6", "C4", "per-vertex A vs per-vertex B"),
    ("L0", "C0", "patch-local shared filters vs dense QKS (baseline TN)"),
    ("L1", "C1", "patch-local shared filters vs dense QKS (per-vertex baseline TN)"),
    ("L4", "C4", "LeNet-style (patch + Morton + per-vertex B) vs dense per-vertex B"),
    ("L4", "R4", "Morton (quadtree) vs raster leaf order: 2-D locality in the tree"),
    ("L4", "U4", "position-shared filters vs locally connected (weight sharing)"),
    ("U4", "C4", "local unshared filters vs dense: locality alone"),
    ("L4", "L1", "feedforward B vs baseline with the patch encoder"),
    ("T4", "L4", "two-stage (re-encoded statistics) vs single-tree LeNet-style"),
    ("T4", "T1", "feedforward B vs baseline in the two-stage model"),
    ("L2", "L0", "shared B vs shared baseline with the patch encoder"),
    ("R4", "R1", "feedforward B vs baseline, raster order"),
    ("L4lin", "L4", "trainable linear head vs the paper's fixed readout"),
    ("F2", "F0", "B vs baseline with frozen QKS (TN-only learning)"),
    ("F5", "F0", "A vs baseline with frozen QKS (TN-only learning)"),
    ("C0", "C0old", "new sweep vs original TNLayer implementation (regression)"),
    # CPTP vertices (Kraus rank R); K4 is the unitary TTN written as a CPTP layer
    ("K4", "C0", "CPTP layer at R = chi vs the unitary layer (regression, must be 0)"),
    ("K8", "K4", "Kraus rank 8 vs 4 (no feedforward, shared)"),
    ("K16", "K4", "Kraus rank 16 vs 4 (no feedforward, shared)"),
    ("K16", "C1", "Kraus rank 16 shared vs per-node unitaries (more parameters, same information)"),
    ("K32", "K16", "Kraus rank 32 vs 16 (Choi argument: no new channels)"),
    ("K16p", "K16", "per-node vs shared gates at rank 16"),
    ("K8B", "C2", "rank 8 + feedforward vs unitary + feedforward (shared)"),
    ("K8B", "C4", "rank 8 shared + feedforward vs per-node unitary + feedforward (similar parameters)"),
    ("K16B", "C2", "rank 16 + feedforward vs unitary + feedforward (shared)"),
    ("K16B", "K8B", "rank 16 vs 8 with feedforward"),
    ("K8pB", "C4", "per-node rank 8 + feedforward vs per-node unitary + feedforward"),
    ("FK8", "F0", "rank 8 vs unitary, frozen QKS, no feedforward"),
    ("FK16", "F0", "rank 16 vs unitary, frozen QKS, no feedforward"),
    ("FK8B", "F2", "rank 8 + feedforward vs unitary + feedforward, frozen QKS"),
    # parameter-matched frozen controls
    ("FK8B", "F4", "shared rank 8 + feedforward vs per-node unitary + feedforward, frozen QKS (similar parameters)"),
    ("FK8pB", "F4", "per-node rank 8 + feedforward vs per-node unitary + feedforward, frozen QKS"),
    ("F4", "F2", "per-node vs shared unitary + feedforward, frozen QKS (parameters only)"),
    # Haar-random initialization of the dilation
    ("RK8", "K8", "Haar-random vs embedded initialization, rank 8"),
    ("RK16", "K16", "Haar-random vs embedded initialization, rank 16"),
    ("RK8B", "K8B", "Haar-random vs embedded initialization, rank 8 + feedforward"),
    ("RK16B", "K16B", "Haar-random vs embedded initialization, rank 16 + feedforward"),
    ("RK8B", "C2", "Haar-random rank 8 + feedforward vs unitary + feedforward"),
    # wide root readout (chi * W outcomes)
    ("K8BW8", "K8B", "root rank 8 (32 outcomes) vs unitary root (16), rank 8 + feedforward"),
    ("K8BW16", "K8B", "root rank 16 (64 outcomes) vs unitary root (16), rank 8 + feedforward"),
    ("K8BW16", "K8BW8", "root rank 16 vs 8"),
    ("K8W16", "K8", "root rank 16 vs unitary root, rank 8, no feedforward"),
    # patch encoder (CIFAR-10): controls L0 / L2 / L4
    ("LK8", "L0", "CPTP rank 8 vs unitary, patch encoder, no feedforward"),
    ("LK16", "L0", "CPTP rank 16 vs unitary, patch encoder, no feedforward"),
    ("LK8B", "L2", "CPTP rank 8 + feedforward vs unitary + feedforward, shared, patch encoder"),
    ("LK16B", "L2", "CPTP rank 16 + feedforward vs unitary + feedforward, shared, patch encoder"),
    ("LK8B", "L4", "shared rank 8 + feedforward vs per-node unitary + feedforward (similar parameters), patch encoder"),
    ("LK8pB", "L4", "per-node rank 8 + feedforward vs per-node unitary + feedforward, patch encoder"),
    ("LK16B", "LK8B", "rank 16 vs 8 with feedforward, patch encoder"),
    ("LK32", "LK16", "rank 32 vs 16, patch encoder, no feedforward"),
    ("LK8BW16", "LK8B", "root rank 16 vs unitary root, rank 8 + feedforward, patch encoder"),
    ("LK16B", "L4", "shared rank 16 + feedforward vs per-node unitary + feedforward (parameter-matched only at chi = 8: 2.31M vs 2.29M)"),
    ("LK16B", "LK8pB", "shared rank 16 vs per-node rank 8, both with feedforward (553k vs 559k params, chi = 4)"),
    ("LK16", "L0", "CPTP rank 16 vs unitary, no feedforward, patch encoder"),
    # readout width vs root parameters (CIFAR-10, chi = 4): W8 = root rank 8, W16 = root rank 16, F = outcomes folded to 16
    ("LK8BW8", "LK8B", "root rank 8 (32 outcomes) vs unitary root (16), rank 8 + feedforward"),
    ("LK8BW8F", "LK8B", "root rank 8, outcomes folded to 16: root parameters without readout width"),
    ("LK8BW16F", "LK8B", "root rank 16, outcomes folded to 16: root parameters without readout width"),
    ("LK8BW16", "LK8BW16F", "readout width at fixed root parameters (64 vs 16 outcomes, root rank 16)"),
    ("LK8BW8", "LK8BW8F", "readout width at fixed root parameters (32 vs 16 outcomes, root rank 8)"),
    ("LK8BW16", "LK8BW8", "root rank 16 vs 8 (64 vs 32 outcomes)"),
    ("LK16B", "LU8Bf112", "rank 16 vs shared unitary with 112 feedforward features at matched parameters, chi = 8"),
    ("LU8Bf112", "L4", "shared unitary + 112 features vs per-node unitary, both about 2.3M parameters, chi = 8"),
    ("LU8Bf112", "L2", "shared unitary with 112 vs 16 feedforward features, chi = 8"),
    ("LK8BW16F", "LK8BRf272", "root environment (rank 16, folded) vs unitary root with 272 features at equal root parameters, chi = 4"),
    ("LK8BRf272", "LK8B", "unitary root with 272 features vs 16 features, rank-8 vertices, chi = 4"),
    # chi = 8: rank 32 with feedforward
    ("LK32B", "LK16B", "rank 32 vs 16 with feedforward, chi = 8"),
    ("LK32B", "L4", "rank 32 + feedforward (shared) vs per-node unitary + feedforward, chi = 8 (parameters not matched)"),
    # A1: alternative matched-parameter unitary allocations at chi = 8
    ("LK16B", "LU8pBf16R22", "rank 16 vs per-node unitary with 22 root features"),
    ("LK16B", "LU8Bf187L012", "rank 16 vs unitary with feedforward at layers 0-2"),
    ("LK16B", "LU8Bhidden111", "rank 16 vs unitary with trainable hidden feedforward features"),
    ("LK16B", "LU8Brandom112", "rank 16 vs unitary with fixed random-projection features"),
    ("LU8pBf16R22", "LU8Bf112", "per-node plus root features vs shared monomials"),
    ("LU8Bf187L012", "LU8Bf112", "bottom-layer features vs shared monomials"),
    ("LU8Bhidden111", "LU8Bf112", "trainable hidden features vs monomials"),
    ("LU8Brandom112", "LU8Bf112", "random-projection features vs monomials"),
]

for _base in ("LK16B", "L4", "LU8Brandom112"):
    for _lf in ("05", "1", "2"):
        for _wf in ("0", "1", "3"):
            if (_lf, _wf) != ("1", "1"):
                COMPARISONS.append((f"A2{_base}l{_lf}w{_wf}", _base,
                                    f"A2 learning-rate/weight-decay setting vs default for {_base}"))
COMPARISONS.append(("A2LK16Bl1w0", "A2L4l1w0", "both at lr 0.001 and wd 0; A2 partial grid"))

# Fixed categorical slot per *model type* (colour follows the entity, never the rank).
# Palette: dataviz reference categorical order (light surface).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SLOT = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 7, "C0old": 6, "C2c": 7}
SLOT.update({"F0": SLOT["C0"], "F2": SLOT["C2"], "F5": SLOT["C5"]})
SLOT.update({"C2s16": 2, "C2s128": 2, "C2s1024": 2, "C2root": 2, "C2top2": 2})
SLOT.update({"T1": 6, "T4": 7, "L0": 0, "L1": 1, "L2": 2, "L4": 4, "R1": 3, "R4": 5, "U1": 6, "U4": 7, "L4lin": 3, "C4lin": 3})
LABEL = {
    # Plain-language scheme: [encoder] - [feedforward] - [gate sharing].
    # "shared gates" = one unitary reused by every node of a tree layer (the paper's setting);
    # "per-node gates" = each tree node has its own unitary.
    # "feedforward (multishot)" = measured outcome statistics of the discarded register set the
    # next gate's parameters (the design note's multishot Case B);
    # "feedforward (single-shot)" = the gate is conditioned on the measured outcome itself (Case A).
    "C0": "no feedforward, shared gates",
    "C1": "no feedforward, per-node gates",
    "C2": "feedforward (multishot), shared gates",
    "C3": "feedforward control: constant input",
    "C4": "feedforward (multishot), per-node gates",
    "C5": "feedforward (single-shot), shared gates",
    "C6": "feedforward (single-shot), per-node gates",
    "C0old": "original implementation",
    "F0": "no feedforward, shared gates",
    "F2": "feedforward (multishot), shared gates",
    "F5": "feedforward (single-shot), shared gates",
    "C2c": "feedforward, centred statistics",
    "C2s16": "feedforward from 16 measurements",
    "C2s128": "feedforward from 128 measurements",
    "C2s1024": "feedforward from 1024 measurements",
    "C2root": "feedforward at the top node only",
    "C2top2": "feedforward in the top two layers",
    "L0": "local filters, no feedforward, shared gates",
    "L1": "local filters, no feedforward, per-node gates",
    "L2": "local filters, feedforward, shared gates",
    "L4": "local filters, feedforward, per-node gates",
    "R1": "local filters (row order), no feedforward",
    "R4": "local filters (row order), feedforward",
    "U1": "per-position filters, no feedforward",
    "U4": "per-position filters, feedforward",
    "T1": "two-stage, no feedforward",
    "T4": "two-stage, feedforward",
    "L4lin": "local filters, feedforward, trainable output layer",
    "C4lin": "feedforward, trainable output layer",
    "K4": "unitary (Kraus rank 4), shared gates",
    "K8": "Kraus rank 8, shared gates",
    "K16": "Kraus rank 16, shared gates",
    "K32": "Kraus rank 32, shared gates",
    "K16p": "Kraus rank 16, per-node gates",
    "K8B": "rank 8 + feedforward, shared gates",
    "K16B": "rank 16 + feedforward, shared gates",
    "K8pB": "rank 8 + feedforward, per-node gates",
    "FK8": "Kraus rank 8, shared gates",
    "FK16": "Kraus rank 16, shared gates",
    "FK8B": "rank 8 + feedforward, shared gates",
    "F4": "feedforward (multishot), per-node gates",
    "FK8pB": "rank 8 + feedforward, per-node gates",
    "RK8": "Haar init, rank 8",
    "RK16": "Haar init, rank 16",
    "RK8B": "Haar init, rank 8 + feedforward",
    "RK16B": "Haar init, rank 16 + feedforward",
    "K8BW8": "rank 8 + feedforward, root rank 8",
    "K8BW16": "rank 8 + feedforward, root rank 16",
    "K8W8": "rank 8, root rank 8",
    "K8W16": "rank 8, root rank 16",
    "LK8": "rank 8, patch encoder",
    "LK16": "rank 16, patch encoder",
    "LK32": "rank 32, patch encoder",
    "LK8B": "rank 8 + feedforward, shared gates",
    "LK16B": "rank 16 + feedforward, shared gates",
    "LK8pB": "rank 8 + feedforward, per-node gates",
    "LK8BW16": "rank 8 + feedforward, root rank 16",
    "LK8BW8F": "rank 8 + feedforward, root rank 8 (folded)",
    "LK8BW16F": "rank 8 + feedforward, root rank 16 (folded)",
    "LU8Bf112": "unitary, shared gates, 112 feedforward features",
    "LU8pBf16R22": "unitary, per-node gates, 22 root features",
    "LU8Bf187L012": "unitary, shared gates, 187 features at layers 0-2",
    "LU8Bhidden111": "unitary, shared gates, 111 hidden features",
    "LU8Brandom112": "unitary, shared gates, 112 random-projection features",
    "LK8BRf272": "rank 8, unitary root with 272 root feedforward features",
    "LK32B": "rank 32 + feedforward, shared gates",
}
# every line in the per-node facet is a per-node-gate model, so the qualifier is dropped there
FACET_LABELS = {
    "per-node gates": {
        "C1": "no feedforward",
        "C6": "feedforward (single-shot)",
        "C4": "feedforward (multishot)",
    }
}
_MATCH_KEYS = ["mode", "share_per_layer", "ff_null", "shots", "ff_layers", "ff_features", "freeze_qks", "impl",
               "encoder", "leaf_order", "share_positions", "head", "nepisodes"]


def infer_config(cfg: dict) -> str:
    name = cfg.get("config")
    if name:
        # a run launched from a named config but with an overridden head is a distinct arm
        ref = rfe.make_cfg(**rfe.CONFIGS[name]) if name in rfe.CONFIGS else {}
        if str(cfg.get("head")) != str(ref.get("head", "auto")) and str(cfg.get("head")) == "linear":
            return name + "lin"
        return name
    for name, ov in rfe.CONFIGS.items():
        ref = rfe.make_cfg(**ov)
        if all(str(cfg.get(k)) == str(ref.get(k)) for k in _MATCH_KEYS):
            return name
    return cfg.get("name", "?")


def load_results(paths):
    runs = []
    for p in paths:
        if os.path.basename(p) in ("summary.json",) or os.path.basename(p).startswith("smoke"):
            continue
        with open(p) as f:
            r = json.load(f)
        if "epochs" not in r or "final_test" not in r:
            continue
        r["config"] = infer_config(r["cfg"])
        r["seed"] = int(r["cfg"]["seed"])
        runs.append(r)
    return runs


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def fmt(m, s, digits=4):
    if np.isnan(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def mcnemar_p(errors_a, errors_b):
    """Exact two-sided McNemar test on discordant test-set errors."""
    a, b = set(errors_a), set(errors_b)
    n_a_only, n_b_only = len(a - b), len(b - a)
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0, n_a_only, n_b_only
    try:
        from scipy.stats import binomtest

        p = binomtest(min(n_a_only, n_b_only), n, 0.5, alternative="two-sided").pvalue
    except Exception:  # noqa: BLE001
        from scipy.stats import binom

        k = min(n_a_only, n_b_only)
        p = min(1.0, 2 * binom.cdf(k, n, 0.5))
    return float(p), n_a_only, n_b_only


def summary_table(groups):
    lines = [
        "| config | model | seeds | TN params | trainable | final test acc | final test CE | best val acc (epoch) | ms/step | min/run |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(groups, key=lambda k: (k not in SLOT, SLOT.get(k, 99), k)):
        rs = groups[c]
        acc = mean_sem([r["final_test"]["acc"] for r in rs])
        ce = mean_sem([r["final_test"]["loss"] for r in rs])
        bv = mean_sem([r["best_val"]["val_acc"] for r in rs if r.get("best_val")])
        ep = np.mean([r["best_val"]["epoch"] for r in rs if r.get("best_val")]) if any(r.get("best_val") for r in rs) else float("nan")
        ms = np.median([r["ms_per_step"] for r in rs])
        wall = np.mean([r["total_wall_s"] / 60 for r in rs])
        lines.append(
            f"| {c} | {LABEL.get(c, c)} | {len(rs)} | {rs[0]['params']['tn']:,} | {rs[0]['params']['trainable']:,} | "
            f"{fmt(*acc)} | {fmt(*ce)} | {fmt(*bv)} ({ep:.1f}) | {ms:.0f} | {wall:.1f} |"
        )
    return "\n".join(lines)


def comparison_table(groups):
    lines = [
        "| candidate | control | isolates | seeds | Δ test acc (pp) | Δ test CE | sign | McNemar p per seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cand, ctrl, what in COMPARISONS:
        if cand not in groups or ctrl not in groups:
            continue
        by_seed_c = {r["seed"]: r for r in groups[cand]}
        by_seed_k = {r["seed"]: r for r in groups[ctrl]}
        seeds = sorted(set(by_seed_c) & set(by_seed_k))
        if not seeds:
            continue
        d_acc = [100 * (by_seed_c[s]["final_test"]["acc"] - by_seed_k[s]["final_test"]["acc"]) for s in seeds]
        d_ce = [by_seed_c[s]["final_test"]["loss"] - by_seed_k[s]["final_test"]["loss"] for s in seeds]
        ps = []
        for s in seeds:
            p, na, nb = mcnemar_p(by_seed_c[s].get("test_errors", []), by_seed_k[s].get("test_errors", []))
            ps.append(f"s{s}: p={p:.3g} ({na}/{nb})")
        sign = "+" * sum(d > 0 for d in d_acc) + "-" * sum(d < 0 for d in d_acc) + "0" * sum(d == 0 for d in d_acc)
        ma, sa = mean_sem(d_acc)
        mc, sc = mean_sem(d_ce)
        lines.append(f"| {cand} | {ctrl} | {what} | {len(seeds)} | {fmt(ma, sa, 2)} | {fmt(mc, sc, 4)} | {sign} | {'; '.join(ps)} |")
    lines.append("")
    lines.append("Δ = candidate − control, paired by seed. McNemar counts are (errors only in candidate / errors only in control).")
    return "\n".join(lines)


def plot_curves(groups, out_prefix, facet_name, configs):
    """Mean curve per configuration with a 95% confidence band across seeds (t-interval)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from scipy import stats as _st

    configs = [c for c in configs if c in groups]
    if not configs:
        return None
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    # magma sampled over the configs in facet order (clipped away from the near-white end)
    cmap = plt.get_cmap("magma")
    facet_colors = {c: cmap(x) for c, x in zip(configs, np.linspace(0.12, 0.75, max(len(configs), 2)))}
    # facet-specific label overrides (drop qualifiers shared by every line in the facet)
    facet_labels = FACET_LABELS.get(facet_name, {})
    label = lambda c: facet_labels.get(c, LABEL.get(c, c))  # noqa: E731
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=surface, sharex=True)
    panels = [
        (axes[0, 0], "loss", "training loss (CE)"),
        (axes[0, 1], "val_loss", "validation loss (CE, test set)"),
        (axes[1, 0], "acc", "training accuracy"),
        (axes[1, 1], "val_acc", "validation accuracy (test set)"),
    ]
    handles = {}
    for ax, key, title in panels:
        ax.set_facecolor(surface)
        mean_lo, mean_hi = np.inf, -np.inf
        for c in configs:
            color = facet_colors[c]
            curves = []
            for r in groups[c]:
                ep = [h["epoch"] for h in r["epochs"] if h.get(key) is not None]
                v = [h[key] for h in r["epochs"] if h.get(key) is not None]
                if v:
                    curves.append((ep, v))
            if curves:
                n = min(len(v) for _, v in curves)
                arr = np.asarray([v[:n] for _, v in curves], dtype=float)
                epochs = curves[0][0][:n]
                mean = arr.mean(axis=0)
                if arr.shape[0] > 1:
                    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                    tcrit = float(_st.t.ppf(0.975, arr.shape[0] - 1))
                    ax.fill_between(epochs, mean - tcrit * sem, mean + tcrit * sem, color=color, alpha=0.18, lw=0)
                (h,) = ax.plot(epochs, mean, color=color, lw=2.0, label=label(c))
                handles[c] = h
                mean_lo, mean_hi = min(mean_lo, float(mean.min())), max(mean_hi, float(mean.max()))
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8.5)
        if "acc" in key:
            lo, hi = ax.get_ylim()
            hi = min(hi, 1.0)
            ax.set_ylim(lo, hi)  # accuracy cannot exceed 1; don't let CI bands imply it
            # ticks every 0.01 (0.95, 0.97, 0.99, ... visible); coarser when the span is wide
            span = hi - lo
            step = 0.01 if span <= 0.2 else (0.02 if span <= 0.4 else 0.05)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        if "loss" in key:
            ax.set_yscale("log")
            if np.isfinite(mean_lo):
                # limits from the mean curves (a single deep CI dip must not stretch the axis),
                # ticks at 1-2-5 per decade with plain decimal labels
                ax.set_ylim(0.8 * mean_lo, 1.15 * mean_hi)
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) / 10, numticks=12))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    for ax in axes[1]:
        ax.set_xlabel("epoch", color=ink2, fontsize=9)
    tasks = sorted({r["cfg"]["task"] for c in configs for r in groups[c]})
    task_name = {"binary": "(3 vs 5)-MNIST", "multiclass": "MNIST", "cifar10": "CIFAR-10", "imagenet32": "ImageNet-32"}
    dataset = " / ".join(task_name.get(t, t) for t in tasks)
    depths = sorted({r["params"].get("nlayers") for c in configs for r in groups[c]} - {None})
    depth_note = f", {depths[0]}-layer tree" if len(depths) == 1 else ""
    fig.suptitle(f"QKS+TTN on {dataset} — {facet_name}{depth_note}", x=0.01, ha="left", fontsize=12, color=ink)
    fig.legend(
        [handles[c] for c in configs if c in handles],
        [label(c) for c in configs if c in handles],
        loc="lower center",
        ncol=min(3, len(handles)),
        frameon=False,
        fontsize=9,
        labelcolor=ink2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = f"{out_prefix}_{facet_name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", help="name under qks-tn-library/results/")
    ap.add_argument("--dir", help="directory of result JSONs (alternative to --sweep)")
    ap.add_argument("--tag", default=None, help="output file prefix (default: sweep name)")
    args = ap.parse_args(argv)
    if not args.sweep and not args.dir:
        ap.error("give --sweep or --dir")
    d = args.dir or os.path.join(HERE, "results", args.sweep)
    tag = args.tag or args.sweep or os.path.basename(os.path.normpath(d))
    runs = load_results(sorted(glob.glob(os.path.join(d, "*.json"))))
    if not runs:
        print(f"no results in {d}")
        return
    groups = defaultdict(list)
    for r in runs:
        groups[r["config"]].append(r)
    tasks = sorted({r["cfg"]["task"] for r in runs})
    md = [f"# {tag}: {len(runs)} runs, task(s) {', '.join(tasks)}", "", "## Per configuration", "", summary_table(groups), "", "## Paired comparisons", "", comparison_table(groups), ""]
    prefix = os.path.join(d, f"{tag}")
    figs = []
    for facet, confs in [
        ("per-node gates", ["C1", "C6", "C4"]),
        ("trainable QKS", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C0old"]),
        ("frozen QKS", ["F0", "F2", "F5"]),
        ("patch encoder", ["L0", "L1", "L2", "L4", "R1", "R4", "U1", "U4", "L4lin"]),
        ("two-stage", ["L1", "L4", "T1", "T4"]),
        ("ablations", ["C2", "C2c", "C2s16", "C2s128", "C2s1024", "C2root", "C2top2"]),
        ("CPTP rank, no feedforward", ["K4", "K8", "K16", "K32", "K16p"]),
        ("CPTP rank, feedforward", ["C2", "K8B", "K16B", "K8pB", "C4"]),
        ("CPTP rank, frozen QKS", ["F0", "FK8", "FK16", "F2", "F4", "FK8B", "FK8pB"]),
        ("CPTP random init", ["K8", "RK8", "K16", "RK16", "K8B", "RK8B", "K16B", "RK16B"]),
        ("CPTP wide root", ["K8", "K8W16", "K8B", "K8BW8", "K8BW16"]),
        ("CPTP patch encoder, no feedforward", ["L0", "LK8", "LK16"]),
        ("CPTP patch encoder, feedforward", ["L2", "L4", "LK8B", "LK16B", "LK32B", "LK8pB", "LK8BW16"]),
        ("CPTP root width vs parameters", ["LK8B", "LK8BW8F", "LK8BW8", "LK8BW16F", "LK8BW16", "LK8BRf272"]),
        ("matched-parameter unitary controls", ["L2", "L4", "LU8Bf112", "LK16B"]),
    ]:
        present = [c for c in confs if c in groups]
        if facet in ("ablations", "patch encoder", "two-stage") or facet.startswith("CPTP"):
            if len(present) < 2:
                continue
        if False:
            continue
        if facet == "per-node gates" and len(present) < 2:
            continue
        p = plot_curves(groups, prefix + "_curves", facet, confs)
        if p:
            figs.append(p)
    md += ["## Figures", ""] + [f"- `{os.path.relpath(p, HERE)}`" for p in figs]
    text = "\n".join(md)
    with open(prefix + "_summary.md", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {prefix}_summary.md")


if __name__ == "__main__":
    main()
