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
]

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
    # "feedforward (statistics)" = measured outcome statistics of the discarded register set the
    # next gate's parameters (the design note's multishot Case B);
    # "feedforward (single-shot)" = the gate is conditioned on the measured outcome itself (Case A).
    "C0": "no feedforward, shared gates",
    "C1": "no feedforward, per-node gates",
    "C2": "feedforward (statistics), shared gates",
    "C3": "feedforward control: constant input",
    "C4": "feedforward (statistics), per-node gates",
    "C5": "feedforward (single-shot), shared gates",
    "C6": "feedforward (single-shot), per-node gates",
    "C0old": "original implementation",
    "F0": "no feedforward, shared gates",
    "F2": "feedforward (statistics), shared gates",
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
    from scipy import stats as _st

    configs = [c for c in configs if c in groups]
    if not configs:
        return None
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    # magma sampled over the configs in facet order (clipped away from the near-white end)
    cmap = plt.get_cmap("magma")
    facet_colors = {c: cmap(x) for c, x in zip(configs, np.linspace(0.12, 0.75, max(len(configs), 2)))}
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
                (h,) = ax.plot(epochs, mean, color=color, lw=2.0, label=LABEL.get(c, c))
                handles[c] = h
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8.5)
        if "loss" in key:
            ax.set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("epoch", color=ink2, fontsize=9)
    tasks = sorted({r["cfg"]["task"] for c in configs for r in groups[c]})
    task_name = {"binary": "(3 vs 5)-MNIST", "multiclass": "MNIST", "cifar10": "CIFAR-10", "imagenet32": "ImageNet-32"}
    dataset = " / ".join(task_name.get(t, t) for t in tasks)
    fig.suptitle(f"QKS+TTN on {dataset} — {facet_name}", x=0.01, ha="left", fontsize=12, color=ink)
    fig.legend(
        [handles[c] for c in configs if c in handles],
        [LABEL.get(c, c) for c in configs if c in handles],
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
    ]:
        present = [c for c in confs if c in groups]
        if facet in ("ablations", "patch encoder", "two-stage") and len(present) < 1:
            continue
        if facet == "per-node gates" and len(present) < 3:
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
