"""One figure summarising the whole study: what each architectural lever is worth.

Reads every sweep under ``results/`` and draws, per dataset, a horizontal bar chart of the
configurations that matter, coloured by bond dimension (the strongest lever) with the
published model marked. Usage::

    .venv/bin/python qks-tn-library/make_summary_figure.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (dataset, label, matcher) -- matcher picks runs by their recorded cfg/params
ROWS = [
    ("cifar10", "paper's model (dense, chi=4)", dict(encoder="dense", mode="baseline", share=True, chi=4, groups=1)),
    ("cifar10", "+ per-vertex unitaries", dict(encoder="dense", mode="baseline", share=False, chi=4, groups=1)),
    ("cifar10", "+ feedforward B", dict(encoder="dense", mode="B", share=False, chi=4, groups=1)),
    ("cifar10", "+ patch-local encoder", dict(encoder="patch", mode="baseline", share=False, chi=4, groups=1)),
    ("cifar10", "+ patch + feedforward B", dict(encoder="patch", mode="B", share=False, chi=4, groups=1)),
    ("cifar10", "+ G=2 feature maps", dict(encoder="patch", mode="B", share=False, chi=4, groups=2)),
    ("cifar10", "+ G=4 feature maps", dict(encoder="patch", mode="B", share=False, chi=4, groups=4)),
    ("cifar10", "two-stage re-encoding", dict(encoder="twostage", mode="B", share=False, chi=4, groups=1)),
    ("cifar10", "chi=8, patch, no feedforward", dict(encoder="patch", mode="baseline", share=False, chi=8, groups=1)),
    ("cifar10", "chi=8, patch + feedforward B", dict(encoder="patch", mode="B", share=False, chi=8, groups=1)),
    ("cifar10", "chi=16, patch, no feedforward", dict(encoder="patch", mode="baseline", share=True, chi=16, groups=1)),
    ("cifar10", "chi=16, patch + feedforward B", dict(encoder="patch", mode="B", share=True, chi=16, groups=1)),
    ("multiclass", "paper's model (dense, chi=4)", dict(encoder="dense", mode="baseline", share=True, chi=4, groups=1)),
    ("multiclass", "+ feedforward B", dict(encoder="dense", mode="B", share=False, chi=4, groups=1)),
    ("multiclass", "+ patch-local encoder", dict(encoder="patch", mode="baseline", share=False, chi=4, groups=1)),
    ("multiclass", "+ patch + feedforward B", dict(encoder="patch", mode="B", share=False, chi=4, groups=1)),
    ("multiclass", "chi=8, dense + feedforward B", dict(encoder="dense", mode="B", share=False, chi=8, groups=1)),
    ("multiclass", "chi=8, patch + feedforward B", dict(encoder="patch", mode="B", share=False, chi=8, groups=1)),
]

PALETTE = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_all():
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, "results", "*", "*.json")):
        base = os.path.basename(f)
        if base == "summary.json" or base.startswith("smoke"):
            continue
        sweep = os.path.basename(os.path.dirname(f))
        if sweep.startswith(("bench", "gpubench", "chi8bench", "ckpt-smoke", "smoke")):
            continue
        try:
            r = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "final_test" not in r or "params" not in r:
            continue
        c, p = r["cfg"], r["params"]
        if c.get("epochs") != 15 or c.get("limit_train"):
            continue  # compare like with like: the 15-epoch protocol
        if c.get("head") not in (None, "auto"):
            continue  # fixed readout only
        if c.get("freeze_qks"):
            continue  # trainable features only (the frozen-QKS arm is a separate experiment)
        chi = int(p.get("chi", 4))
        # older runs predate the recorded leaf count, and dense runs recorded it as nepisodes//2
        # (wrong for chi != 4); recompute it from the configuration instead
        if p.get("encoder", "dense") == "dense":
            leaves = int(c["nepisodes"] // np.log2(chi))
        else:
            leaves = int(p.get("leaves", 64))
        if leaves != 64 * int(c.get("groups", 1)):
            continue  # one tree size (64 leaves per feature map, 6 layers) so rows are comparable
        key = (
            c["task"],
            p.get("encoder", "dense"),
            c["mode"],
            bool(c["share_per_layer"]),
            chi,
            int(c.get("groups", 1)),
            c.get("leaf_order", "morton"),
            bool(c.get("share_positions", True)),
            c.get("ff_null"),
            c.get("shots"),
            (int(p.get("patch", 0)), int(p.get("stride", 0))),  # keep receptive-field geometries apart
        )
        runs[key].append(r["final_test"]["acc"])
    return runs


def pick(runs, task, m):
    out = []
    for k, v in runs.items():
        if (
            k[0] == task
            and k[1] == m["encoder"]
            and k[2] == m["mode"]
            and k[3] == m["share"]
            and k[4] == m["chi"]
            and k[5] == m["groups"]
            and k[6] == "morton"
            and k[7] is True
            and k[8] is None
            and k[9] is None
            and (k[10] == (0, 0) or k[10] in ((7, 3), (4, 4)))  # default geometry per dataset
        ):
            out.extend(v)
    return out


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_all()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.0), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, task, title in ((axes[0], "cifar10", "CIFAR-10"), (axes[1], "multiclass", "MNIST")):
        rows = [(lab, m) for t, lab, m in ROWS if t == task]
        labels, means, errs, colors = [], [], [], []
        for lab, m in rows:
            vals = pick(runs, task, m)
            if not vals:
                continue
            labels.append(f"{lab}  (n={len(vals)})")
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            colors.append(PALETTE[m["chi"]])
        lo, hi = min(means), max(means)
        pad = (hi - lo) * 0.12 + 1e-3
        xlim = (max(0.0, lo - pad * 2), hi + pad * 3)
        y = np.arange(len(labels))
        ax.barh(y, means, xerr=errs, color=colors, height=0.62, error_kw=dict(ecolor=INK2, lw=1.2, capsize=3))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_title(f"{title} — test accuracy", loc="left", fontsize=11, color=INK)
        for yi, (mu, e) in enumerate(zip(means, errs)):
            ax.text(mu + e + (xlim[1] - xlim[0]) * 0.012, yi, f"{mu:.4f}", va="center", fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c]) for c in (4, 8, 16)]
    fig.legend(handles, ["bond dimension χ = 4", "χ = 8", "χ = 16"], loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=INK2)
    fig.suptitle(
        "QKS+TTN: what each architectural change is worth (15 epochs, fixed readout, mean ± s.e.m. over seeds)",
        x=0.01, ha="left", fontsize=12.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = os.path.join(HERE, "results", "study_summary.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
