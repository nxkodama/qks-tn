"""W49: run ``analyze_nodes.analyze`` on Modal (GPU) for checkpoints on the ``qks-tn-results`` Volume.

    python3 -m modal run -e env-ai-algs qks-tn-library-W49/modal_w49_analysis.py --tags LK16Baf_s0 --limit 0 --plan-kwargs ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("qks-tn-w49-analysis")
image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .pip_install("tensorflow-addons==0.23.0", "tensornetwork==0.4.6", "scipy", "numpy<2",
                 "datasets", "huggingface_hub", "pillow", "pyarrow")
    .add_local_dir(os.path.join(HERE, "common_functions"), remote_path="/root/qks/common_functions")
    .add_local_file(os.path.join(HERE, "run_ff_experiment.py"), remote_path="/root/qks/run_ff_experiment.py")
    .add_local_file(os.path.join(HERE, "analyze_nodes.py"), remote_path="/root/qks/analyze_nodes.py")
)
results_vol = modal.Volume.from_name("qks-tn-results")
data_vol = modal.Volume.from_name("qks-tn-data")

A = "cifar10_B_shared_holdout5000_kraus16_sched-cosine1_aug-flip_lr0.002_wd0.0002_patch4s4_g1_morton_chi8_s"
B = "cifar10_B_shared_holdout5000_kraus16_sched-cosine1_lr0.002_patch4s4_g1_morton_chi8_s"
RUNS = {
    "LK16Baf_s0": ("cptp-v10-W43-e63-lk", A + "0", "ckpt-63"),
    "LK16Baf_s1": ("cptp-v10-W43-e63-lk", A + "1", "ckpt-63"),
    "LK16B_s0": ("cptp-v7-W32-scr-lr0.002-wd0.0004", B + "0", "ckpt-15"),
    "LK16B_s1": ("cptp-v11-W48-lk16b-s1", B + "1", "ckpt-15"),
}


@app.function(image=image, gpu="L40S", cpu=8, memory=24576, timeout=6 * 60 * 60,
              volumes={"/out": results_vol, "/data": data_vol})
def run(tag: str, sweep: str, name: str, ckpt_name: str, limit: int, plan_names: str, plan_kwargs: str, bs: int, gpu_eigh: bool) -> dict:
    sys.path.insert(0, "/root/qks")
    import analyze_nodes as an

    results_vol.reload()
    data_vol.reload()
    d = f"/out/{sweep}/{name}"
    saved = json.load(open(f"{d}/result.json")) if os.path.exists(f"{d}/result.json") else None
    plan = an.passes_plan_w49(**(json.loads(plan_kwargs) if plan_kwargs else {}))
    if plan_names:
        keep = set(plan_names.split(","))
        plan = [p for p in plan if p[0] in keep]
    t0 = time.time()
    res = an.analyze(f"{d}/ckpt", ckpt_name, saved_result=saved, n_spec=0, bs=bs, limit=(limit or None), plan=plan,
                     log=lambda s: print(s, flush=True), spectrum=False, gpu_eigh=gpu_eigh)
    res["tag"] = tag
    res["modal_wall_s"] = time.time() - t0
    return res


@app.local_entrypoint()
def main(tags: str = "LK16Baf_s0", limit: int = 0, plan_names: str = "", plan_kwargs: str = "", bs: int = 128, gpu_eigh: bool = False, outdir: str = "", suffix: str = ""):
    out = Path(outdir or os.path.join(HERE, "..", "..", "lab-nathan", "projects", "cptp", "orch", "2026-11-02", "W49", "analysis"))
    out.mkdir(parents=True, exist_ok=True)
    hs = {t: run.spawn(t, *RUNS[t], limit, plan_names, plan_kwargs, bs, gpu_eigh) for t in tags.split(",")}
    for t, h in hs.items():
        res = h.get()
        (out / f"{t}{suffix}.json").write_text(json.dumps(res))
        print(f"wrote {out / (t + suffix + '.json')} wall {res['modal_wall_s']:.0f}s")
