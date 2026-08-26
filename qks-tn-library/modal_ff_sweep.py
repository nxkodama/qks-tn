"""Run QKS+TTN feedforward experiments as concurrent GPU jobs on Modal.

Each (config, seed) pair is one job.  Per-epoch train/validation metrics are written to
the ``qks-tn-results`` Volume (``/out/<sweep>/<job>/history.json``, committed after every
epoch so progress is visible mid-run) and the final result JSON is also written locally to
``qks-tn-library/results/<sweep>/<job>.json`` by the local entrypoint.

Smoke test (1 epoch on 2000 images)::

    python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0 --seeds 0 --epochs 1 --limit-train 2000

Full multiclass sweep::

    python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C1,C2,C3,C4,C5,F0,F2,F5 --seeds 0,1,2

GPU type: ``QKS_GPU=A10G python3 -m modal run ...`` (default ``T4``; ``QKS_GPU=none`` for CPU).
Mid-run progress::

    python3 -m modal volume ls qks-tn-results ff-v1
    python3 -m modal volume get qks-tn-results ff-v1/<job>/history.json -
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("QKS_GPU", "T4")
GPU_SPEC = None if GPU.lower() == "none" else GPU

app = modal.App("qks-tn-feedforward")

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .pip_install("tensorflow-addons==0.23.0", "tensornetwork==0.4.6", "scipy", "numpy<2")
    .add_local_dir(os.path.join(HERE, "common_functions"), remote_path="/root/qks/common_functions")
    .add_local_file(os.path.join(HERE, "run_ff_experiment.py"), remote_path="/root/qks/run_ff_experiment.py")
)

results_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=4 if GPU_SPEC is None else 2,
    memory=8192,
    timeout=4 * 60 * 60,
    volumes={"/out": results_vol, "/data": data_vol},
)
def train_job(cfg: dict, sweep: str) -> dict:
    import time

    sys.path.insert(0, "/root/qks")
    import run_ff_experiment as rfe

    cfg = dict(cfg)
    cfg["cache_dir"] = "/data"
    out_dir = f"/out/{sweep}/{cfg['name']}"
    os.makedirs(out_dir, exist_ok=True)
    cfg["out"] = f"{out_dir}/result.json"

    # Deskewed-MNIST cache shared across jobs (first job computes, later ones reuse).
    data_vol.reload()
    t0 = time.time()
    rfe.load_data(cfg["task"], cfg["cache_dir"])
    data_vol.commit()
    print(f"[{cfg['name']}] data ready in {time.time() - t0:.0f}s", flush=True)

    def on_epoch(state):
        with open(f"{out_dir}/history.json", "w") as f:
            json.dump(state, f)
        results_vol.commit()

    res = rfe.train(cfg, on_epoch=on_epoch, log=lambda s: print(s, flush=True))
    results_vol.commit()
    return res


@app.local_entrypoint()
def main(
    configs: str = "C0,C1,C2,C3,C4,C5,F0,F2,F5",
    seeds: str = "0,1,2",
    epochs: int = 15,
    task: str = "multiclass",
    sweep: str = "ff-v1",
    limit_train: int = 0,
):
    sys.path.insert(0, HERE)
    import run_ff_experiment as rfe

    cfgs = []
    for c in [x.strip() for x in configs.split(",") if x.strip()]:
        for s in [int(x) for x in seeds.split(",") if x.strip()]:
            ov = dict(rfe.CONFIGS[c])
            ov.update(task=task, seed=s, epochs=epochs)
            if limit_train:
                ov["limit_train"] = int(limit_train)
            cfg = rfe.make_cfg(**ov)
            cfg["config"] = c
            cfgs.append(cfg)
    print(f"spawning {len(cfgs)} jobs on {GPU} (sweep={sweep}, task={task}, epochs={epochs})")
    handles = [train_job.spawn(cfg, sweep) for cfg in cfgs]

    out_dir = Path(HERE) / "results" / sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for cfg, h in zip(cfgs, handles):
        row = dict(name=cfg["name"], config=cfg["config"], seed=cfg["seed"])
        try:
            res = h.get()
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {cfg['name']}: {exc}")
            row["error"] = str(exc)
            summary.append(row)
            continue
        (out_dir / f"{cfg['name']}.json").write_text(json.dumps(res))
        row.update(
            final_test=res["final_test"],
            best_val=res["best_val"],
            params=res["params"],
            ms_per_step=res["ms_per_step"],
            wall_min=res["total_wall_s"] / 60,
            devices=res["devices"],
        )
        summary.append(row)
        print(
            f"{cfg['name']}: test_acc {res['final_test']['acc']:.4f} test_loss {res['final_test']['loss']:.4f} "
            f"best_val_acc {res['best_val']['val_acc']:.4f}@{res['best_val']['epoch']} "
            f"({res['ms_per_step']:.0f} ms/step, {res['total_wall_s']/60:.1f} min, {res['devices']})"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json")
