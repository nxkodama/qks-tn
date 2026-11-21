"""Run QKS+TTN feedforward experiments as concurrent GPU jobs on Modal.

Each (config, seed) pair is one job.  Per-epoch train/validation metrics are written to
the ``qks-tn-results`` Volume (``/out/<sweep>/<job>/history.json``, committed after every
epoch so progress is visible mid-run) and the final result JSON is also written locally to
``qks-tn-library/results/<sweep>/<job>.json`` by the local entrypoint.

Smoke test (1 epoch on 2000 images)::

    python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0 --seeds 0 --epochs 1 --limit-train 2000

Full multiclass sweep::

    python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C1,C2,C3,C4,C5,F0,F2,F5 --seeds 0,1,2

Continue the same jobs for more epochs (resumes from the per-epoch checkpoints on the Volume;
pass ``--fresh`` to ignore them). Cycle-aligned totals for the cosine-restart schedule are 15, 31, 63::

    python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C2,C4 --seeds 0,1,2 --epochs 31 --sweep ff-v1

GPU type: ``--gpu A10G`` (or ``QKS_GPU=A10G`` env; default ``T4``; ``none`` for CPU).
Other datasets: ``--task cifar10`` / ``--task imagenet32`` (``--nepisodes 64`` for a 5-layer tree);
the dataset is downloaded/decoded once into the ``qks-tn-data`` Volume by a CPU job.
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
    .pip_install("tensorflow-addons==0.23.0", "tensornetwork==0.4.6", "scipy", "numpy<2",
                 "datasets", "huggingface_hub", "pillow", "pyarrow")
    .add_local_dir(os.path.join(HERE, "common_functions"), remote_path="/root/qks/common_functions")
    .add_local_file(os.path.join(HERE, "run_ff_experiment.py"), remote_path="/root/qks/run_ff_experiment.py")
)

results_vol = modal.Volume.from_name("qks-tn-results", create_if_missing=True)
data_vol = modal.Volume.from_name("qks-tn-data", create_if_missing=True)


# Host memory per task: the training set is held in a CPU tf.Variable (uint8 for images).
TASK_MEMORY_MB = {"binary": 8192, "multiclass": 8192, "cifar10": 8192, "imagenet32": 24576, "timn100": 8192, "svhn": 8192}


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 60 * 60, volumes={"/data": data_vol})
def prepare_data(task: str) -> dict:
    """Download / decode a dataset into the shared data Volume once (CPU only)."""
    import time

    sys.path.insert(0, "/root/qks")
    import run_ff_experiment as rfe

    data_vol.reload()
    t0 = time.time()
    X_train, y_train, X_test, y_test = rfe.load_data(task, "/data", log=lambda s: print(s, flush=True))
    data_vol.commit()
    info = dict(task=task, n_train=int(len(y_train)), n_test=int(len(y_test)), dim=int(X_train.shape[1]),
                dtype=str(X_train.dtype), seconds=time.time() - t0)
    print(f"[prepare_data] {info}", flush=True)
    return info


def _train_job(cfg: dict, sweep: str) -> dict:
    import time

    sys.path.insert(0, "/root/qks")
    import run_ff_experiment as rfe

    cfg = dict(cfg)
    cfg["cache_dir"] = "/data"
    out_dir = f"/out/{sweep}/{cfg['name']}"
    results_vol.reload()
    os.makedirs(out_dir, exist_ok=True)
    cfg["out"] = f"{out_dir}/result.json"
    # per-epoch checkpoints live on the Volume next to the metrics, so re-running the same
    # sweep with a larger --epochs resumes each job where it stopped
    cfg["ckpt_dir"] = f"{out_dir}/ckpt"
    cfg.setdefault("resume", True)

    # Dataset cache shared across jobs (prepare_data / the first job computes, later ones reuse).
    data_vol.reload()
    t0 = time.time()
    rfe.load_data(cfg["task"], cfg["cache_dir"], log=lambda s: print(s, flush=True))
    data_vol.commit()
    print(f"[{cfg['name']}] data ready in {time.time() - t0:.0f}s", flush=True)

    def on_epoch(state):
        with open(f"{out_dir}/history.json", "w") as f:
            json.dump(state, f)
        results_vol.commit()

    res = rfe.train(cfg, on_epoch=on_epoch, log=lambda s: print(s, flush=True))
    results_vol.commit()
    return res


@app.cls(
    image=image,
    gpu=GPU_SPEC,
    cpu=4 if GPU_SPEC is None else 2,
    memory=8192,
    timeout=14 * 60 * 60,  # W39, W44: 255-epoch LK16B runs need about 9.5 h
    volumes={"/out": results_vol, "/data": data_vol},
)
class Trainer:
    """One training job per call; GPU / memory are overridden per run via ``Trainer.with_options``."""

    @modal.method()
    def train(self, cfg: dict, sweep: str) -> dict:
        return _train_job(cfg, sweep)


@app.local_entrypoint()
def main(
    configs: str = "C0,C1,C2,C3,C4,C5,F0,F2,F5",
    seeds: str = "0,1,2",
    epochs: int = 15,
    task: str = "multiclass",
    sweep: str = "ff-v1",
    limit_train: int = 0,
    fresh: bool = False,
    nepisodes: int = 128,
    gpu: str = "",
    memory_mb: int = 0,
    groups: int = 0,
    patch: int = 0,
    stride: int = 0,
    pad: int = -1,
    sigma_scale: float = 0.0,
    head: str = "",
    chi: int = 0,
    holdout_5000: bool = False,
    lr: float = 0.0,
    wd: float = -1.0,
    schedule: str = "",
):
    """``fresh=True`` ignores existing checkpoints on the Volume and retrains from scratch.
    ``nepisodes`` sets the tree depth (128 -> 6 layers, 256 -> 7, 512 -> 8, 64 -> 5).
    ``gpu`` overrides the GPU type for this run (T4, L4, A10G, L40S, A100-40GB, A100-80GB, H100);
    ``memory_mb`` overrides host memory (default depends on the task)."""
    sys.path.insert(0, HERE)
    import run_ff_experiment as rfe

    gpu_name = gpu or GPU
    mem = int(memory_mb) or TASK_MEMORY_MB.get(task, 8192)
    job = Trainer.with_options(gpu=(None if gpu_name.lower() == "none" else gpu_name), memory=mem)()

    if task in ("cifar10", "imagenet32", "timn100", "svhn"):
        print(f"preparing {task} on the data Volume (no-op if cached) ...")
        print(prepare_data.remote(task))

    cfgs = []
    for c in [x.strip() for x in configs.split(",") if x.strip()]:
        for s in [int(x) for x in seeds.split(",") if x.strip()]:
            ov = dict(rfe.CONFIGS[c])
            ov.update(task=task, seed=s, epochs=epochs, nepisodes=int(nepisodes))
            if groups:
                ov["groups"] = int(groups)
            if patch:
                ov["patch"] = int(patch)
            if stride:
                ov["stride"] = int(stride)
            if pad >= 0:
                ov["pad"] = int(pad)
            if sigma_scale:
                ov["sigma_scale"] = float(sigma_scale)
            if head:
                ov["head"] = head
            if chi:
                ov["chi"] = int(chi)
                if not ov.get("encoder", "dense") == "patch":
                    # dense encoder: leaves = nepisodes / log2(chi); keep 64 leaves (6 layers)
                    import math

                    ov["nepisodes"] = int(nepisodes // 2 * int(math.log2(int(chi))))
            if limit_train:
                ov["limit_train"] = int(limit_train)
            if holdout_5000:
                ov["holdout_5000"] = True
            if lr:
                ov["lr"] = float(lr)
            if wd >= 0:
                ov["wd"] = float(wd)
            if schedule:
                ov["schedule"] = schedule
            cfg = rfe.make_cfg(**ov)
            cfg["config"] = c
            cfg["resume"] = not fresh
            cfgs.append(cfg)
    print(f"spawning {len(cfgs)} jobs on {gpu_name} / {mem} MB (sweep={sweep}, task={task}, epochs={epochs}, nepisodes={nepisodes})")
    handles = [job.train.spawn(cfg, sweep) for cfg in cfgs]

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
            epochs=len(res["epochs"]),
            resumed_from_epoch=res.get("resumed_from_epoch", 0),
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
