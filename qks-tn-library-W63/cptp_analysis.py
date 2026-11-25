"""Checkpoint-only mechanism analysis for the registered CIFAR-10 chi=8 study.

Run with ``python3 -m modal run -e env-ai-algs qks-tn-library/cptp_analysis.py``.
Only the 15-epoch LK16B checkpoints are used; no optimizer step is taken.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SWEEP = "cptp-v2-chi8"
DEST = "cptp-v5-mechanism"
JOB = "cifar10_B_shared_kraus16_patch4s4_g1_morton_chi8_s{}"

app = modal.App("cptp-v5-mechanism")
image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .pip_install("tensorflow-addons==0.23.0", "tensornetwork==0.4.6", "scipy", "numpy<2",
                 "datasets", "huggingface_hub", "pillow", "pyarrow")
    .add_local_dir(str(HERE / "common_functions"), remote_path="/root/qks/common_functions")
    .add_local_file(str(HERE / "run_ff_experiment.py"), remote_path="/root/qks/run_ff_experiment.py")
)
results_vol = modal.Volume.from_name("qks-tn-results")
data_vol = modal.Volume.from_name("qks-tn-data")


@app.function(image=image, gpu="L40S", cpu=2, memory=16384, timeout=3 * 60 * 60,
              volumes={"/out": results_vol, "/data": data_vol})
def analyze_seed(seed: int) -> dict:
    import numpy as np
    import tensorflow as tf

    sys.path.insert(0, "/root/qks")
    import run_ff_experiment as rfe

    results_vol.reload()
    data_vol.reload()
    job = JOB.format(seed)
    src = Path("/out") / SWEEP / job
    saved = json.loads((src / "result.json").read_text())
    cfg = json.loads((src / "ckpt" / "cfg.json").read_text())
    assert cfg["config"] == "LK16B" and cfg["seed"] == seed
    assert cfg["task"] == "cifar10" and cfg["chi"] == 8 and cfg["kraus_rank"] == 16
    assert cfg["epochs"] == 15 and cfg["mode"] == "B" and cfg["share_per_layer"]
    assert cfg["encoder"] == "patch" and cfg["leaf_order"] == "morton" and cfg["groups"] == 1
    assert cfg.get("ff_layers") is None and cfg.get("ff_dim") is None
    assert {k: cfg[k] for k in cfg if k not in ("out", "ckpt_dir")} == {
        k: saved["cfg"][k] for k in cfg if k not in ("out", "ckpt_dir")}

    _, _, X_test, y_test = rfe.load_data("cifar10", "/data", log=lambda s: print(s, flush=True))
    effective_cfg = dict(rfe.DEFAULTS)
    effective_cfg.update(cfg)  # older saved configs predate fields with inert defaults
    model, qks, tn, _ = rfe.build_model(effective_cfg, saved["steps_per_epoch"], p=X_test.shape[1], n_classes=10)
    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    restore = str(src / "ckpt" / "cycle-015" / "ckpt")
    assert tf.io.gfile.exists(restore + ".index")
    ckpt.restore(restore).expect_partial()
    chi, R, nlayer = tn.chi, tn.R, tn.nlayers
    assert (chi, R, nlayer) == (8, 16, 6)

    # The conditional Kraus spectrum is the sorted eigenvalue spectrum of
    # Gram(K_e, K_f) / d, equivalent to the normalized Choi spectrum.
    # The entropy is S(Tr_keep V rho V*) in bits for each node and test input.
    @tf.function
    def forward(xb, fixed=(), dephase=False, statistics=False):
        msgs = tf.transpose(qks(xb), [0, 3, 1, 2])
        ff_input = None
        p_sums, spectra, entropies = [], [], []
        for l in range(nlayer):
            n = tn.n_vertices[l]
            rr = R if l < nlayer - 1 else tn.Rroot
            rho = tf.einsum("nvim,nvjo->nvijmo", msgs[:, 0::2], msgs[:, 1::2])
            rho = tf.reshape(rho, [-1, n, chi * chi, chi * chi])
            V = tn.isometry(l, ff_input)
            if statistics and l < nlayer - 1:
                vv = tf.broadcast_to(V, [tf.shape(xb)[0], n, chi * chi, chi * rr])
                K = tf.transpose(tf.reshape(vv, [-1, n, chi * chi, rr, chi]), [0, 1, 3, 4, 2])
                gram = tf.einsum("bneki,bnfki->bnef", tf.math.conj(K), K)
                eig = tf.linalg.eigvalsh(gram)
                eig = tf.sort(tf.maximum(tf.math.real(eig), 0.0), axis=-1, direction="DESCENDING")
                spectra.append(tf.reduce_sum(eig / (chi * chi), axis=[0, 1]))
            sigma = tf.matmul(tf.matmul(V, rho, transpose_a=True), tf.math.conj(V))
            s6 = tf.reshape(sigma, [-1, n, rr, chi, rr, chi])
            if statistics and l < nlayer - 1:
                env = tf.einsum("bneifi->bnef", s6)
                lam = tf.maximum(tf.math.real(tf.linalg.eigvalsh(env)), 0.0)
                ent = -tf.reduce_sum(lam * (tf.math.log(tf.maximum(lam, 1e-30)) / tf.math.log(2.0)), axis=-1)
                entropies.append(tf.reduce_sum(ent))
            if dephase and l < nlayer - 1:
                eye = tf.reshape(tf.eye(rr, dtype=s6.dtype), [1, 1, rr, 1, rr, 1])
                s6 = s6 * eye
            diag = tf.math.real(tf.linalg.diag_part(tf.reshape(s6, [-1, n, rr * chi, rr * chi])))
            if l < nlayer - 1:
                msgs = tf.einsum("bneiej->bnij", s6)
                p = tf.reduce_sum(tf.reshape(diag, [-1, n, rr, chi]), axis=-1)
                p_sums.append(tf.reduce_sum(p, axis=0))
                if l in fixed:
                    p = tf.broadcast_to(fixed[l][tf.newaxis], tf.shape(p))
                ff_input = tf.concat([p[:, 0::2], p[:, 1::2]], axis=-1)
            else:
                out = tf.reshape(tf.transpose(tf.reshape(diag, [-1, rr, chi]), [0, 2, 1]), [-1, chi * rr])
        probs = model.layers[-1](out)
        return probs, p_sums, spectra, entropies

    # Keep the collection pass separate from ablation runs. Fixed features are
    # the per-node means across all 10,000 test examples of the trained model.
    batch = int(cfg["batch_size"])
    batches = [(tf.convert_to_tensor(X_test[i:i + batch], tf.float32) / 255.0,
                y_test[i:i + batch]) for i in range(0, len(y_test), batch)]
    ntest = len(y_test)
    p_total = [np.zeros((n, R), np.float64) for n in tn.n_vertices[:-1]]
    s_total = [np.zeros(R, np.float64) for _ in range(nlayer - 1)]
    h_total = np.zeros(nlayer - 1, np.float64)
    baseline_pred = []
    for xb, _ in batches:
        probs, ps, ss, hs = forward(xb, statistics=True)
        baseline_pred.extend(np.argmax(probs.numpy(), axis=-1).tolist())
        for l in range(nlayer - 1):
            p_total[l] += ps[l].numpy()
            s_total[l] += ss[l].numpy()
            h_total[l] += float(hs[l])
    baseline_acc = float(np.mean(np.asarray(baseline_pred) == y_test))
    saved_acc = float(saved["final_test"]["acc"])
    assert abs(baseline_acc - saved_acc) <= 1e-4, (baseline_acc, saved_acc)
    fixed_mean = tuple(tf.constant(p / ntest, tf.float32) for p in p_total)

    def accuracy(fixed=(), dephase=False):
        pred = []
        for xb, _ in batches:
            probs, _, _, _ = forward(xb, fixed=fixed, dephase=dephase)
            pred.extend(np.argmax(probs.numpy(), axis=-1).tolist())
        return float(np.mean(np.asarray(pred) == y_test))

    dephased_acc = accuracy(dephase=True)
    fixed_per_layer = [accuracy(fixed={l: fixed_mean[l]}) for l in range(nlayer - 1)]
    fixed_all_acc = accuracy(fixed={l: fixed_mean[l] for l in range(nlayer - 1)})
    result = dict(seed=seed, sweep=SWEEP, job=job, checkpoint=restore, saved_cfg=cfg,
                  n_test=ntest, baseline_acc=baseline_acc, saved_test_acc=saved_acc,
                  dephased_acc=dephased_acc, fixed_per_layer_acc=fixed_per_layer,
                  fixed_all_acc=fixed_all_acc,
                  kraus_spectrum=[(s / (ntest * tn.n_vertices[l])).tolist() for l, s in enumerate(s_total)],
                  environment_entropy_bits=[float(h / (ntest * tn.n_vertices[l])) for l, h in enumerate(h_total)],
                  fixed_feature_mean=[v.numpy().tolist() for v in fixed_mean])
    dest = Path("/out") / DEST
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"e1_e2_seed{seed}.json").write_text(json.dumps(result, indent=2))
    results_vol.commit()
    print(json.dumps({k: v for k, v in result.items() if k not in ("saved_cfg", "fixed_feature_mean")}), flush=True)
    return result


@app.local_entrypoint()
def main(seeds: str = "0,1,2"):
    target = HERE / "results" / DEST
    target.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in seeds.split(",")]
    calls = [analyze_seed.spawn(s) for s in seed_list]
    for s, call in zip(seed_list, calls):
        result = call.get()
        (target / f"e1_e2_seed{s}.json").write_text(json.dumps(result, indent=2))
        print(f"saved seed {s} in {target}", flush=True)
