# Overnight research queue (started 2026-08-25 ~21:45 PDT, 8 h budget)

Durable agenda for the autonomous loop. Each wake-up: (1) collect finished sweeps and analyse them,
(2) record findings here + in `FEEDFORWARD.md`, (3) start the next unstarted item, (4) commit.
Modal jobs are launched with `--detach` so they survive client/network drops.

Cost guide (T4 $0.59/h): a 15-epoch MNIST/CIFAR job is 6–15 min ≈ $0.10; keep the night under ~$40.

## Status legend
`[ ]` not started `[~]` running `[x]` done (result recorded) `[-]` dropped (with reason)

## Queue

- [x] **A. Collect `lenet-v2-*`** (CIFAR done; MNIST-overlap 3/6) — overlapping receptive fields (CIFAR 11×11/s3, MNIST 9×9/s3/p1) and
  G=2 feature maps, configs L1/L4 × 3 seeds. Launched 21:34. Analyse vs `lenet-v1` (CIFAR L1 0.3906,
  L4 0.4552; MNIST L1 0.8809, L4 0.9759).
- [~] **B. Collect `imagenet32-L5`** — 4 jobs, 15 epochs, dense encoder, 1000-way linear head.
  At epoch 7: val acc 0.55–0.75 % (chance 0.1 %). Analyse when done; expect the 16-outcome readout to
  be the binding constraint → motivates item D.
- [~] **C. χ = 8 benchmark + MNIST/CIFAR sweeps.** χ = 8 is implemented (3 filters/register, 64-outcome
  readout, 2.29 M TN params for per-vertex B). First benchmark ms/step on T4 / L40S / A100-40GB with
  `--configs L4 --chi 8 --limit-train 9600 --epochs 2` (the step should become compute-bound, so the
  bigger GPUs may finally win on cost). Then 15-epoch sweeps `chi8-mnist`, `chi8-cifar` with
  configs L1,L4 (+ dense C0,C4 for the paper-model comparison), 3 seeds, on the cheapest GPU per step.
  **Benchmarked 22:00:** L40S 37 ms/step ($0.47/job), A100-40GB 50 ($0.68), T4 135 ($0.52) — as predicted
  the step is now compute-bound and the L40S wins on both speed and cost; use `--gpu L40S` for chi=8.
  Sweeps `chi8-cifar` and `chi8-mnist` (C0,C4,L1,L4 x 3 seeds, 15 epochs) launched 22:02 on L40S.
- [~] **D. χ = 8 on ImageNet-32** (launched 23:05: C0,C4 x seed 0, 7 epochs, L40S, ~$11) — the 64-outcome readout is the point; 1 seed, configs C0/C4 or L1/L4,
  epochs 7 (cycle boundary) to bound cost. Only after C shows χ = 8 trains sanely.
- [ ] **E. Two-stage "deep kitchen sinks"** (design in FEEDFORWARD.md / the plan file): fold 2×2 blocks
  into the batch, run a 2-layer stage-1 tree, read its 16 classical probabilities per block, re-encode
  with a second patch-QKS on the 4×4×16 field, second tree, head. Implement only if time remains after
  C/D; helpers `fold_blocks`/`unfold_blocks` were specified but not written.
- [ ] **F. Ablations still unrun from ff-v1**: finite shots (`C2s16/128/1024`), `--ff-layers 5` and `4,5`,
  centred features (`C2c`). Cheap (~15 min/job); run on MNIST if GPU capacity is idle.
- [ ] **G. Housekeeping** — keep `FEEDFORWARD.md` current; final summary of the night at the end.

## Findings log (append; newest last)

- 21:45 χ = 8 implemented and tested (24/24 tests pass). χ = 4 remains bit-compatible: dense encoder
  matches the original `FeatureEncodingLayer` to 6e−8, and `fixed_readout_matrix(16,10)/(16,2)`
  reproduce `ConstMul`/`ConstMulBinary` exactly. χ = 8 dense encoder needs 192 episodes for 64 leaves;
  patch encoder uses 3 filters per register (147 params on CIFAR 4×4×3).
- 22:00 **chi = 8 early signal is strong**: 2 epochs on 9,600 CIFAR images gives val acc 0.322 vs 0.167
  for chi = 4 under the identical setting (L4, patch encoder). The 64-outcome readout and the 64x64
  vertex unitaries both help; full 15-epoch sweeps launched.
- 22:15 **Overlapping receptive fields: null result (item A, CIFAR).** 11x11/stride 3 (728 encoder
  params, 7x the field) gives L1 0.3835 +/- 0.0024 and L4 0.4496 +/- 0.0023 -- slightly *worse* than the
  non-overlapping 4x4/stride 4 default (0.3906 / 0.4552). Feedforward gain is unchanged (+6.6 +/- 0.5 pp).
  So receptive-field size is not a lever either; the encoder really is saturated at chi = 4. Drops the
  "overlapping patches" item from the expressivity list -- what is left is bond dimension and depth.
- 22:40 **G = 2 feature maps IS a lever (CIFAR, item A).** Two chi=4 registers per position (128 leaves,
  7 layers, 196 encoder params): L1 0.4043 +/- 0.0015 and **L4 0.4737 +/- 0.0013** vs G=1's 0.3906 / 0.4552
  -- **+1.85 pp on the best model**, +++ on every seed, for ~1.25x the step cost (26 vs 21 ms/step).
  Contrast with the overlapping-patch null: what helps is *more registers*, not a bigger receptive field
  or more encoder parameters. Launched G = 4 (256 leaves, 8 layers) on CIFAR to see if it keeps going.
- 22:40 **`ff-v2-L6` complete (63 epochs, depth 6, 3 seeds):** C0 0.9711 +/- 0.0014, C1 0.9720 +/- 0.0015,
  C2 0.9809 +/- 0.0009, **C4 0.9859 +/- 0.0003**. Every ff-v1 ordering survives 4x longer training:
  C2-C0 = +0.98 +/- 0.21 pp, C4-C1 = +1.39 +/- 0.18 pp (both +++ , McNemar p <= 3e-6); per-vertex
  unitaries alone remain worthless (C1-C0 = +0.08 +/- 0.08 pp, n.s. on all three seeds).
- 23:05 **MNIST overlapping patches (9x9/s3/p1): marginal.** L1 0.8952 +/- 0.0013, L4 0.9780 +/- 0.0009
  vs 7x7/s3's 0.8809 / 0.9759 -- a small (+0.2 pp on L4) gain, unlike the CIFAR case where overlap was
  slightly negative. Not a lever worth pursuing on either dataset.
- 23:05 **chi = 8 first results (dense C0, the paper's own model, 3 seeds):**
  MNIST **0.9747 +/- 0.0011** vs chi=4's 0.9682 +/- 0.0017 (**+0.65 pp**) -- note this *baseline* at chi=8
  now beats chi=4 *with* shared feedforward (0.9736);
  CIFAR **0.3218 +/- 0.0119** vs 0.2862 +/- 0.0089 (**+3.6 pp**). Doubling the register size helps the
  unmodified architecture everywhere, at 13 ms/step on an L40S (same as chi=4 on a T4). The chi=8
  feedforward arms (C4, L1, L4) are still running.
