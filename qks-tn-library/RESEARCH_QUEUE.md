# Overnight research queue (started 2026-08-25 ~21:45 PDT, 8 h budget)

Durable agenda for the autonomous loop. Each wake-up: (1) collect finished sweeps and analyse them,
(2) record findings here + in `FEEDFORWARD.md`, (3) start the next unstarted item, (4) commit.
Modal jobs are launched with `--detach` so they survive client/network drops.

Cost guide (T4 $0.59/h): a 15-epoch MNIST/CIFAR job is 6–15 min ≈ $0.10; keep the night under ~$40.

## Status legend
`[ ]` not started `[~]` running `[x]` done (result recorded) `[-]` dropped (with reason)

## Queue

- [x] **A. Collect `lenet-v2-*`** (CIFAR done; MNIST-overlap 3/6) — overlapping receptive fields (CIFAR 11×11/s3, MNIST 9×9/s3/p1) and
  G=2 feature maps, configs L1/L4 × 3 seeds. Launched at the start of the night. Analyse vs `lenet-v1` (CIFAR L1 0.3906,
  L4 0.4552; MNIST L1 0.8809, L4 0.9759).
- [x] **B. Collect `imagenet32-L5`** (done) — 4 jobs, 15 epochs, dense encoder, 1000-way linear head.
  At epoch 7: val acc 0.55–0.75 % (chance 0.1 %). Analyse when done; expect the 16-outcome readout to
  be the binding constraint → motivates item D.
- [x] **C. χ = 8 benchmark + MNIST/CIFAR sweeps.** χ = 8 is implemented (3 filters/register, 64-outcome
  readout, 2.29 M TN params for per-vertex B). First benchmark ms/step on T4 / L40S / A100-40GB with
  `--configs L4 --chi 8 --limit-train 9600 --epochs 2` (the step should become compute-bound, so the
  bigger GPUs may finally win on cost). Then 15-epoch sweeps `chi8-mnist`, `chi8-cifar` with
  configs L1,L4 (+ dense C0,C4 for the paper-model comparison), 3 seeds, on the cheapest GPU per step.
  **Benchmarked:** L40S 37 ms/step ($0.47/job), A100-40GB 50 ($0.68), T4 135 ($0.52) — as predicted
  the step is now compute-bound and the L40S wins on both speed and cost; use `--gpu L40S` for chi=8.
  Sweeps `chi8-cifar` and `chi8-mnist` (C0,C4,L1,L4 x 3 seeds, 15 epochs) launched on L40S.
- [x] **D. χ = 8 on ImageNet-32** (both arms done) — the 64-outcome readout is the point; 1 seed, configs C0/C4 or L1/L4,
  epochs 7 (cycle boundary) to bound cost. Only after C shows χ = 8 trains sanely.
- [x] **E. Two-stage (implemented, run, negative result) "deep kitchen sinks"** (design in FEEDFORWARD.md / the plan file): fold 2×2 blocks
  into the batch, run a 2-layer stage-1 tree, read its 16 classical probabilities per block, re-encode
  with a second patch-QKS on the 4×4×16 field, second tree, head. Implement only if time remains after
  C/D; helpers `fold_blocks`/`unfold_blocks` were specified but not written.
- [x] **F. Ablations** (finite shots done; ff-layers/centred still unrun): finite shots (`C2s16/128/1024`), `--ff-layers 5` and `4,5`,
  centred features (`C2c`). Cheap (~15 min/job); run on MNIST if GPU capacity is idle.
- [x] **G. Housekeeping** — keep `FEEDFORWARD.md` current; final summary of the night at the end.

## Findings log (append; newest last)

*Entries are numbered in the order they were recorded. (Earlier drafts carried wall-clock stamps that
were estimates rather than readings of the system clock; they have been replaced with ordinals. Verified
times: the night's work began ~21:30 PDT 2026-08-25 and this correction was made at 23:47.)*

- **[1]** χ = 8 implemented and tested (24/24 tests pass). χ = 4 remains bit-compatible: dense encoder
  matches the original `FeatureEncodingLayer` to 6e−8, and `fixed_readout_matrix(16,10)/(16,2)`
  reproduce `ConstMul`/`ConstMulBinary` exactly. χ = 8 dense encoder needs 192 episodes for 64 leaves;
  patch encoder uses 3 filters per register (147 params on CIFAR 4×4×3).
- **[2]** **chi = 8 early signal is strong**: 2 epochs on 9,600 CIFAR images gives val acc 0.322 vs 0.167
  for chi = 4 under the identical setting (L4, patch encoder). The 64-outcome readout and the 64x64
  vertex unitaries both help; full 15-epoch sweeps launched.
- **[3]** **Overlapping receptive fields: null result (item A, CIFAR).** 11x11/stride 3 (728 encoder
  params, 7x the field) gives L1 0.3835 +/- 0.0024 and L4 0.4496 +/- 0.0023 -- slightly *worse* than the
  non-overlapping 4x4/stride 4 default (0.3906 / 0.4552). Feedforward gain is unchanged (+6.6 +/- 0.5 pp).
  So receptive-field size is not a lever either; the encoder really is saturated at chi = 4. Drops the
  "overlapping patches" item from the expressivity list -- what is left is bond dimension and depth.
- **[4]** **G = 2 feature maps IS a lever (CIFAR, item A).** Two chi=4 registers per position (128 leaves,
  7 layers, 196 encoder params): L1 0.4043 +/- 0.0015 and **L4 0.4737 +/- 0.0013** vs G=1's 0.3906 / 0.4552
  -- **+1.85 pp on the best model**, +++ on every seed, for ~1.25x the step cost (26 vs 21 ms/step).
  Contrast with the overlapping-patch null: what helps is *more registers*, not a bigger receptive field
  or more encoder parameters. Launched G = 4 (256 leaves, 8 layers) on CIFAR to see if it keeps going.
- **[5]** **`ff-v2-L6` complete (63 epochs, depth 6, 3 seeds):** C0 0.9711 +/- 0.0014, C1 0.9720 +/- 0.0015,
  C2 0.9809 +/- 0.0009, **C4 0.9859 +/- 0.0003**. Every ff-v1 ordering survives 4x longer training:
  C2-C0 = +0.98 +/- 0.21 pp, C4-C1 = +1.39 +/- 0.18 pp (both +++ , McNemar p <= 3e-6); per-vertex
  unitaries alone remain worthless (C1-C0 = +0.08 +/- 0.08 pp, n.s. on all three seeds).
- **[6]** **MNIST overlapping patches (9x9/s3/p1): marginal.** L1 0.8952 +/- 0.0013, L4 0.9780 +/- 0.0009
  vs 7x7/s3's 0.8809 / 0.9759 -- a small (+0.2 pp on L4) gain, unlike the CIFAR case where overlap was
  slightly negative. Not a lever worth pursuing on either dataset.
- **[7]** **chi = 8 first results (dense C0, the paper's own model, 3 seeds):**
  MNIST **0.9747 +/- 0.0011** vs chi=4's 0.9682 +/- 0.0017 (**+0.65 pp**) -- note this *baseline* at chi=8
  now beats chi=4 *with* shared feedforward (0.9736);
  CIFAR **0.3218 +/- 0.0119** vs 0.2862 +/- 0.0089 (**+3.6 pp**). Doubling the register size helps the
  unmodified architecture everywhere, at 13 ms/step on an L40S (same as chi=4 on a T4). The chi=8
  feedforward arms (C4, L1, L4) are still running.
- **[8]** **chi = 8 on CIFAR-10 is the headline: every lever compounds.** (3 seeds, 15 epochs, L40S)

  | model | chi = 4 | chi = 8 | delta |
  |---|---|---|---|
  | C0 dense baseline (the paper's model) | 0.2862 | 0.3218 +/- 0.0119 | +3.6 pp |
  | C4 dense + feedforward B | 0.3501 | 0.3779 +/- 0.0100 | +2.8 pp |
  | L1 patch, no feedforward | 0.3906 | 0.4466 +/- 0.0029 | +5.6 pp |
  | **L4 patch + feedforward B** | 0.4552 | **0.5290 +/- 0.0023** | **+7.4 pp** |

  Cumulative over the paper's own model: **0.2862 -> 0.5290, +24.3 pp (85 % relative)**, from three
  independent changes -- locality, feedforward, bond dimension -- none of which cannibalises the others.
  The gains from locality (L4-C4 = +15.1 +/- 1.1 pp) and from feedforward (L4-L1 = +8.2 +/- 0.3 pp) are
  both *larger* at chi = 8 than at chi = 4 (+10.5 and +6.5 pp), i.e. a wider register lets the other two
  levers express more. Cost 36 ms/step on an L40S (~$0.47 per 15-epoch job).
  Launched chi=8 x G=2 (the two winners combined) on CIFAR, L4, 3 seeds.
- **[9]** **chi = 8 on MNIST (3 seeds, 15 epochs):** C0 0.9747 +/- 0.0011, L1 0.9386 +/- 0.0019,
  C4 0.9874 +/- 0.0003, **L4 0.9883 +/- 0.0003** -- the best MNIST number of the whole study (chi=4 was
  0.9820 dense / 0.9759 patch at 15 epochs, 0.9859 at 63 epochs). Two notes:
  (i) at chi = 8 the patch model *matches* the dense one on MNIST (L4-C4 = +0.09 +/- 0.03 pp, McNemar
  n.s. on all seeds) with **150 encoder parameters vs 150,720** -- locality stops being a handicap once
  the registers are wide enough; (ii) feedforward is still worth +4.97 +/- 0.20 pp with the patch encoder.
  MNIST is now clearly saturating (98.8 %), so CIFAR-10 and ImageNet-32 are the discriminating tasks.
- **[10]** **Feature maps saturate at G = 2 (CIFAR, L4):** G=1 0.4552 -> G=2 0.4737 (+1.85 pp) ->
  G=4 0.4798 +/- 0.0018 (+0.61 pp for 1.5x the step cost, 39 vs 26 ms/step). Diminishing returns.
  For comparison chi=8 with G=1 reaches 0.5290 at 36 ms/step -- **widening the registers beats adding
  more of them**, which is the same message as every other null tonight: what matters is how much
  quantum state each vertex carries, not how many classical features feed it.
- **[11]** **ImageNet-32 is not separating (both chi = 4 and chi = 8).** chi=4 at epoch 10-13:
  baseline 0.38 %, feedforward-B 0.19 % val acc; chi=8 at epoch 2: 0.19-0.26 %. Everything is within a
  few multiples of the 0.1 % chance rate and the configs are not ordered. Reading: 1000 classes through
  a 16- (or 64-) dimensional readout with a *linear* head is the binding constraint, and it is far more
  binding than anything the tree does -- consistent with the CIFAR finding that the fixed readout beats
  a trainable linear head. Report as a negative result; a real ImageNet attempt needs a much wider
  readout (chi >= 32, or several parallel trees concatenated), not more tree capacity.
- **[12]** **Two-stage "deep kitchen sinks" (partial: T1 3 seeds, T4 1 seed):** T1 0.3822 +/- 0.0024,
  T4 0.4420 vs the single-tree L1 0.3906 / L4 0.4552. The classical hand-off does *not* beat simply
  contracting one deeper tree, though it is far more parameter-efficient (20,910 TN params vs 79,305).
  Feedforward still helps inside it (+5.5 pp). Full numbers when the last two seeds land.
- **[13]** **Two-stage "deep kitchen sinks": complete, and it does not help (CIFAR, 3 seeds).**
  T1 0.3822 +/- 0.0024, **T4 0.4329 +/- 0.0055** vs the single-tree L1 0.3906 / L4 0.4552 -- i.e. -0.8 pp
  (baseline) and -2.2 pp (feedforward) for the extra classical hand-off. It *is* parameter-efficient
  (20,910 TN params vs 79,305), but it sits on the same accuracy-vs-parameters curve as the single tree
  (L2: 11,828 params -> 0.4022; T4: 21,298 -> 0.4329; L4: 79,403 -> 0.4552), so the measure-and-re-encode
  step buys nothing beyond the parameters it spends. Feedforward still helps inside it (+5.07 +/- 0.42 pp).
  Conclusion: of the four expressivity levers proposed, **bond dimension >> feature maps > (nothing) >
  receptive field = two-stage re-encoding**.
- **[14]** **chi = 8 and G = 2 do NOT stack (CIFAR, L4, 3 seeds):** 0.5303 +/- 0.0015 vs chi=8 alone
  0.5290 +/- 0.0023 -- +0.13 pp for 2x the parameters (4.65M vs 2.29M) and 1.6x the step cost
  (59 vs 36 ms/step). Once the registers are wide, extra feature maps are redundant: **bond dimension
  subsumes feature maps**. (At chi = 4, G=2 was worth +1.85 pp.) Best CIFAR configuration remains
  chi = 8, G = 1, patch-local, per-vertex feedforward B: **0.5290**.
- **[15]** **Finite-shot feedforward is cheap (MNIST, C2, 3 seeds)** -- the hardware-realism result:
  exact statistics 0.9736 +/- 0.0018; **1024 shots 0.9735 +/- 0.0014; 128 shots 0.9736 +/- 0.0015;
  16 shots 0.9712 +/- 0.0004**. 128 shots per vertex is indistinguishable from exact and even 16 shots
  costs only 0.24 pp. The multishot feedforward does not need deep statistics to work, which is the
  practical objection one would raise against Case B on real hardware.
- **[16]** **ImageNet-32, chi = 8 vs chi = 4 (baselines, seed 0):** chi=8 reaches **0.68 % test acc in 7
  epochs** vs chi=4's 0.48 % in 15 (chance 0.10 %), i.e. the 64-outcome readout is worth ~+40 % relative
  in less than half the training. The direction confirms the readout diagnosis, but 6.8x chance is still
  useless in absolute terms: a 1000-class problem read out through 64 numbers is the wrong shape.
  What would actually be needed: chi >= 32 (1024 outcomes), or K parallel trees whose readouts are
  concatenated, or a class-structured readout -- not more tree capacity. Recorded as the boundary of the
  approach rather than a result to chase.
- **[17]** **ImageNet-32 complete (1 seed each, chance = 0.10 %):**

  | model | epochs | test acc | x chance |
  |---|---|---|---|
  | chi=4 baseline shared | 15 | 0.0048 | 4.8 |
  | chi=4 baseline per-vertex | 15 | 0.0054 | 5.4 |
  | chi=4 **feedforward B** per-vertex | 15 | 0.0058 | 5.8 |
  | chi=4 **feedforward B** shared | 15 | 0.0067 | 6.7 |
  | chi=8 baseline shared | **7** | **0.0068** | 6.8 |

  Two readable signals despite the tiny absolute numbers: **feedforward helps** (0.0067 vs 0.0048 for the
  matched shared-unitary pair, +40 % relative) and **chi = 8 matches the best chi=4 model in half the
  epochs**. But every model is within an order of magnitude of chance -- reading 1000 classes out of 16
  (or 64) probabilities is the wrong shape of problem for this ansatz, and no amount of tree capacity
  fixes it. The honest conclusion is a scope statement: **this architecture is a 10-class-scale method**;
  scaling it to ImageNet needs a fundamentally wider readout (chi >= 32, parallel trees, or a structured
  class code), which is a different piece of work.
- **[18]** **chi = 8 ImageNet-32, both arms (7 epochs, seed 0):** baseline shared 0.0068 (6.8x chance),
  per-vertex feedforward B 0.0039 (3.9x). The feedforward arm is *worse* here -- 2.29 M TN parameters on
  1.28 M images for 7 epochs is under-trained, not over-capacity (its training accuracy was still rising
  at 0.0031). Nothing to conclude beyond the scope statement already recorded: ImageNet-32 is
  readout-bound for this ansatz and is not a productive target without a much wider readout.
- **[19]** **chi = 16 unblocked and running.** The dense Hermitian basis is (K, d, d): 0.13 GB at chi=8
  but **34 GB at chi=16**, which is why chi=8 was the practical ceiling. Replaced by a scatter-based
  construction (`params_to_hermitian_scatter`, auto-selected above chi=8, numerically identical to the
  basis version and covered by a test). First data point, CIFAR L0 (patch, shared unitaries, no
  feedforward), 1 seed: **chi=4 0.3277 -> chi=16 0.4240** (84 ms/step, 393,210 TN params, 256-outcome
  readout). The bond-dimension lever has still not saturated. The mode-B arm at chi=16 runs at
  447 ms/step (~3 h) and is in progress.

---

# FINAL SUMMARY OF THE NIGHT (written 00:25 PDT 2026-08-26)

All queue items are closed. **210 result files** across the study; tonight's Modal spend is roughly
**$30** (T4 for chi=4 work, L40S for chi=8), inside the $40 target.

## What was built
- **chi = 8 bond dimension** (`qks_tn_ff.py`): encoders generalised to k = log2(chi) filters per register
  with a k-fold Kronecker product; `fixed_readout_matrix` generalises the paper's 0/1 readout to any
  number of outcomes (and reproduces `ConstMul`/`ConstMulBinary` exactly at chi = 4, verified to 6e-8).
- **Two-stage "deep kitchen sinks"** (`qks_tn_twostage.py`): fold 2x2 blocks into the batch, contract a
  2-layer tree, re-encode the measured statistics with a second patch encoder and tree.
- 24 unit tests across `tests/test_ff_equivalence.py` and `tests/test_lenet.py`, all passing.

## The result that matters
On CIFAR-10, three independent changes to the published model compound:

| model | CIFAR-10 test acc |
|---|---|
| the paper's model (dense QKS, chi=4, shared unitaries) | 0.2862 +/- 0.0089 |
| + patch-local encoder | 0.3906 +/- 0.0027 |
| + multishot feedforward (mode B, per-vertex) | 0.4552 +/- 0.0042 |
| + bond dimension chi = 8 | **0.5290 +/- 0.0023** |

**+24.3 pp, 85 % relative**, every step positive on every seed. On MNIST the same stack reaches
**0.9883 +/- 0.0003** (paper: 0.9673), and there the 150-parameter patch encoder *matches* the
150,720-parameter dense one.

## Ranking of the expressivity levers (all measured, 3 seeds each)
1. **Bond dimension** chi 4 -> 8: +7.4 pp on the best CIFAR model; also subsumes feature maps.
2. **Feedforward** (mode B): +6.5 pp at chi=4, +8.2 pp at chi=8 -- and the gain *grows* with chi.
3. **Locality** (patch encoder): +10.5 pp at chi=4, +15.1 pp at chi=8.
4. **Feature maps** G: +1.85 pp at G=2, +0.61 pp more at G=4, **+0.13 pp at chi=8** (redundant once
   registers are wide).
5. **Nulls**: receptive-field size, Morton vs raster leaf order, position-shared vs per-position filters,
   trainable vs fixed readout (fixed is *better* by 8.5 pp), and the two-stage re-encoding (lands on the
   single tree's accuracy-vs-parameters curve).

## Practical findings
- **Finite shots are cheap**: 128 shots/vertex is indistinguishable from exact statistics; 16 shots costs
  0.24 pp. The obvious hardware objection to Case B does not bite.
- **Hardware economics flip with chi**: at chi=4 the step is kernel-launch-bound and a T4 is cheapest;
  at chi=8 it is compute-bound and an L40S is both fastest and cheapest (37 vs 135 ms/step, $0.47/job).
- **ImageNet-32 is out of scope** for this ansatz: everything lands within an order of magnitude of
  chance because 1000 classes are read out of 16-64 probabilities.
- Two pipeline bugs were found and fixed (windowed shuffle on class-sorted data; cosine-schedule drift
  from a ragged final batch), and per-epoch + per-annealing-cycle checkpointing was added.

## Deliverables
- `FEEDFORWARD.md` -- the full write-up (all sweeps, tables, curves, negative results).
- `img/study_summary.png` -- one figure covering every configuration at the common 15-epoch protocol
  (built by `qks-tn-library/make_summary_figure.py`; three bookkeeping bugs were found and fixed while
  building it: frozen-QKS runs were being pooled with trainable ones, two receptive-field geometries
  were averaged together, and `build_model` recorded `leaves = nepisodes // 2` which is wrong for
  chi != 4).
- `qks-tn-library/results/<sweep>/` -- per-epoch histories, per-cycle checkpoints, summaries.

## Suggested next steps
1. **chi = 16** -- unblocked tonight (scatter Hermitian) and confirmed still improving: the shared-unitary
   baseline goes 0.3277 (chi=4) -> 0.4240 (chi=16) on CIFAR. Worth a proper 3-seed sweep, and chi=32 needs
   only the same scatter path (though 1 M parameters per unitary makes it a different kind of model).
2. Longer training (63 epochs) of the best chi=8 configuration; at 15 epochs it was still improving.
3. A wider readout for many-class problems (parallel trees concatenated, or a structured class code) if
   ImageNet-scale is wanted at all.
4. Unrun ablations: `--ff-layers` (root-only / top-two) and centred features at chi = 8.
