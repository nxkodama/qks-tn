# Feedforward measurement in QKS+TTN

Does using the discarded register's measurement outcome — instead of tracing it out — improve
image classification with the QKS+TTN ansatz of [arXiv:2208.13895](https://arxiv.org/abs/2208.13895)?

Design note: *Feedforward measurement in QKS+TTN: single-shot vs multishot* (Notion). Two variants:

| | Case A — single-shot | Case B — multishot statistics |
|---|---|---|
| fed-forward object | outcome bits `b` of the discarded register | outcome probabilities `p = diag(Tr_keep σ)` |
| parent unitary | stack `U(b_L, b_R)` (χ² = 16 unitaries per layer) | `expm(i(H0 + Σ_k p_L[k] H_k^L + Σ_k p_R[k] H_k^R))` |
| linear in the input state | yes → exact tensor network | no → one forced bottom-up sweep |
| expressivity vs. baseline | none (block-diagonal widening of the parent) | genuinely nonlinear |

![what each architectural change is worth](img/study_summary.png)

*Every configuration at the same 15-epoch protocol with the paper's fixed readout, 3 seeds each. The
published model is the top bar of each panel; colour is bond dimension. Regenerate with
`.venv/bin/python qks-tn-library/make_summary_figure.py`.*

## Code

Everything lives in `qks-tn-library/`; the original notebooks and `common_functions/qks_tn.py` are untouched.

- `common_functions/qks_tn_ff.py` — `FeedforwardTNLayer(mode='baseline'|'A'|'B', share_per_layer, shots,
  ff_layers, ff_features, ff_null)`: the dissipative TTN as an explicit batched bottom-up sweep (no
  `vectorized_map`, no `tensornetwork`), plus `FastFeatureEncodingLayer` (batched QKS with identical
  weights/initialisation). `mode='baseline'` reproduces `qks_tn.TNLayer` to 1e-7 (forward) and 1e-4
  (relative gradient) and is ~4.6× faster on CPU.
- `tests/test_ff_equivalence.py` — 13 tests: old-vs-new equivalence, zero-coupling reductions of A and B
  to the baseline, unitarity, normalisation, shot sampling, finite differences. Run with
  `.venv/bin/python qks-tn-library/tests/test_ff_equivalence.py`.
- `run_ff_experiment.py` — notebook-faithful pipeline (deskewed MNIST, AdamW + cosine restarts, batch 32,
  15 epochs) with named configs (`--config C0 … F5`), per-epoch train/validation logging to JSON.
- `modal_ff_sweep.py` — one Modal GPU job per (config, seed); per-epoch history on the `qks-tn-results`
  Volume; `python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C2 --seeds 0,1,2`.
- `analyze_ff_runs.py` — tables (mean ± s.e.m., paired per-seed differences, McNemar tests) and curves.

Two facts about the original code that the implementation had to respect: the vertex map is
`σ = Uᵀ ρ conj(U)` (from how `tf.reshape` orders the legs), and all vertices of a layer share one unitary
(only 6 of `TNLayer`'s 63 parameter columns are live — 1,530 of 16,065 parameters).

## Experiment (sweep `ff-v1`, 10-class MNIST, 3 seeds per config, T4 GPUs)

Same seed ⇒ same QKS initialisation, same base unitaries, same data order across configs.
Validation = the 10k test set evaluated every epoch (training set is the paper's full 60k).

| config | model | TN params | trainable | test acc | test CE |
|---|---|---|---|---|---|
| C0 | baseline, shared unitaries (the paper's model) | 1,530 | 102,010 | 0.9682 ± 0.0017 | 0.151 ± 0.007 |
| C1 | baseline, per-vertex unitaries | 16,065 | 116,545 | 0.9671 ± 0.0015 | 0.144 ± 0.004 |
| C3 | B with constant *p* (information-free control) | 11,730 | 112,210 | 0.9672 ± 0.0018 | 0.154 ± 0.008 |
| C5 | **A: single-shot feedforward** | 21,930 | 122,410 | **0.9729 ± 0.0009** | 0.119 ± 0.002 |
| C2 | **B: multishot feedforward** | 11,730 | 112,210 | **0.9736 ± 0.0018** | 0.114 ± 0.010 |
| C4 | **B, per-vertex** | 79,305 | 179,785 | **0.9820 ± 0.0003** | 0.068 ± 0.003 |
| F0 | baseline, QKS frozen (TN-only learning) | 1,530 | 1,530 | 0.641 ± 0.026 | 1.40 ± 0.02 |
| F5 | A, QKS frozen | 21,930 | 21,930 | 0.789 ± 0.011 | 1.10 ± 0.01 |
| F2 | B, QKS frozen | 11,730 | 11,730 | **0.868 ± 0.004** | 0.68 ± 0.02 |

Paired per-seed differences (candidate − control; McNemar exact test on test-set errors):

| comparison | isolates | Δ acc (pp) | Δ CE | sign by seed | McNemar p |
|---|---|---|---|---|---|
| C1 − C0 | 10× more TN parameters, no feedforward | −0.11 ± 0.05 | −0.007 | − − − | n.s. all seeds |
| C3 − C0 | B's extra parameters with *p* held constant | −0.10 ± 0.16 | +0.003 | + − − | n.s. (one seed worse) |
| C2 − C3 | the fed-forward *information* itself | +0.63 ± 0.36 | −0.040 | + + + | 4e-4, 4e-20, 0.44 |
| C2 − C0 | multishot feedforward vs the paper's model | +0.53 ± 0.29 | −0.037 | + + − | 6e-6, 3e-10, 0.94 |
| C5 − C0 | single-shot feedforward vs the paper's model | +0.47 ± 0.24 | −0.032 | + + + | 5e-11, 2e-3, 0.65 |
| C5 − C1 | A vs parameter-matched baseline | +0.58 ± 0.23 | −0.024 | + + + | 1e-11, 3e-3, 0.09 |
| C4 − C1 | per-vertex B vs per-vertex baseline | **+1.48 ± 0.12** | **−0.075** | + + + | 1e-27, 5e-24, 5e-20 |
| F2 − F0 | B, TN-only learning | **+22.7 ± 2.4** | −0.72 | + + + | ≈ 0 |
| F5 − F0 | A, TN-only learning | **+14.8 ± 2.7** | −0.30 | + + + | ≈ 0 |

![trainable QKS](img/ff-v1_curves_trainable_QKS.png)
![frozen QKS](img/ff-v1_curves_frozen_QKS.png)

## Reading

1. **Feedforward helps, and the gain comes from the information, not the parameters.** More unitaries
   alone (C1) or the same B parametrisation fed a constant (C3) do nothing; B fed the real statistics
   (C2) beats its information-free twin on every seed. Adding unitaries *without* information is even
   slightly negative.
2. **Per-vertex B is the clear winner with trainable features: 98.2 % vs 96.8 %** (−55 % test cross-entropy),
   consistent on all three seeds (σ = 0.03 pp). With shared unitaries the two feedforward variants are
   similar (+0.5 pp; A is the more seed-stable, B the lower-loss), and one seed of B is a wash.
3. **When only the tensor network learns, feedforward changes the picture qualitatively**: 64 % → 87 % (B)
   and 79 % (A). This is the setting where the design note's expressivity argument is tested most
   directly, and B — the only variant that is nonlinear in the input state — is well ahead of A.
4. Cost is modest: on a T4, baseline 13 ms/step, A 18, B 27 (per-vertex B 29); a 15-epoch multiclass run
   is 8–16 minutes.

Decision rule stated before the sweep (paired difference vs the parameter-matched control positive on
every seed and > 2 s.e.m.): met by C4 vs C1, C5 vs C1, F2 vs F0, F5 vs F0; C2 vs C3 is positive on all
seeds but its s.e.m. (0.36 pp) is inflated by the seed-2 tie, so it needs more seeds to clear the bar.

### Bond dimension chi = 8 (sweep `chi8-cifar`, 3 seeds, 15 epochs)

Registers become 3 qubits, vertex unitaries 64x64 (4,095 parameters each) and the readout 64
probabilities instead of 16. The encoder generalises to log2(chi) filters per register; chi = 4 remains
bit-compatible with the published code.

| model | chi = 4 | chi = 8 |
|---|---|---|
| C0 dense baseline (the paper's model) | 0.2862 ± 0.0089 | 0.3218 ± 0.0119 |
| C4 dense + feedforward B | 0.3501 ± 0.0084 | 0.3779 ± 0.0100 |
| L1 patch-local, no feedforward | 0.3906 ± 0.0027 | 0.4466 ± 0.0029 |
| **L4 patch-local + feedforward B** | 0.4552 ± 0.0042 | **0.5290 ± 0.0023** |

On MNIST the same sweep gives C0 0.9747 ± 0.0011, C4 0.9874 ± 0.0003 and **L4 0.9883 ± 0.0003** (the
best MNIST result here), and the patch model *matches* the dense one (+0.09 ± 0.03 pp, McNemar n.s.)
with 150 encoder parameters against 150,720 — locality stops costing anything once the registers are wide.

**Locality, feedforward and bond dimension compound**: 0.2862 → 0.5290 is +24.3 pp (85 % relative) over
the published model, and the locality and feedforward gains are each *larger* at chi = 8 (+15.1 and
+8.2 pp) than at chi = 4 (+10.5 and +6.5 pp) — a wider register gives the other two levers more to work
with. At 64x64 unitaries the step becomes compute-bound, which flips the hardware economics: an L40S is
both the fastest and the cheapest option (37 ms/step, $0.47 per 15-epoch job) where a T4 was cheapest at
chi = 4.

### Two-stage "deep kitchen sinks" (`qks_tn_twostage.py`, sweep `twostage-cifar`)

LeNet's second convolution has a quantum analogue: measure after the first pooling stage and re-encode
the resulting statistics. Implemented by folding each 2×2 block into the batch (so stage-1 unitaries are
shared across blocks, as a convolution's are), contracting a 2-layer tree, scattering the 16 block
outcome distributions back onto a 4×4×16 classical field, and re-encoding it with a second 3×3 patch
encoder and tree.

| model | TN params | CIFAR-10 test acc |
|---|---|---|
| L2 single tree, shared B | 11,730 | 0.4022 ± 0.0043 |
| **T4 two-stage, per-vertex B** | 20,910 | 0.4329 ± 0.0055 |
| L4 single tree, per-vertex B | 79,305 | 0.4552 ± 0.0042 |

**Negative result**: the two-stage model lands *on* the single tree's accuracy-vs-parameters curve, not
above it — the classical re-encoding buys nothing beyond the parameters it spends (−2.2 pp against L4 at
4× fewer parameters). Feedforward still helps inside it (+5.1 ± 0.4 pp). Ranking of the expressivity
levers tested: **bond dimension ≫ feature maps > receptive field ≈ two-stage re-encoding ≈ nothing**.

### Finite-shot feedforward (sweep `shots-mnist`, config C2, 3 seeds)

Mode B feeds *estimated* probabilities on hardware, so the estimate is noisy. Replacing the exact `p`
with a Multinomial(M, p)/M sample per vertex per example (straight-through gradient):

| shots M | MNIST test acc |
|---|---|
| exact | 0.9736 ± 0.0018 |
| 1024 | 0.9735 ± 0.0014 |
| 128 | 0.9736 ± 0.0015 |
| 16 | 0.9712 ± 0.0004 |

128 shots per vertex is indistinguishable from exact statistics and even 16 shots costs 0.24 pp — the
practical objection to Case B on hardware (estimating a distribution at every vertex) is mild.

### Do the levers stack?

Not all of them. At chi = 4, G = 2 feature maps were worth +1.85 pp; at chi = 8 they add +0.13 ± 0.15 pp
(0.5303 vs 0.5290) for twice the parameters and 1.6× the step cost — **bond dimension subsumes feature
maps**. Locality and feedforward, by contrast, keep compounding with chi (their gains *grew* from +10.5
and +6.5 pp at chi = 4 to +15.1 and +8.2 pp at chi = 8). Best configuration found:
**patch-local encoder + per-vertex multishot feedforward at chi = 8, G = 1 — CIFAR-10 0.5290 ± 0.0023.**

### ImageNet-32 (1000 classes): the readout is the wall

| model | epochs | test acc (chance 0.10 %) |
|---|---|---|
| chi=4 baseline shared | 15 | 0.48 % |
| chi=4 baseline per-vertex | 15 | 0.54 % |
| chi=4 feedforward B per-vertex | 15 | 0.58 % |
| chi=4 feedforward B shared | 15 | 0.67 % |
| chi=8 baseline shared | 7 | 0.68 % |

Feedforward helps here too (+40 % relative on the matched pair) and chi = 8 matches the best chi = 4
model in half the epochs — but everything sits within an order of magnitude of chance. Reading 1000
classes out of 16 (or 64) root probabilities is the wrong shape of problem: **this ansatz is a
10-class-scale method**, and scaling it needs a wider readout (chi ≥ 32, parallel trees, or a structured
class code), not more tree capacity.

### Beyond chi = 8

The dense Hermitian basis `(K, d, d)` is what caps the bond dimension in practice: 0.13 GB at chi = 8 but
**34 GB at chi = 16**. `params_to_hermitian_scatter` builds the generator by scattering parameters
straight into the matrix instead (identical numerics, selected automatically above chi = 8), which lifts
the ceiling. First CIFAR-10 data point for the shared-unitary patch baseline without feedforward:

| chi | outcomes | TN params | test acc | ms/step (L40S) |
|---|---|---|---|---|
| 4 | 16 | 1,530 | 0.3277 ± 0.0042 | 13 |
| 16 | 256 | 393,210 | 0.4240 (1 seed) | 84 |

and with feedforward B on the same shared-unitary patch model, chi = 16 reaches **0.5644** (10.9 M TN
parameters, 426 ms/step) — **the best CIFAR-10 result in this study**, above chi = 8 per-vertex B
(0.5290) and **+27.8 pp over the published model**. The feedforward gain itself keeps growing with the
register width: **+6.5 pp at chi = 4, +8.2 pp at chi = 8, +14.0 pp at chi = 16**. Bond dimension has not
saturated, and it is the lever that makes the others pay.

## Checkpointing and training for more epochs

Every run writes a `tf.train.Checkpoint` (model weights, AdamW moments and the optimizer step counter that
drives the cosine-restart schedules) after each epoch, next to the metric history:

- locally: `qks-tn-library/runs/ckpt/<run name>/` (disable with `--no-checkpoint`);
- on Modal: `/out/<sweep>/<job>/ckpt/` on the `qks-tn-results` Volume.

`ckpt-<epoch>` is the rolling latest (one kept); in addition the state at the end of every annealing
cycle — epochs 1, 3, 7, 15, 31, 63 — is kept permanently in `cycle-<epoch>/` (checkpoint + history), so
any cycle boundary can be restored with `--resume-from <ckpt_dir>/cycle-015/ckpt` (the local runner) or
compared post hoc.

To continue a run, re-issue it with a larger `--epochs`; the Modal launcher resumes automatically
(`--fresh` retrains from scratch), the local runner needs `--resume`:

```bash
python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C2,C4 --seeds 0,1,2 --epochs 31 --sweep <sweep>
.venv/bin/python qks-tn-library/run_ff_experiment.py --config C2 --task binary --epochs 31 --resume
```

The schedule is `CosineDecayRestarts(first_decay_steps = 1 epoch)` with period doubling, so the warm
restarts fall at epochs 1, 3, 7, 15, 31, …; stop at a cycle boundary (15, 31, 63 epochs) — stopping
mid-cycle leaves the learning rate high. Resumed training reproduces the schedule and the epoch-boundary
state exactly; only the data-shuffle order differs from an uninterrupted run. The `ff-v1` sweep predates
checkpointing, so extending it would retrain those jobs from scratch (31 epochs ≈ 15–30 min per job).

Longer/deeper follow-up (launched 2026-08-25): sweeps `ff-v2-L6` (128 episodes, 6 layers) and `ff-v2-L7`
(`--nepisodes 256`, 7 layers, QKS 200,960 params) with the minimal config set C0, C1, C2, C4 × 3 seeds at
**63 epochs** (the fifth cycle boundary), with cycle checkpoints, to test whether the ff-v1 ordering holds
with longer training and a deeper tree.

## Longer training and deeper trees (sweeps `ff-v2-L6`, `ff-v2-L7`; 63 epochs, 3 seeds)

MNIST test accuracy (3 seeds; columns use the plain-language names: "statistics"/"single-shot" are the
two feedforward variants, "shared"/"per-node" the gate sharing):

| depth / epochs | no ff, shared | no ff, per-node | ff stats, shared | ff stats, per-node | ff single-shot, shared | ff single-shot, per-node |
|---|---|---|---|---|---|---|
| 6 layers, 15 ep (`ff-v1`) | 0.9682 ± 0.0017 | 0.9671 ± 0.0015 | 0.9736 ± 0.0018 | 0.9820 ± 0.0003 | 0.9729 ± 0.0009 | 0.9731 ± 0.0009 |
| 6 layers, 63 ep | 0.9711 ± 0.0014 | 0.9720 ± 0.0015 | 0.9809 ± 0.0009 | 0.9859 ± 0.0003 | — | — |
| 6 layers, 127 ep | 0.9712 ± 0.0014 | 0.9734 ± 0.0011 | 0.9829 ± 0.0014 | 0.9868 ± 0.0003 | 0.9774 ± 0.0002 | 0.9797 ± 0.0005 |
| 7 layers, 63 ep | 0.9782 ± 0.0013 | 0.9784 ± 0.0004 | 0.9841 ± 0.0005 | 0.9871 ± 0.0004 | — | — |
| 7 layers, 127 ep | 0.9786 ± 0.0014 | 0.9789 ± 0.0004 | 0.9851 ± 0.0003 | **0.9881 ± 0.0010** | 0.9811 ± 0.0008 | 0.9834 ± 0.0006 |

![127-epoch curves, per-node gates](img/ff-v2-L6_curves_per_node.png)

*The 127-epoch curves for the per-node-gate models (five annealing cycles; warm restarts visible at
epochs 16, 32 and 64). Lines are means over 3 seeds; bands are 95% confidence intervals (t-interval).
The feedforward models recover from each restart faster and to a lower loss, while the baseline flattens
after epoch ~20 — longer training widens the feedforward gap rather than closing it, and single-shot
feedforward sits cleanly between the two. All three models use per-node gates.*

![127-epoch curves, per-node gates, depth 7](img/ff-v2-L7_curves_per_node.png)

*The same comparison on the 7-layer tree (256 episodes). Multishot feedforward ends at
0.9881 ± 0.0010 vs 0.9789 ± 0.0004 without it; single-shot feedforward again lands between the two at
0.9834 ± 0.0006 (+0.45 ± 0.06 pp over its baseline, McNemar-significant on every seed, yet
0.47 ± 0.16 pp behind multishot).*

Longer training helps the feedforward models far more than the baselines: from 15 to 127 epochs the
baselines gain ~0.3–0.6 pp while statistics feedforward gains ~0.5–0.9 pp, so the paired gap *widens*
(shared: +0.54 pp at 15 ep → +1.17 ± 0.26 pp at 127 ep; per-node: +1.35 ± 0.15 pp at 127 ep, all seeds
positive, McNemar p ≤ 5·10⁻⁸). Single-shot feedforward shows the same pattern at a smaller scale
(depth 6: +0.62 ± 0.13 pp over its baseline at 127 ep; depth 7: +0.45 ± 0.06 pp; per-node beats shared
at both depths) but stays 0.5–0.7 pp behind statistics feedforward on every seed — consistent with the design note's argument that
single-shot conditioning is only a widened unitary, while statistics feedforward is genuinely nonlinear.
A deeper tree lifts everything by ~0.5–1 pp; per-node gates without feedforward remain nearly worthless
(+0.03 ± 0.10 pp at depth 7). The best MNIST number in the dense-encoder family is now
**0.9881 ± 0.0010** (depth 7, statistics feedforward, per-node gates, 127 ep), edging the chi=8
15-epoch record of 0.9883 from the other direction.

Two pipeline bugs were found and fixed on 2026-08-25 while extending to new datasets — both leave the
MNIST results above untouched (60000/32 is exact; MNIST is not class-sorted):
1. *Windowed index shuffle* (250k-row buffer) on the class-sorted HF ImageNet-32 train split ⇒ training
   at chance. Now a full permutation of indices every epoch.
2. *Cosine-schedule drift*: `steps_per_epoch = n // 32` but the ragged last batch was not dropped, so
   CIFAR-10 (1563 vs 1562 steps) and ImageNet-32 drifted one step per epoch and finished 15 steps into a
   restarted cycle at peak learning rate (CIFAR-10 val acc fell 0.32 → 0.29 between epochs 14 and 15).
   Training batches now use `drop_remainder=True`. CIFAR-10 numbers from sweep `cifar10-L5` are
   therefore superseded by `lenet-v1-cifar`.

## Other datasets: CIFAR-10 and ImageNet-32 (5-layer trees)

`--task cifar10` (tf.keras) and `--task imagenet32` (HF `benjamin-paine/imagenet-1k-32x32`: ImageNet-1k
centre-cropped + Lanczos-resampled to 32×32; 1,281,167 train / 50,000 val) are supported with
`--nepisodes 64` (32 leaves, 5 layers). Inputs are the 3072 raw pixel values in [0, 1] (no deskewing);
the kitchen-sink weight scale is `σ = 0.125·√(784/3072) ≈ 0.063` so the encoding angles have the same
variance as on MNIST. CIFAR-10 keeps the paper's fixed 16→10 readout; ImageNet-32 (1000 classes) uses a
trainable linear 16→1000 logits head (17k parameters, identical across configs). The training set is
held in a CPU `tf.Variable` (uint8, 3.9 GB for ImageNet-32) and batches are gathered by shuffled index,
which avoids TensorFlow's 2 GB graph-constant limit. `prepare_data` downloads/decodes each dataset
once into the `qks-tn-data` Volume (ImageNet-32: 164 s).

GPU choice (per-vertex B, 5 layers, 3072-dim inputs, batch 32; one ImageNet-32 job = 600k steps):

| GPU | ms/step | wall/job | $/h | $/job |
|---|---|---|---|---|
| T4 | 22 | 3.7 h | 0.59 | 2.16 |
| L4 | 24 | 4.0 h | 0.80 | 3.20 |
| A10G | 20 | 3.3 h | 1.10 | 3.67 |
| L40S | 17 | 2.8 h | 1.95 | 5.53 |
| A100-40GB | 33 | 5.5 h | 2.10 | 11.6 |

The step is launch-latency-bound (thousands of 16×16 complex kernels), so larger GPUs barely help;
T4 is cheapest overall. Sweeps `cifar10-L5` and `imagenet32-L5` (C0, C1, C2, C4 × seed 0, 15 epochs)
were launched 2026-08-25 on T4s.

## LeNet-style architecture (`qks_tn_lenet.py`, sweeps `lenet-v1-mnist`, `lenet-v1-cifar`)

LeNet's ingredients — local receptive fields, weight sharing across positions, pooling, feature maps,
FC layers — mapped onto the primitives here:

| LeNet | QKS+TTN primitive |
|---|---|
| conv filter on a patch, shared across positions | `PatchQKSEncodingLayer`: two kitchen-sink filters applied to every `patch×patch` window (a `conv2d`); the two angles form one χ=4 leaf register per position. MNIST 7×7/stride 3 → 8×8 positions, **100 encoder parameters** (vs 100,480 dense); CIFAR 4×4×3/stride 4 → 8×8, 98 parameters |
| feature maps | `groups=G` registers per position (leaves position-major, group-minor: the first log₂G merges are 1×1 channel mixing) |
| conv + 2×2 pool | a TTN vertex (unitary + dissipative trace) on leaves ordered along a **Z/Morton curve**, so successive merges alternate x and y and every two layers contract a 2×2 block (quadtree) |
| nonlinearity between stages | feedforward mode B |
| FC → classes | trainable linear head (`--head linear`) or the paper's fixed readout |

Controls: `R*` = same weights with raster leaf order (isolates 2-D locality in the tree; fully paired
with `L*` — identical initialisation, permutation only); `U*` = locally connected filters (one filter bank
per position, 6,400 params on MNIST; isolates weight sharing); dense `C*` (isolates locality).
σ for the encoder follows the dimensional rule σ = 0.125·√(784/(patch²·C)) (0.50 on MNIST, 0.51 on
CIFAR); `--sigma-scale` exposes a multiplier. Tests: `tests/test_lenet.py` (patch encoder ≡ dense
encoder with a patch-sparse position-shared `W`; Morton quadtree structure; parameter counts; seed pairing).
Expectation stated up front: with a 100-parameter encoder the MNIST models may lose to the 100k-parameter
dense QKS; the informative comparisons are L4 vs R4 (locality in the tree), L4 vs U4 (sharing) and the
CIFAR-10 numbers, where the dense encoder is weak (~30 %).

### Results (15 epochs, 3 seeds; CIFAR-10 dense controls re-run with the fixed schedule)

| config | model | TN params | trainable | CIFAR-10 test acc | MNIST test acc |
|---|---|---|---|---|---|
| C0 | dense, baseline (shared) — the paper's model | 1,530 | 394,874 | 0.2862 ± 0.0089 | 0.9682 ± 0.0017* |
| C1 | dense, baseline (per-vertex) | 16,065 | 409,409 | 0.3387 ± 0.0032 | 0.9671 ± 0.0015* |
| C4 | dense, B (per-vertex) | 79,305 | 472,649 | 0.3501 ± 0.0084 | 0.9820 ± 0.0003* |
| L0 | patch, Morton, baseline (shared) | 1,530 | **1,628** | 0.3277 ± 0.0042 | 0.8057 ± 0.0040 |
| L1 | patch, Morton, baseline (per-vertex) | 16,065 | 16,163 | 0.3906 ± 0.0027 | 0.8809 ± 0.0096 |
| L2 | patch, Morton, B (shared) | 11,730 | 11,828 | 0.4022 ± 0.0043 | 0.9583 ± 0.0004 |
| **L4** | **patch, Morton, B (per-vertex)** | 79,305 | 79,403 | **0.4552 ± 0.0042** | **0.9759 ± 0.0027** |
| R4 | patch, raster, B (per-vertex) | 79,305 | 79,403 | 0.4512 ± 0.0029 | 0.9745 ± 0.0019 |
| U4 | locally connected, B (per-vertex) | 79,305 | 85,577 | 0.4630 ± 0.0002 | 0.9766 ± 0.0004 |
| L4lin | L4 with a trainable linear head | 79,305 | 79,573 | 0.3702 ± 0.0081 | 0.9476 ± 0.0295 |

\* MNIST dense numbers from sweep `ff-v1` (same protocol, 6-layer tree).

Paired per-seed differences (all "+++"/"−−−" = same sign on every seed):

| comparison | isolates | CIFAR-10 Δ acc | MNIST Δ acc |
|---|---|---|---|
| L4 − C4 | the whole LeNet-style change vs dense | **+10.52 ± 0.71 pp** (p ≤ 3e−79) | −0.6 pp |
| U4 − C4 | locality alone (unshared filters) | **+11.29 ± 0.86 pp** | — |
| L4 − L1 | feedforward B with a patch encoder | **+6.47 ± 0.68 pp** | **+9.50 ± 0.93 pp** |
| L2 − L0 | feedforward B, shared unitaries | +7.45 ± 0.58 pp | **+15.27 ± 0.41 pp** |
| L4 − R4 | Morton (quadtree) vs raster leaf order | +0.41 ± 0.14 pp (McNemar n.s.) | +0.14 ± 0.21 pp |
| L4 − U4 | position-shared vs per-position filters | −0.78 ± 0.44 pp | −0.07 ± 0.29 pp |
| L4lin − L4 | trainable linear head vs fixed readout | **−8.51 ± 0.44 pp** | −2.83 ± 2.68 pp |

![CIFAR-10 patch encoder](img/lenet-v1-cifar_curves.png)
![MNIST patch encoder](img/lenet-v1-mnist_curves.png)

**Reading.**
1. **Locality is the dominant factor on CIFAR-10**: 0.286 → 0.455 (+16.9 pp over the paper's model,
   +10.5 pp over dense per-vertex B) with **6× fewer trainable parameters** (79k vs 473k), because the
   98-parameter shared encoder replaces 196,672 dense QKS weights.
2. **Feedforward and locality are complementary and roughly additive**: +11 pp from locality, +6.5 pp
   from feedforward, both positive on every seed.
3. **The quadtree ordering does not matter** (L4 − R4 ≈ +0.4 pp on CIFAR, +0.1 pp on MNIST, McNemar
   n.s.): what helps is that each leaf sees a *local patch*, not that siblings are spatial neighbours.
4. **Weight sharing is free**: 98 shared filter parameters match 6,272 per-position ones to within
   ±0.8 pp on both datasets (U4 is marginally ahead on CIFAR).
5. **On MNIST the patch model nearly matches the dense one** (0.9759 vs 0.9820) with a 100-parameter
   encoder instead of 100,480 — and feedforward is worth far more there (+9.5 to +15.3 pp) than with
   the dense encoder (+0.5 pp), the same pattern as the frozen-QKS experiment: mode B carries the model
   when the features are weak.
6. **The fixed 16→10 readout beats a trainable linear head** (−8.5 pp on CIFAR). Capacity is not the
   constraint at the readout; keeping the outputs as normalised probabilities is.

### Follow-ups (`lenet-v2-*`, CIFAR-10, 3 seeds)

| change | L1 (no feedforward) | L4 (feedforward B) |
|---|---|---|
| baseline: 4×4/stride 4, G=1 | 0.3906 ± 0.0027 | 0.4552 ± 0.0042 |
| overlapping 11×11/stride 3 | 0.3835 ± 0.0024 | 0.4496 ± 0.0023 (**null**) |
| **G = 2 feature maps** (128 leaves, 7 layers) | 0.4043 ± 0.0015 | **0.4737 ± 0.0013** (+1.85 pp) |
| G = 4 feature maps (256 leaves, 8 layers) | 0.4093 ± 0.0039 | 0.4798 ± 0.0018 (+0.61 pp more) |

A bigger receptive field does nothing; a second register per position helps on every seed. Together with
the weight-sharing and leaf-ordering nulls this says the model is limited by *how much quantum state it
carries* (registers per position, bond dimension), not by how it reads the image.

## Not yet done

- Finite-shot noise (`--shots 16/128/1024`, configs `C2s*`), `--ff-layers` ablations (root only /
  top two layers), centred features (`C2c`) — all implemented and runnable, not yet swept.
- More seeds for C2 vs C3 (cheap: ~15 GPU-minutes per job).
- Binary (3 vs 5) is at ceiling (4–5 errors of 1,902 for every model) and cannot discriminate.

## Reproducing

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "tensorflow==2.15.*" tensorflow-addons==0.23.0 tensornetwork==0.4.6 scipy matplotlib
.venv/bin/python qks-tn-library/tests/test_ff_equivalence.py
.venv/bin/python qks-tn-library/run_ff_experiment.py --config C2 --task binary --epochs 15 --seed 0
python3 -m modal run qks-tn-library/modal_ff_sweep.py --configs C0,C1,C2,C3,C4,C5,F0,F2,F5 --seeds 0,1,2 --sweep ff-v1
.venv/bin/python qks-tn-library/analyze_ff_runs.py --sweep ff-v1
```
