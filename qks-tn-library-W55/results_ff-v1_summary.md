# ff-v1: 30 runs, task(s) multiclass

## Per configuration

| config | model | seeds | TN params | trainable | final test acc | final test CE | best val acc (epoch) | ms/step | min/run |
|---|---|---|---|---|---|---|---|---|---|
| C0 | baseline (shared) | 3 | 1,530 | 102,010 | 0.9682 ± 0.0017 | 0.1511 ± 0.0071 | 0.9684 ± 0.0017 (13.7) | 14 | 8.2 |
| F0 | baseline (shared) | 3 | 1,530 | 1,530 | 0.6409 ± 0.0255 | 1.4004 ± 0.0219 | 0.6421 ± 0.0249 (13.7) | 16 | 9.3 |
| C1 | baseline (per-vertex) | 3 | 16,065 | 116,545 | 0.9671 ± 0.0015 | 0.1438 ± 0.0044 | 0.9672 ± 0.0015 (14.0) | 16 | 9.5 |
| C2 | B: multishot feedforward | 3 | 11,730 | 112,210 | 0.9736 ± 0.0018 | 0.1137 ± 0.0099 | 0.9736 ± 0.0018 (15.0) | 27 | 15.1 |
| F2 | B: multishot feedforward | 3 | 11,730 | 11,730 | 0.8677 ± 0.0039 | 0.6824 ± 0.0213 | 0.8678 ± 0.0037 (14.7) | 32 | 17.8 |
| C3 | B, constant p (null) | 3 | 11,730 | 112,210 | 0.9672 ± 0.0018 | 0.1540 ± 0.0080 | 0.9674 ± 0.0017 (14.7) | 24 | 13.1 |
| C4 | B (per-vertex) | 3 | 79,305 | 179,785 | 0.9820 ± 0.0003 | 0.0683 ± 0.0027 | 0.9820 ± 0.0003 (15.0) | 29 | 15.9 |
| C5 | A: single-shot feedforward | 3 | 21,930 | 122,410 | 0.9729 ± 0.0009 | 0.1193 ± 0.0017 | 0.9731 ± 0.0008 (14.0) | 19 | 11.0 |
| F5 | A: single-shot feedforward | 3 | 21,930 | 21,930 | 0.7885 ± 0.0114 | 1.0994 ± 0.0140 | 0.7890 ± 0.0116 (13.7) | 21 | 12.2 |
| C6 | A (per-vertex) | 3 | 142,545 | 243,025 | 0.9731 ± 0.0009 | 0.1117 ± 0.0029 | 0.9732 ± 0.0009 (14.0) | 19 | 11.8 |

## Paired comparisons

| candidate | control | isolates | seeds | Δ test acc (pp) | Δ test CE | sign | McNemar p per seed |
|---|---|---|---|---|---|---|---|
| C1 | C0 | per-vertex unitaries vs shared (more parameters, no feedforward) | 3 | -0.11 ± 0.05 | -0.0073 ± 0.0028 | --- | s0: p=0.478 (126/114); s1: p=0.946 (108/106); s2: p=0.19 (104/85) |
| C2 | C0 | multishot feedforward (B) vs baseline | 3 | 0.53 ± 0.29 | -0.0374 ± 0.0139 | ++- | s0: p=6.01e-06 (80/149); s1: p=3.35e-10 (64/157); s2: p=0.935 (77/75) |
| C3 | C0 | information-free B (constant p) vs baseline: reparametrisation only | 3 | -0.10 ± 0.16 | 0.0030 ± 0.0040 | +-- | s0: p=0.204 (91/110); s1: p=0.0109 (125/87); s2: p=0.442 (90/79) |
| C2 | C3 | B vs its information-free twin: value of the fed-forward information | 3 | 0.63 ± 0.36 | -0.0403 ± 0.0168 | +++ | s0: p=0.000436 (73/123); s1: p=4.46e-20 (42/173); s2: p=0.439 (49/58) |
| C4 | C1 | per-vertex B vs per-vertex baseline | 3 | 1.48 ± 0.12 | -0.0754 ± 0.0027 | +++ | s0: p=1.02e-27 (47/219); s1: p=5.29e-24 (36/179); s2: p=4.54e-20 (41/171) |
| C5 | C0 | single-shot feedforward (A) vs baseline | 3 | 0.47 ± 0.24 | -0.0317 ± 0.0082 | +++ | s0: p=5.16e-11 (52/143); s1: p=0.00159 (64/106); s2: p=0.65 (84/91) |
| C5 | C1 | A vs parameter-matched per-vertex baseline | 3 | 0.58 ± 0.23 | -0.0244 ± 0.0055 | +++ | s0: p=1.35e-11 (66/169); s1: p=0.00291 (83/127); s2: p=0.0872 (94/120) |
| C6 | C1 | per-vertex A vs per-vertex baseline | 3 | 0.60 ± 0.11 | -0.0321 ± 0.0029 | +++ | s0: p=1.27e-08 (61/142); s1: p=0.000736 (60/104); s2: p=3.75e-05 (57/111) |
| C6 | C4 | per-vertex A vs per-vertex B | 3 | -0.89 ± 0.07 | 0.0434 ± 0.0004 | --- | s0: p=8.68e-13 (129/38); s1: p=3.59e-14 (138/39); s2: p=2.46e-09 (120/44) |
| F2 | F0 | B vs baseline with frozen QKS (TN-only learning) | 3 | 22.67 ± 2.40 | -0.7180 ± 0.0302 | +++ | s0: p=1.06e-300 (398/2199); s1: p=0 (233/2632); s2: p=0 (300/2902) |
| F5 | F0 | A vs baseline with frozen QKS (TN-only learning) | 3 | 14.76 ± 2.66 | -0.3011 ± 0.0271 | +++ | s0: p=3.76e-79 (813/1758); s1: p=3.9e-295 (387/2150); s2: p=1.44e-260 (484/2203) |

Δ = candidate − control, paired by seed. McNemar counts are (errors only in candidate / errors only in control).

## Figures

- `results/ff-v1/ff-v1_curves_trainable_QKS.png`
- `results/ff-v1/ff-v1_curves_frozen_QKS.png`
