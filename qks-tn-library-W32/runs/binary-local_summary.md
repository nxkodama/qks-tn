# binary-local: 4 runs, task(s) binary

## Per configuration

| config | model | seeds | TN params | trainable | final test acc | final test CE | best val acc (epoch) | ms/step | min/run |
|---|---|---|---|---|---|---|---|---|---|
| C0 | baseline (shared) | 1 | 1,530 | 102,010 | 0.9974 | 0.0154 | 0.9974 (14.0) | 15 | 1.5 |
| C2 | B: multishot feedforward | 1 | 11,730 | 112,210 | 0.9974 | 0.0127 | 0.9979 (12.0) | 52 | 5.2 |
| C5 | A: single-shot feedforward | 1 | 21,930 | 122,410 | 0.9974 | 0.0161 | 0.9974 (14.0) | 64 | 6.0 |
| C0old | baseline, original TNLayer | 1 | 16,065 | 116,545 | 0.9979 | 0.0141 | 0.9979 (14.0) | 86 | 11.2 |

## Paired comparisons

| candidate | control | isolates | seeds | Δ test acc (pp) | Δ test CE | sign | McNemar p per seed |
|---|---|---|---|---|---|---|---|
| C2 | C0 | multishot feedforward (B) vs baseline | 1 | 0.00 | -0.0026 | 0 | s0: p=1 (1/1) |
| C5 | C0 | single-shot feedforward (A) vs baseline | 1 | 0.00 | 0.0007 | 0 | s0: p=1 (1/1) |
| C0 | C0old | new sweep vs original TNLayer implementation (regression) | 1 | -0.05 | 0.0013 | - | s0: p=1 (1/0) |

Δ = candidate − control, paired by seed. McNemar counts are (errors only in candidate / errors only in control).

## Figures

- `runs/binary-local_curves_trainable_QKS.png`
