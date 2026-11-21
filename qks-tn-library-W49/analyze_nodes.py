"""W48: Kraus spectrum, discarded-register entropy and truncation test of one trained CPTP model.

Reads a checkpoint written by ``run_ff_experiment.train`` (``<ckpt_dir>/cfg.json`` and ``<ckpt_dir>/<ckpt_name>.*``) and
returns a JSON-serializable dict.  Meant to run on a GPU on Modal (``modal_w48_analysis.py``); the unit tests of the
pieces are in ``tests/test_nodes.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common_functions"))
sys.path.insert(0, HERE)

NL_NONROOT = 5


def passes_plan(nl=NL_NONROOT):
    """Truncation passes: (name, r per non-root layer, variant, renorm)."""
    allr = lambda r: [r] * nl  # noqa: E731
    out = [("full-replica", [None] * nl, "choi", "tp")]
    for r in (16, 15, 14, 12, 8, 4, 2, 1):
        out.append((f"all-choi-tp-r{r}", allr(r), "choi", "tp"))
    for r in (8, 4, 2, 1):
        out.append((f"all-choi-trace-r{r}", allr(r), "choi", "trace"))
    for r in (8, 4, 2, 1):
        out.append((f"all-norm-tp-r{r}", allr(r), "norm", "tp"))
    for r in (1, 8):
        for l in range(nl):
            rl = [None] * nl
            rl[l] = r
            out.append((f"layer{l}-choi-tp-r{r}", rl, "choi", "tp"))
    return out


def passes_plan_w49(rs=(15, 14, 12), ndraw=5, layer_r=15, layer_draws=5, layer_rdraws_variants=("haar", "subset"),
                    r_draws=None, extras=True, root=True, check_choi=True, nl=NL_NONROOT):
    """W49 truncation passes: (name, r per non-root layer, variant, renorm, extra) with extra = dict(draw=..., root_r=...).

    ``r_draws`` optionally maps r to a number of draws (default ``ndraw`` for every r).
    """
    allr = lambda r: [r] * nl  # noqa: E731
    r_draws = {int(k): int(v) for k, v in (r_draws or {}).items()}
    out = [("full-replica", [None] * nl, "choi", "tp", {})]
    if check_choi:
        out.append((f"all-choi-tp-r{rs[0]}", allr(rs[0]), "choi", "tp", {}))
    for r in rs:
        out.append((f"all-diagp-tp-r{r}", allr(r), "diagp", "tp", {}))
        if extras:
            out.append((f"all-norm-tp-r{r}", allr(r), "norm", "tp", {}))
        for d in range(r_draws.get(r, ndraw)):
            out.append((f"all-haar-d{d}-r{r}", allr(r), "haar", "tp", dict(draw=("haar", d))))
            out.append((f"all-subset-d{d}-r{r}", allr(r), "subset", "tp", dict(draw=("subset", d))))
    for l in range(nl):
        rl = [None] * nl
        rl[l] = layer_r
        out.append((f"layer{l}-choi-tp-r{layer_r}", rl, "choi", "tp", {}))
        out.append((f"layer{l}-diagp-tp-r{layer_r}", rl, "diagp", "tp", {}))
        for v in layer_rdraws_variants:
            for d in range(layer_draws):
                out.append((f"layer{l}-{v}-d{d}-r{layer_r}", rl, v, "tp", dict(draw=(v, d))))
    if root:
        rl = [None] * nl
        for v, nd_ in (("choi", 1), ("haar", layer_draws), ("subset", layer_draws)):
            for d in range(nd_):
                out.append((f"root-{v}" + (f"-d{d}" if v != "choi" else "") + "-r7", rl, v, "tp",
                            dict(draw=(v, d) if v != "choi" else None, root_r=7)))
    return out


def make_draw(spec, R):
    """Draw object for a plan row: ("haar", d) -> Haar unitary R x R; ("subset", d) -> permutation of range(R)."""
    if spec is None:
        return None
    import cptp_nodes as nd
    kind, d = spec
    if kind == "haar":
        return nd.haar_unitary(R, 1000 + d)
    return np.random.default_rng(2000 + d).permutation(R)


def _accumulate(acc, stats, n_batch):
    for l, st in enumerate(stats):
        a = acc.setdefault(l, {})
        for k, v in st.items():
            s = v.numpy().sum(axis=0)  # (n, ...) summed over the batch
            a[k] = a.get(k, 0) + s
            if k in ("env_entropy", "eff_rank_sv", "eff_rank_choi", "choi_tail", "dist_init"):
                a[k + "_sq"] = a.get(k + "_sq", 0) + (v.numpy() ** 2).sum(axis=0)
    acc["_n"] = acc.get("_n", 0) + n_batch


def _summarize(acc):
    n = acc["_n"]
    out = {}
    for l, a in acc.items():
        if l == "_n":
            continue
        layer = {}
        for k, v in a.items():
            if k.endswith("_sq"):
                continue
            mean = v / n  # (nverts, ...)
            layer[k + "_per_vertex"] = np.asarray(mean).tolist()
            if mean.ndim == 1:
                layer[k] = float(mean.mean())
                layer[k + "_vertex_min"] = float(mean.min())
                layer[k + "_vertex_max"] = float(mean.max())
                if k + "_sq" in a:
                    var = a[k + "_sq"] / n - mean**2
                    layer[k + "_sample_std"] = float(np.sqrt(np.maximum(var, 0).mean()))
            else:  # spectra (nverts, R)
                layer[k] = np.asarray(mean.mean(axis=0)).tolist()
        out[str(l)] = layer
    return out


def analyze(ckpt_dir, ckpt_name, saved_result=None, n_spec=1024, bs=128, limit=None, cache_dir="/data",
            plan=None, log=print, spectrum=True, gpu_eigh=False):
    import tensorflow as tf

    import cptp_nodes as nd
    import run_ff_experiment as rfe

    t0 = time.time()
    import cptp_nodes as _nd0
    _nd0.GPU_EIGH = bool(gpu_eigh)
    tf.get_logger().setLevel("ERROR")
    cfg = json.load(open(os.path.join(ckpt_dir, "cfg.json")))
    assert cfg["task"] == "cifar10" and cfg.get("holdout_5000") and cfg["kraus_rank"] and cfg["mode"] == "B"
    X_train, y_train, X_test, y_test = rfe.load_data(cfg["task"], cache_dir, log=log)
    X_train, y_train, X_val, y_val, _ = rfe.split_cifar10_holdout(X_train, y_train)
    steps = len(y_train) // int(cfg["batch_size"])
    p = int(X_train.shape[1])
    rng = np.random.default_rng(0)
    spec_idx = np.sort(rng.permutation(len(y_train))[: int(n_spec)])
    if limit:
        X_val, y_val, X_test, y_test = X_val[:limit], y_val[:limit], X_test[:limit], y_test[:limit]

    # seed-matched initialization: a second build with the same cfg gives the same initial parameters
    model0, qks0, tn0, _ = rfe.build_model(cfg, steps, p=p, n_classes=10)
    model, qks, tn, counts = rfe.build_model(cfg, steps, p=p, n_classes=10)
    ck = tf.train.Checkpoint(model=model)
    ck.restore(os.path.join(ckpt_dir, ckpt_name)).expect_partial()
    head = model.layers[-1]
    log(f"restored {ckpt_dir}/{ckpt_name}; params {counts['tn']}")

    def encode(X, y, enc):
        ds = rfe.make_dataset(X, y, bs, shuffle=False, seed=0)
        rhos, ys = [], []
        for xb, yb in ds:
            rhos.append(enc(xb))
            ys.append(yb.numpy().astype(int))
        return rhos, np.concatenate(ys)

    res = dict(cfg=cfg, ckpt=f"{ckpt_dir}/{ckpt_name}", n_spec=int(n_spec), limit=limit, params_tn=counts["tn"])

    # ---- Kraus spectrum and entropies (trained and initialization) on training images
    acc_t, acc_0 = {}, {}
    if spectrum:
        rho_spec_t, _ = encode(X_train[spec_idx], y_train[spec_idx], qks)
        rho_spec_0, _ = encode(X_train[spec_idx], y_train[spec_idx], qks0)
        for r_t, r_0 in zip(rho_spec_t, rho_spec_0):
            _, st = nd.forward_nodes(tn, r_t, collect=True, init_layer=tn0)
            _accumulate(acc_t, st, int(r_t.shape[0]))
            _, st0 = nd.forward_nodes(tn0, r_0, collect=True, init_layer=tn0)
            _accumulate(acc_0, st0, int(r_0.shape[0]))
        res["spectrum_trained"] = _summarize(acc_t)
        res["spectrum_init"] = _summarize(acc_0)
    res["t_spectrum_s"] = time.time() - t0
    log(f"spectrum done {time.time() - t0:.0f}s")

    # ---- reference accuracy: Keras model and replica forward pass
    sets = {}
    for name, (X, y) in dict(holdout=(X_val, y_val), test=(X_test, y_test)).items():
        rho, ys = encode(X, y, qks)
        sets[name] = (rho, ys)
    ds_v = rfe.make_dataset(X_val, y_val, bs, shuffle=False, seed=0)
    ds_t = rfe.make_dataset(X_test, y_test, bs, shuffle=False, seed=0)
    res["keras"] = dict(holdout=float(model.evaluate(ds_v, verbose=0)[1]), test=float(model.evaluate(ds_t, verbose=0)[1]))
    if saved_result:
        res["saved"] = dict(holdout=saved_result["final_val"]["acc"], test=saved_result["final_test"]["acc"])

    def run_pass(trunc):
        accs, info = {}, {}
        for name, (rho, ys) in sets.items():
            pred = []
            for r in rho:
                o = nd.forward_nodes(tn, r, trunc=trunc, info=info)
                pred.append(np.argmax(head(o).numpy(), -1))
            accs[name] = float((np.concatenate(pred) == ys).mean())
        return accs, info.get("fallback", {})

    res["passes"] = []
    for row_in in (plan or passes_plan()):
        name, rl, variant, renorm = row_in[:4]
        extra = row_in[4] if len(row_in) > 4 else {}
        t1 = time.time()
        trunc = None if (all(x is None for x in rl) and not extra.get("root_r")) else dict(r=rl, variant=variant, renorm=renorm)
        if trunc is not None:
            Rdraw = int(cfg["kraus_rank"]) if not extra.get("root_r") else int(cfg["chi"])
            if extra.get("draw") is not None:
                trunc["draw"] = make_draw(extra["draw"], Rdraw)
            if extra.get("root_r"):
                trunc["root_r"] = int(extra["root_r"])
        accs, fb = run_pass(trunc)
        row = dict(name=name, r=rl, variant=variant, renorm=renorm, extra={k: v for k, v in extra.items()},
                   holdout=accs["holdout"], test=accs["test"],
                   fallback={str(k): int(v) for k, v in fb.items()}, seconds=time.time() - t1)
        res["passes"].append(row)
        log(f"{name}: holdout {accs['holdout']:.4f} test {accs['test']:.4f} fallback {row['fallback']} ({row['seconds']:.0f}s)")
    res["wall_s"] = time.time() - t0
    return res
