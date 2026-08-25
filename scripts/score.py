#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score.py -- one command, every number, held-out, with confidence intervals.

Scores a trained checkpoint against the HELD-OUT split in ONE declared
coordinate, and reports the hardware cost of the graph it was trained on.

Exists because the scoring was scattered: train.py scored in-sample (fixed),
certify.py reported a subset, tail asymmetry lived in a separate script, and
the resource estimate lived in the deployment script.  Comparing two
checkpoints meant running four things and reconciling coordinates by hand.

    python scripts/score.py run_heron6
    python scripts/score.py run_heron6 run_native4 run_converged   # compare
"""
from __future__ import annotations
import argparse, json, math, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qpce import QPCE, QPCEConfig, load_dataset, make_train_test
from qpce.data import Standardiser
from qpce.energy import energy_distance
from qpce.certify import independence_floor
from qpce.metrics import spearman_matrix
from qpce.config import edge_colour, fidelity_estimate


def rank_summary(Y_model, Y_data):
    """Mean absolute rank correlation of model and data, and their distance."""
    import numpy as _np
    Sm, Sd = spearman_matrix(Y_model), spearman_matrix(Y_data)
    iu = _np.triu_indices(Y_model.shape[1], 1)
    return {"mean_abs_rho_model": float(_np.abs(Sm[iu]).mean()),
            "mean_abs_rho_data": float(_np.abs(Sd[iu]).mean()),
            "frobenius": float(_np.linalg.norm(Sm - Sd))}

QS = (0.25, 0.10, 0.05, 0.02, 0.01)


def _rp(p):
    return p if os.path.exists(p) else os.path.join(_ROOT, p)


def _outdir(sub, explicit=None):
    """Where results go.

    Everything lands under <project root>/outputs/ unless
    overridden with --outdir.  Before this, each script wrote to the
    CURRENT WORKING DIRECTORY, so the same command produced files in
    different places depending on where it was invoked.
    """
    d = explicit or os.path.join(_ROOT, "outputs", sub)
    os.makedirs(d, exist_ok=True)
    return d


def lam_q(Y, q, upper=False, U=None):
    U = np.argsort(np.argsort(Y, 0), 0) / (len(Y) + 1.0) if U is None else U
    iu = np.triu_indices(Y.shape[1], 1)
    ind = ((U > 1 - q) if upper else (U < q)).astype(float)
    return float(np.mean([(ind[:, i] * ind[:, j]).mean() / q
                          for i, j in zip(*iu)]))


def w1_per_pixel(A, B):
    n = min(len(A), len(B))
    qs = np.linspace(0, 1, 512)
    return np.array([np.abs(np.quantile(A[:, j], qs) - np.quantile(B[:, j], qs)).mean()
                     for j in range(A.shape[1])])


def score_one(run, Ytr, Yte, std, F, n_gen, n_boot, seed, verbose=True):
    ck = np.load(os.path.join(run, "params.npz"), allow_pickle=True)
    mdl = QPCE.load(os.path.join(run, "params.npz"), QPCEConfig())
    edges = [tuple(sorted(map(int, e))) for e in ck["edges"]]
    L, n = ck["beta"].shape
    Yg = mdl.generate(n_gen, seed=seed)

    ns = min(len(Yte), 1400)
    E = energy_distance(std(Yg[:ns]), std(Yte[:ns]))
    rk = rank_summary(Yg, Yte)
    # CI by SUBSAMPLING WITHOUT REPLACEMENT, not by bootstrap.
    # The energy distance is a U-statistic; resampling with replacement puts
    # duplicate points into it, which inflates the pairwise-distance terms and
    # biases D downward -- badly enough that the point estimate can fall
    # OUTSIDE its own "95 % CI".  Subsample at m = ns/2 without replacement and
    # rescale the half-width by sqrt(m/ns), since Var[U] ~ c/n.
    rng = np.random.default_rng(0)
    mm = ns // 2
    bD, bR = [], []
    for _ in range(n_boot):
        i = rng.permutation(len(Yg))[:mm]
        k = rng.permutation(ns)[:mm]
        bD.append(1 - energy_distance(std(Yg[i]), std(Yte[k])) / F)
        j = rng.permutation(len(Yg))[:len(Yg) // 2]
        bR.append(rank_summary(Yg[j], Yte)["mean_abs_rho_model"])
    sc = math.sqrt(mm / ns)

    U = np.argsort(np.argsort(Yg, 0), 0) / (len(Yg) + 1.0)
    tails = {q: (lam_q(Yg, q, False, U), lam_q(Yg, q, True, U)) for q in QS}
    chi = edge_colour(edges)[0]
    # Is this graph a subgraph of any real device?  If not, L*|E| pulses and
    # L*chi' layers are a FICTION: the circuit must be routed, and routing
    # inflated the published C8(1,2) from 16 layers to 98.
    from qpce import COUPLER_GRAPHS
    native = next((g for g in ("grid2x4", "ladder", "ring8", "path8")
                   if sorted(COUPLER_GRAPHS[g](n)) == sorted(edges)), None)
    tq = L * len(edges)
    Fq = fidelity_estimate(tq, 6.2e-3, n=n, one_q=(L + 1) * n)

    return {"run": run, "L": L, "n": n, "edges": len(edges),
            "native_graph": native,
            "chi_prime": chi, "two_qubit_pulses": tq,
            "two_qubit_layers": L * chi,
            "duration_us": (L * chi * 68e-9 + (L + 1) * 32e-9) * 1e6,
            "F_pred_6.2e-3": Fq,
            "shots_required": int(math.ceil(512 / Fq ** 2 / 64) * 64),
            "energy": E, "D": 1 - E / F,
            "D_ci": [float(1 - E / F - sc * (1 - E / F - np.percentile(bD, 2.5))),
                     float(1 - E / F + sc * (np.percentile(bD, 97.5) - (1 - E / F)))],
            "mean_abs_rho": rk["mean_abs_rho_model"],
            "rho_ci": [float(rk["mean_abs_rho_model"]
                             - math.sqrt(0.5) * (rk["mean_abs_rho_model"] - np.percentile(bR, 2.5))),
                       float(rk["mean_abs_rho_model"]
                             + math.sqrt(0.5) * (np.percentile(bR, 97.5) - rk["mean_abs_rho_model"]))],
            "mean_abs_rho_data": rk["mean_abs_rho_data"],
            "frobenius": rk["frobenius"],
            "lambda_L": {str(q): tails[q][0] for q in QS},
            "lambda_U": {str(q): tails[q][1] for q in QS},
            "Lambda_0.05": tails[0.05][0] - tails[0.05][1],
            "w1_mean": float(w1_per_pixel(Yg, Yte).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="data/cal_shower_img_8q.npy")
    ap.add_argument("--n-gen", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--outdir", default=None,
                    help="directory for results "
                         "(default: <root>/outputs/)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    _od = _outdir("score", a.outdir)
    a.out = a.out or os.path.join(_od, "score.json")
    a.data = _rp(a.data)

    X, _ = load_dataset(a.data, 4000, 7)
    sp = make_train_test(X, 0.35, 7)
    Ytr, Yte = sp["Y_train"], sp["Y_test"]
    std = Standardiser(Ytr)
    F, Fsd = independence_floor(Ytr, std, seed=7)
    print(f"[coordinate] z-scored;  floor F = {F:.5f} +- {Fsd:.5f}  "
          f"(from {len(Ytr)} train rows)")
    print(f"[reference]  scored against {len(Yte)} HELD-OUT rows;  "
          f"data mean|rho_S| = {rank_summary(Yte, Yte)['mean_abs_rho_data']:.4f}")

    res = [score_one(_rp(r), Ytr, Yte, std, F, a.n_gen, a.n_boot, a.seed)
           for r in a.runs]

    print(f"\n{'run':<20}{'graph':>7}{'L':>3}{'2q':>5}{'lyr':>5}{'us':>7}"
          f"{'D':>8}{'95% CI':>18}{'mean|rho|':>11}{'95% CI':>16}{'Frob':>7}")
    for r in res:
        print(f"{os.path.basename(r['run']):<20}{r['edges']:>4}e{r['chi_prime']:>2}c"
              f"{r['L']:>3}{r['two_qubit_pulses']:>5}{r['two_qubit_layers']:>5}"
              f"{r['duration_us'] if r['native_graph'] else float('nan'):>7.2f}"
              f"{r['D']:>8.4f}"
              f"  [{r['D_ci'][0]:+.3f},{r['D_ci'][1]:+.3f}]"
              f"{r['mean_abs_rho']:>11.4f}"
              f"  [{r['rho_ci'][0]:.3f},{r['rho_ci'][1]:.3f}]{r['frobenius']:>7.3f}")

    print(f"\n{'run':<20}{'W1':>9}{'Lam(.05)':>10}   lambda_L(q) / lambda_U(q)")
    for r in res:
        tl = "  ".join(f"{q}:{r['lambda_L'][str(q)]:.3f}/{r['lambda_U'][str(q)]:.3f}"
                       for q in QS)
        print(f"{os.path.basename(r['run']):<20}{r['w1_mean']:>9.5f}"
              f"{r['Lambda_0.05']:>+10.4f}   {tl}")

    print(f"\n{'run':<20}{'native?':>10}{'F@6.2e-3':>10}{'shots req':>11}"
          f"{'total (5k germs)':>19}")
    for r in res:
        nat = r["native_graph"] or "NO - routed"
        print(f"{os.path.basename(r['run']):<20}{nat:>10}{r['F_pred_6.2e-3']:>10.3f}"
              f"{r['shots_required']:>11,}{5000 * r['shots_required']:>19,}")
    if any(r["native_graph"] is None for r in res):
        print("  ! runs marked 'NO - routed' are NOT a subgraph of any device. Their")
        print("    pulse/layer/shot numbers assume zero SWAPs and are a FICTION:")
        print("    measured, routing takes C8(1,2) from 16 layers to 98 and its")
        print("    shot budget from 7.0M to 166M. ")

    json.dump({"floor": F, "floor_sd": Fsd, "runs": res},
              open(a.out, "w"), indent=2, default=float)
    print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
