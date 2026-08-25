#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_numbers.py -- every number quoted in the manuscript, computed from the
checkout, written to paper_numbers.json and numbers.tex.

One coordinate (z-scored), one split (65/35, seed 7), held-out scoring only.
Model = the DEPLOYED checkpoint outputs/checkpoint (path8, L=6, K=1 shared germ).
Dense-graph reference = run_converged (C8(1,2), L=4), quoted only as the
undeployable ceiling.

    python scripts/paper_numbers.py
"""
from __future__ import annotations
import json, math, os, sys

import numpy as np
from scipy.stats import spearmanr, kendalltau

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

def rank_summary(Y_model, Y_data):
    """Spearman summary shared by every rank-based number in the manuscript.

    Returns the mean absolute off-diagonal rank correlation of the model and of
    the data, and the Frobenius distance between the two correlation matrices.
    Defined here so that this script has no dependency outside the package.
    """
    Sm, Sd = spearman_matrix(Y_model), spearman_matrix(Y_data)
    iu = np.triu_indices(Y_model.shape[1], 1)
    return {"mean_abs_rho_model": float(np.abs(Sm[iu]).mean()),
            "mean_abs_rho_data": float(np.abs(Sd[iu]).mean()),
            "frobenius": float(np.linalg.norm(Sm - Sd))}



RUN = os.path.join(_ROOT, "outputs/checkpoint")
NGEN = 20000
SEED = 99
QSTAR = 0.05          # the q at which per-pair and aggregate tails are quoted

out = {}


def lam_pair(U, i, j, q, upper=False):
    ind = (U > 1 - q) if upper else (U < q)
    return float((ind[:, i] * ind[:, j]).mean() / q)


def lam_mean(U, q, upper=False):
    n = U.shape[1]
    iu = np.triu_indices(n, 1)
    return float(np.mean([lam_pair(U, i, j, q, upper) for i, j in zip(*iu)]))


def ranks(Y):
    return np.argsort(np.argsort(Y, 0), 0) / (len(Y) + 1.0)


# ---------------------------------------------------------------- data
X, _ = load_dataset(os.path.join(_ROOT, "data/cal_shower_img_8q.npy"), 4000, 7)
sp = make_train_test(X, 0.35, 7)
Ytr, Yte = sp["Y_train"], sp["Y_test"]
std = Standardiser(Ytr)
F, Fsd = independence_floor(Ytr, std, seed=7)
out["floor"] = F
out["floor_sd"] = Fsd

Ute = ranks(Yte)
rho_data = spearman = spearmanr(Yte)[0]
np.fill_diagonal(rho_data, 0.0)
iu = np.triu_indices(8, 1)
out["rho_data"] = float(np.abs(rho_data[iu]).mean())

# Ellipticity consistency check.
# For ANY elliptical copula, Kendall's tau fixes the dispersion parameter exactly:
#     tau = (2/pi) asin(rho)   <=>   rho = sin(pi*tau/2).
# For the Gaussian member the rank correlation then follows as
#     rho_S = (6/pi) asin(rho/2).
# We estimate rho from tau pair by pair and compare the prediction with measured rho_S.
devs = []
for i, j in zip(*iu):
    t = kendalltau(Yte[:, i], Yte[:, j])[0]
    r = spearmanr(Yte[:, i], Yte[:, j])[0]
    rho_disp = math.sin(math.pi * t / 2.0)
    devs.append(abs(r - (6.0 / math.pi) * math.asin(rho_disp / 2.0)))
out["ellip_max_dev"] = float(max(devs))

# data tails at QSTAR with bootstrap CIs (200 replicas)
rng = np.random.default_rng(0)
lL, lU = lam_mean(Ute, QSTAR, False), lam_mean(Ute, QSTAR, True)
bl, bu = [], []
for _ in range(200):
    k = rng.integers(0, len(Yte), len(Yte))
    Ub = ranks(Yte[k])
    bl.append(lam_mean(Ub, QSTAR, False))
    bu.append(lam_mean(Ub, QSTAR, True))
out["lamL_data"] = lL
out["lamL_data_ci"] = [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))]
out["lamU_data"] = lU
out["lamU_data_ci"] = [float(np.percentile(bu, 2.5)), float(np.percentile(bu, 97.5))]
out["Lambda_data"] = lL - lU
out["tail_q"] = QSTAR

# ---------------------------------------------------------------- model (sim)
mdl = QPCE.load(os.path.join(RUN, "params.npz"), QPCEConfig())
Yg = mdl.generate(NGEN, seed=SEED)
Ug = ranks(Yg)

ns = min(len(Yte), 1400)
E = energy_distance(std(Yg[:ns]), std(Yte[:ns]))
out["E_model"] = float(E)
out["D"] = float(1 - E / F)
rk = rank_summary(Yg, Yte)
out["rho_model"] = rk["mean_abs_rho_model"]
out["frobenius"] = rk["frobenius"]

# subsampling CI (same estimator as score.py)
rng = np.random.default_rng(0)
mm = ns // 2
bD, bR = [], []
for _ in range(200):
    i = rng.permutation(len(Yg))[:mm]
    k = rng.permutation(ns)[:mm]
    bD.append(1 - energy_distance(std(Yg[i]), std(Yte[k])) / F)
    j = rng.permutation(len(Yg))[:len(Yg) // 2]
    bR.append(rank_summary(Yg[j], Yte)["mean_abs_rho_model"])
sc = math.sqrt(mm / ns)
D = out["D"]
out["D_ci"] = [float(D - sc * (D - np.percentile(bD, 2.5))),
               float(D + sc * (np.percentile(bD, 97.5) - D))]
r0 = out["rho_model"]
out["rho_ci"] = [float(r0 - math.sqrt(0.5) * (r0 - np.percentile(bR, 2.5))),
                 float(r0 + math.sqrt(0.5) * (np.percentile(bR, 97.5) - r0))]

# W1 metrics in intensity units
qs = np.linspace(0, 1, 512)
w1 = np.array([np.abs(np.quantile(Yg[:, j], qs) - np.quantile(Yte[:, j], qs)).mean()
               for j in range(8)])
out["w1_cell_mean"] = float(w1.mean())
Eg, Ed = Yg.sum(1), Yte.sum(1)
w1E = float(np.abs(np.quantile(Eg, qs) - np.quantile(Ed, qs)).mean())
out["w1_E"] = w1E
out["w1_E_pct_mean"] = float(100.0 * w1E / Ed.mean())
out["w1_E_pct_width"] = float(100.0 * w1E / Ed.std())

# energy-sum width ratio (independence == 1 exactly)
def width_ratio(Y):
    return float(Y.sum(1).std() / math.sqrt(Y.var(0).sum()))
out["width_ratio_data"] = width_ratio(Yte)
out["width_ratio_model"] = width_ratio(Yg)

# joint two-sample permutation test on the energy distance, honest.
# n=m=700 disjoint halves of held-out vs generated, 500 permutations.
rng = np.random.default_rng(123)
A = std(Yg[:700])
B = std(Yte[:700])
obs = energy_distance(A, B)
pool = np.vstack([A, B])
cnt = 0
NPERM = 500
for _ in range(NPERM):
    per = rng.permutation(len(pool))
    if energy_distance(pool[per[:700]], pool[per[700:]]) >= obs:
        cnt += 1
out["perm_p"] = float((cnt + 1) / (NPERM + 1))
out["perm_obs"] = float(obs)

# model tails at QSTAR + Lambda
out["lamL_model"] = lam_mean(Ug, QSTAR, False)
out["lamU_model"] = lam_mean(Ug, QSTAR, True)
out["Lambda_model"] = out["lamL_model"] - out["lamU_model"]

# per-pair tails at QSTAR for pairs quoted in the table
for tag, (i, j) in {"ab": (1, 2), "bc": (2, 3), "gh": (6, 7)}.items():
    out[f"lamL_{tag}_M"] = lam_pair(Ug, i, j, QSTAR, False)
    out[f"lamL_{tag}_D"] = lam_pair(Ute, i, j, QSTAR, False)
    out[f"lamU_{tag}_M"] = lam_pair(Ug, i, j, QSTAR, True)
    out[f"lamU_{tag}_D"] = lam_pair(Ute, i, j, QSTAR, True)

# ---------------------------------------------------------------- strip test
c = mdl.circuit
rngE = np.random.default_rng(11)
Ew = rngE.random((NGEN, c.n_wires))
for freeze, key in ((True, "strip_frozen"), (False, "strip_free")):
    Z = c.no_entanglement_reference(Ew, freeze_shared=freeze)
    R = spearmanr(Z)[0]
    np.fill_diagonal(R, 0)
    out[key + "_max"] = float(np.abs(R).max())
    out[key + "_mean"] = float(np.abs(R[iu]).mean())

# sampling ceiling for the max-of-28 statistic at NGEN independent draws
rng = np.random.default_rng(5)
mx = []
for _ in range(200):
    W = rng.random((NGEN, 8))
    R = spearmanr(W)[0]
    np.fill_diagonal(R, 0)
    mx.append(np.abs(R).max())
out["strip_floor"] = float(np.percentile(mx, 97.5))

# ---------------------------------------------------------------- Gaussian ref
# Gaussian copula fitted on rank-transformed TRAIN, scored on held-out.
from scipy.stats import norm
Utr = ranks(Ytr)
G = norm.ppf(np.clip(Utr, 1e-9, 1 - 1e-9))
C = np.corrcoef(G.T)
Lch = np.linalg.cholesky(C)
rng = np.random.default_rng(3)
Zg = rng.standard_normal((NGEN, 8)) @ Lch.T
Ugau = norm.cdf(Zg)
# map through train marginals (inverse empirical CDF) to intensity units
Ygau = np.column_stack([np.quantile(Ytr[:, j], Ugau[:, j]) for j in range(8)])
rkg = rank_summary(Ygau, Yte)
out["G_rho"] = rkg["mean_abs_rho_model"]
out["G_frob"] = rkg["frobenius"]
Eg_ = energy_distance(std(Ygau[:ns]), std(Yte[:ns]))
out["G_D"] = float(1 - Eg_ / F)
Ug2 = ranks(Ygau)
out["G_lamL"] = lam_mean(Ug2, QSTAR, False)
out["G_lamU"] = lam_mean(Ug2, QSTAR, True)
out["G_lamL_err"] = abs(out["G_lamL"] - lL)
out["G_lamU_err"] = abs(out["G_lamU"] - lU)
out["M_lamL_err"] = abs(out["lamL_model"] - lL)
out["M_lamU_err"] = abs(out["lamU_model"] - lU)

# independence row: lambda(q) = q exactly; Frobenius = ||rho_data||_F; D = 0
out["I_frob"] = float(np.sqrt((rho_data[iu] ** 2).sum() * 2))
out["I_lamL_err"] = abs(QSTAR - lL)
out["I_lamU_err"] = abs(QSTAR - lU)

# ---------------------------------------------------------------- write
os.makedirs(os.path.join(_ROOT, "outputs"), exist_ok=True)
jpath = os.path.join(_ROOT, "outputs", "paper_numbers.json")
json.dump(out, open(jpath, "w"), indent=1)
print("saved", jpath)


def f(x, d=4):
    return f"{x:.{d}f}"


macros = {
    # simulation, deployed model, held-out
    "numRho": f(out["rho_model"]),
    "numRhoCiLo": f(out["rho_ci"][0], 3), "numRhoCiHi": f(out["rho_ci"][1], 3),
    "numRhoData": f(out["rho_data"]),
    "numResidF": f(out["frobenius"], 3),
    "numD": f(out["D"], 3),
    "numDCiLo": f(out["D_ci"][0], 3), "numDCiHi": f(out["D_ci"][1], 3),
    "numFfloor": f(out["floor"], 4), "numFfloorSd": f(out["floor_sd"], 4),
    "numEmodel": f(out["E_model"], 4),
    "numWoneCell": f(out["w1_cell_mean"], 4),
    "numWoneCellAbs": f"{out['w1_cell_mean']:.1e}".replace("e-0", r"\times10^{-") + "}",
    "numWoneE": f(out["w1_E"], 4),
    "numWoneEpct": f(out["w1_E_pct_mean"], 1),
    "numWoneEwidth": f(out["w1_E_pct_width"], 1),
    "numWidthData": f(out["width_ratio_data"], 3),
    "numWidthModel": f(out["width_ratio_model"], 3),
    "numPermP": f(out["perm_p"], 3),
    "numNgen": f"{NGEN:,}".replace(",", r"\,"),
    # strip / verification
    "numStripT": f(out["strip_frozen_max"], 4),
    "numStripFree": f(out["strip_free_max"], 3),
    "numStripFreeMean": f(out["strip_free_mean"], 3),
    "numStripFloor": f(out["strip_floor"], 4),
    # tails
    "numTailQ": f"{QSTAR:g}",
    "numLamLdata": f(out["lamL_data"], 3),
    "numLamLdataCiLo": f(out["lamL_data_ci"][0], 3),
    "numLamLdataCiHi": f(out["lamL_data_ci"][1], 3),
    "numLamUdata": f(out["lamU_data"], 3),
    "numLamUdataCiLo": f(out["lamU_data_ci"][0], 3),
    "numLamUdataCiHi": f(out["lamU_data_ci"][1], 3),
    "numLambdaData": f(out["Lambda_data"], 3),
    "numLamLmodel": f(out["lamL_model"], 3),
    "numLamUmodel": f(out["lamU_model"], 3),
    "numLambdaModel": f(out["Lambda_model"], 3),
    "numLamLerr": f(out["M_lamL_err"], 3),
    "numLamUerr": f(out["M_lamU_err"], 3),
    "numEllipMaxDev": f(out["ellip_max_dev"], 3),
    # per-pair table
    "numLamLabM": f(out["lamL_ab_M"], 2), "numLamLabD": f(out["lamL_ab_D"], 2),
    "numLamUabM": f(out["lamU_ab_M"], 2), "numLamUabD": f(out["lamU_ab_D"], 2),
    "numLamLbcM": f(out["lamL_bc_M"], 2), "numLamLbcD": f(out["lamL_bc_D"], 2),
    "numLamUbcM": f(out["lamU_bc_M"], 2), "numLamUbcD": f(out["lamU_bc_D"], 2),
    "numLamLghM": f(out["lamL_gh_M"], 2), "numLamLghD": f(out["lamL_gh_D"], 2),
    "numLamUghM": f(out["lamU_gh_M"], 2), "numLamUghD": f(out["lamU_gh_D"], 2),
    # Gaussian reference / independence
    "numGresidF": f(out["G_frob"], 3),
    "numGD": f(out["G_D"], 3),
    "numGlamLerr": f(out["G_lamL_err"], 3),
    "numGlamUerr": f(out["G_lamU_err"], 3),
    "numIresidF": f(out["I_frob"], 3),
    "numID": "0",
    "numIlamLerr": f(out["I_lamL_err"], 3),
    "numIlamUerr": f(out["I_lamU_err"], 3),
    # Gaussian-copula reference, the correctness anchor of Sec. VII
    "numGrho": f(out["G_rho"], 4),
    "numGlamL": f(out["G_lamL"], 3),
    "numGlamU": f(out["G_lamU"], 3),
    "numGLambda": f(out["G_lamL"] - out["G_lamU"], 3),
    # hardware
}
tex = ["% autogenerated by scripts/paper_numbers.py -- do not edit by hand"]
for k, v in macros.items():
    tex.append(rf"\newcommand{{\{k}}}{{{v}}}")
tpath = os.path.join(_ROOT, "outputs", "numbers.tex")
open(tpath, "w").write("\n".join(tex) + "\n")
print("saved", tpath)

for k, v in macros.items():
    print(f"{k:<22}{v}")
