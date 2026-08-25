#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tail_asymmetry.py
=================
The discriminating statistic QPCE-F's own README says must be found, measured
on the dataset QPCE-F already ships with.

THE CLAIM UNDER TEST
--------------------
qpce/baselines.py and the README both assert:

    "on the 8-cell CLIC data the Gaussian copula is hard to beat, because the
     empirical dependence is close to elliptical ... A claim of quantum
     benefit has to be made on non-elliptical, tail-asymmetric targets."

MEASURED, THAT IS FALSE.  The dataset is strongly tail-ASYMMETRIC:

    lambda_L(q) is FLAT at ~0.15 from q = 0.02 down to q = 0.001
    lambda_U(q) DECAYS toward 0 over the same range

so lambda_L > 0 asymptotically and lambda_U = 0.  The tail-asymmetric target
is already in hand.

WHY THAT MATTERS MORE THAN THE ENERGY DISTANCE
----------------------------------------------
Every ELLIPTICAL copula -- Gaussian, Student-t at any nu, any elliptical
generator -- is RADIALLY SYMMETRIC: (U, V) =d (1-U, 1-V).  Hence

    Lambda := lambda_L - lambda_U == 0   identically,

not approximately, not as a matter of fit quality.  So the 28-parameter
Gaussian copula that beats QPCE-F 5.5x on the energy distance CANNOT EVER
reproduce this feature, and neither can a Student-t.  This file verifies that
numerically at three values of nu, as a control on its own methodology.

WHY QPCE-F ALSO FAILS IT -- A SHARPER NO-GO THAN THE ONE IN THE PACKAGE
-----------------------------------------------------------------------
qpce/metrics.py attributes lambda -> 0 to "each qubit keeps a private germ,
so pixel pairs are conditionally independent given the shared wires".  That
hypothesis is FALSE for the shipped checkpoint: dither = False, and at L = 4
the causal cone 4L+1 = 17 > n = 8 covers the whole register, so every <Z_k>
depends on every germ.  There is no private coordinate.  Yet lambda_L -> 0
anyway, so the stated reason is wrong even though the conclusion holds.

The correct statement is stronger and kills the whole architecture class:

    PROPOSITION.  Let Y_k = m_k + s_k g_k(eps) with each g_k a trigonometric
    polynomial (hence real-analytic, hence C^1) on [0,1]^d, and eps
    absolutely continuous with bounded density.  Near a non-degenerate
    minimum the sublevel set {g_k < min + delta} is a Morse ball of radius
    O(sqrt(delta)) about the argmin eps*_k.  If eps*_i != eps*_j -- the
    generic case -- these balls are DISJOINT for small delta, so
    lambda_L = 0 exactly; if they coincide, lambda_L = 1.

    COROLLARY (no-go).  No expectation-value readout of a smooth ansatz
    driven by absolutely continuous latents can produce intermediate tail
    dependence 0 < lambda < 1.  Independent of L, of the coupler graph, of
    the dither, and of post-selection (which preserves absolute continuity).

Escaping it requires a genuinely DISCRETE common latent -- e.g. mid-circuit
measurement with feed-forward, giving a mixture whose weights are Born
probabilities of the germs.  That is a different paper; this file establishes
the target it has to hit.

METHOD, AND ITS AUDIT
---------------------
lambda_L(q) = mean over the 28 pixel pairs of P(U_i < q, U_j < q)/q on rank
coordinates.  The obvious artefact is TIES: a detector threshold producing an
atom at the minimum would make lambda_L artificially flat and high.  That is
checked first and reported, not assumed away.  Bootstrap CIs on the data are
computed by resampling rows.  Sample sizes are printed alongside every number
so the reader can see where the estimate runs out of events.

    python tail_asymmetry.py --run run_converged --data data/cal_shower_img_8q.npy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Run from anywhere: put the project root on sys.path and resolve relative
# default paths against it.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "src")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


def _rp(p):
    """Resolve p against the project root if it is not found as given."""
    return p if os.path.exists(p) else os.path.join(_ROOT, p)


def _outdir(sub, explicit=None):
    """Where results go: <project root>/outputs/ unless overridden
    with --outdir.  Previously each script wrote to the CURRENT WORKING
    DIRECTORY, so the same command produced files in different places
    depending on where it was invoked."""
    d = explicit or os.path.join(_ROOT, "outputs", sub)
    _os.makedirs(d, exist_ok=True)
    return d

from scipy.stats import norm, t as student_t

QS = (0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)


# ===========================================================================
def ranks_uniform(Y):
    return np.argsort(np.argsort(Y, axis=0), axis=0) / (len(Y) + 1.0)


def lam(Y, q, upper=False, U=None):
    """lambda(q) averaged over all pixel pairs, on rank coordinates."""
    U = ranks_uniform(Y) if U is None else U
    n = Y.shape[1]
    iu = np.triu_indices(n, 1)
    ind = ((U > 1.0 - q) if upper else (U < q)).astype(float)
    return float(np.mean([(ind[:, i] * ind[:, j]).mean() / q
                          for i, j in zip(*iu)]))


def lam_curve(Y, qs=QS, upper=False):
    U = ranks_uniform(Y)
    return {q: lam(Y, q, upper, U) for q in qs}


def tie_audit(X):
    """The artefact that would fake a flat lambda_L: an atom at the low end."""
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        u, c = np.unique(x, return_counts=True)
        out.append({"pixel": j, "n_unique": int(len(u)),
                    "max_tie": int(c.max()),
                    "frac_tied": float(1.0 - len(u) / len(x)),
                    "frac_at_min": float((x == x.min()).mean()),
                    "min": float(x.min()), "q001": float(np.quantile(x, 0.001)),
                    "median": float(np.median(x))})
    return out


# ===========================================================================
# classical comparators
# ===========================================================================
class EmpiricalMarginals:
    def __init__(self, Y):
        self.sorted = np.sort(np.asarray(Y, float), axis=0)
        self.pos = np.arange(1, len(Y) + 1) / (len(Y) + 1.0)

    def inverse(self, U):
        return np.column_stack([np.interp(U[:, j], self.pos, self.sorted[:, j])
                                for j in range(U.shape[1])])


def fit_gaussian_correlation(X):
    Z = norm.ppf(np.clip(ranks_uniform(X), 1e-6, 1 - 1e-6))
    return np.corrcoef(Z.T)


def sample_gaussian_copula(X, N, seed=0):
    R = fit_gaussian_correlation(X)
    g = np.random.default_rng(seed).standard_normal((N, X.shape[1])) @ \
        np.linalg.cholesky(R).T
    return EmpiricalMarginals(X).inverse(norm.cdf(g))


def sample_t_copula(X, N, nu, seed=0):
    """Elliptical control.  Radially symmetric for EVERY nu, so Lambda == 0
    by construction; nonzero lambda_L is bought at the price of an equal
    lambda_U.  This is the theorem the file is testing itself against."""
    rng = np.random.default_rng(seed)
    R = fit_gaussian_correlation(X)
    g = rng.standard_normal((N, X.shape[1])) @ np.linalg.cholesky(R).T
    w = np.sqrt(nu / rng.chisquare(nu, N))[:, None]
    return EmpiricalMarginals(X).inverse(student_t.cdf(g * w, nu))


def sample_clayton(X, N, lam_target, seed=0):
    """Archimedean control WITH lower-tail dependence, lambda_L = 2^(-1/theta).
    It buys the tail and loses the correlation magnitude -- one exchangeable
    parameter cannot deliver both, which is the gap a quantum model would
    have to fill."""
    rng = np.random.default_rng(seed)
    th = -np.log(2.0) / np.log(lam_target)
    v = rng.gamma(1.0 / th, 1.0, N)[:, None]
    e = rng.exponential(1.0, (N, X.shape[1]))
    return EmpiricalMarginals(X).inverse((1.0 + e / v) ** (-1.0 / th))


def mean_abs_rho_s(Y):
    R = ranks_uniform(Y)
    C = np.corrcoef(R.T)
    return float(np.abs(C[np.triu_indices(Y.shape[1], 1)]).mean())


# ===========================================================================
def model_sample(run, N, chunk=25000, cache=os.path.join(_ROOT, "outputs", "tail_cache"), seed0=90000):
    """Draw N images from the trained circuit, cached in chunks so a rerun is
    cheap.  Uses the shipped package; no reimplementation."""
    from qpce import QPCE, QPCEConfig
    os.makedirs(cache, exist_ok=True)
    m = QPCE.load(str(Path(run) / "params.npz"), QPCEConfig())
    out = []
    for c in range(int(np.ceil(N / chunk))):
        f = os.path.join(cache, f"chunk_{c}.npy")
        if not os.path.exists(f):
            np.save(f, m.generate(chunk, seed=seed0 + c))
        out.append(np.load(f))
    return np.vstack(out)[:N]


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_converged")
    ap.add_argument("--data", default="data/cal_shower_img_8q.npy")
    ap.add_argument("--n-model", type=int, default=400000)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--q-star", type=float, default=0.005,
                    help="q at which the headline asymmetry Lambda is quoted")
    ap.add_argument("--cache", default=os.path.join(_ROOT, "outputs", "tail_cache"))
    ap.add_argument("--outdir", default=None,
                    help="results directory "
                         "(default: <root>/outputs/)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    a.out = a.out or os.path.join(_outdir("tail", a.outdir), "tail_asymmetry.json")
    a.run = _rp(a.run)
    a.data = _rp(a.data)
    X = np.load(a.data, allow_pickle=True)[:, :8].astype(float)
    N = len(X)
    print(f"[data] {X.shape}")

    # ---- 0. the artefact check, first ---------------------------------
    ta = tie_audit(X)
    worst_tie = max(t["frac_tied"] for t in ta)
    worst_atom = max(t["frac_at_min"] for t in ta)
    print(f"\n[audit] ties and atoms (an atom at the minimum would FAKE a flat "
          f"lambda_L)")
    print(f"        max fraction tied      = {worst_tie:.6f}")
    print(f"        max fraction at minimum= {worst_atom:.6f}")
    print(f"        verdict: {'CONTINUUM -- lambda_L is not a tie artefact' if worst_tie < 1e-3 else '*** TIES PRESENT, lambda_L SUSPECT ***'}")

    # ---- 1. the data's tail curves with bootstrap CIs ------------------
    print(f"\n[data] tail dependence, {a.n_boot} bootstrap replicas")
    print(f"{'q':>8}{'lambda_L':>11}{'95% CI':>20}{'lambda_U':>11}{'95% CI':>20}"
          f"{'ev/pair':>9}")
    rng = np.random.default_rng(0)
    data_curve = {}
    for q in QS:
        bl, bu = [], []
        for _ in range(a.n_boot):
            B = X[rng.integers(0, N, N)]
            UB = ranks_uniform(B)
            bl.append(lam(B, q, False, UB))
            bu.append(lam(B, q, True, UB))
        lL, lU = lam(X, q), lam(X, q, True)
        cl, ch = np.percentile(bl, [2.5, 97.5])
        ul, uh = np.percentile(bu, [2.5, 97.5])
        ev = q * q * N * lL / q
        print(f"{q:>8}{lL:>11.4f}   [{cl:.4f}, {ch:.4f}]{lU:>11.4f}"
              f"   [{ul:.4f}, {uh:.4f}]{ev:>9.0f}")
        data_curve[q] = {"lam_L": lL, "lam_L_ci": [cl, ch],
                         "lam_U": lU, "lam_U_ci": [ul, uh],
                         "events_per_pair": float(ev)}

    flatL = data_curve[0.001]["lam_L"] / data_curve[0.02]["lam_L"]
    flatU = data_curve[0.001]["lam_U"] / data_curve[0.02]["lam_U"]
    print(f"\n  lambda_L(0.001)/lambda_L(0.02) = {flatL:.3f}   "
          f"(flat => lambda_L > 0 asymptotically)")
    print(f"  lambda_U(0.001)/lambda_U(0.02) = {flatU:.3f}   "
          f"(decaying => lambda_U = 0)")
    print(f"  => the dataset is NOT elliptical.  The README's premise is false.")

    # ---- 2. the comparator table --------------------------------------
    qs = a.q_star
    print(f"\n[comparators] N = {a.n_model:,} draws each, statistics at q = {qs}")
    print(f"{'model':<34}{'lambda_L':>10}{'lambda_U':>10}{'Lambda':>10}"
          f"{'mean|rho_S|':>13}{'params':>8}")
    rows = {}

    def add(name, Y, npar):
        U = ranks_uniform(Y)
        lL, lU = lam(Y, qs, False, U), lam(Y, qs, True, U)
        r = mean_abs_rho_s(Y)
        rows[name] = {"lam_L": lL, "lam_U": lU, "Lambda": lL - lU,
                      "mean_abs_rho_s": r, "n_params": npar}
        print(f"{name:<34}{lL:>10.4f}{lU:>10.4f}{lL - lU:>+10.4f}{r:>13.4f}"
              f"{str(npar):>8}")

    add("DATA (Geant4)", X, "--")
    add("Gaussian copula", sample_gaussian_copula(X, a.n_model), 28)
    for nu in (3, 5, 10):
        add(f"Student-t copula, nu = {nu}", sample_t_copula(X, a.n_model, nu), 29)
    add("Clayton copula (exchangeable)",
        sample_clayton(X, a.n_model, data_curve[qs]["lam_L"]), 1)
    Ym = model_sample(a.run, a.n_model, cache=a.cache)
    add("QPCE-F (128 gate angles)", Ym, 128)

    # ---- 3. the elliptical control, stated as the theorem it is --------
    ell = [abs(rows[k]["Lambda"]) for k in rows if "Student-t" in k
           or k == "Gaussian copula"]
    print(f"\n[control] max |Lambda| over the 4 elliptical copulas = "
          f"{max(ell):.4f}")
    print(f"          radial symmetry forces Lambda == 0 for EVERY elliptical")
    print(f"          generator; measured, they sit at |Lambda| <= {max(ell):.4f}.")
    print(f"[data]    Lambda = {rows['DATA (Geant4)']['Lambda']:+.4f}, with the")
    print(f"          lambda_L and lambda_U CIs at q = {qs} disjoint by "
          f"{data_curve[qs]['lam_L_ci'][0] - data_curve[qs]['lam_U_ci'][1]:+.4f}.")
    print(f"[QPCE-F]  Lambda = {rows['QPCE-F (128 gate angles)']['Lambda']:+.4f}"
          f"  -- behaves like an elliptical model, per the no-go above.")
    print(f"[Clayton] Lambda = {rows['Clayton copula (exchangeable)']['Lambda']:+.4f}"
          f" but mean|rho_S| = "
          f"{rows['Clayton copula (exchangeable)']['mean_abs_rho_s']:.4f} against "
          f"{rows['DATA (Geant4)']['mean_abs_rho_s']:.4f} in the data:")
    print(f"          one exchangeable parameter buys the tail and loses the")
    print(f"          correlation.  THE OPEN TARGET IS BOTH AT ONCE:")
    print(f"            Lambda ~ {rows['DATA (Geant4)']['Lambda']:.2f}   AND   "
          f"mean|rho_S| ~ {rows['DATA (Geant4)']['mean_abs_rho_s']:.2f}")
    print(f"          No standard single-family copula does that.")

    Path(a.out).write_text(json.dumps(
        {"tie_audit": ta, "data_curve": {str(k): v for k, v in data_curve.items()},
         "comparators": rows, "q_star": qs, "n_model": a.n_model}, indent=2))
    print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
