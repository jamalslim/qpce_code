#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_paper.py -- publication figures in the manuscript's own style.

    python scripts/plot_paper.py outputs/checkpoint \

Emits, one file per panel, .pdf + .png, into outputs/<run>/ :

    corr_mc.pdf              rank correlation, MC reference (data)
    corr_simulation.pdf      rank correlation, noiseless simulation
    corr_qpce_ibm.pdf        rank correlation, QPU
    marginal_pixel_00..07    density + ratio-to-MC panel, three series
    total_energy             density + ratio-to-MC panel
    tail_marginals           log-density tails, all 8 pixels
    tail_dependence          lambda_L(q) and lambda_U(q), the statistic that
                             separates model FAMILIES rather than fits

ON THE TAIL FIGURES.  A calorimeter paper lives or dies on the tails, and the
linear-scale marginals hide them: the last decade of probability is invisible
under a peak of order 80. Two things are plotted.

(1) log-density marginals with the ratio panel kept linear, so a factor-2
    discrepancy at 1e-2 density is as visible as one at the mode.

(2) TAIL DEPENDENCE, lambda_L(q) = mean_pairs P(U_i<q, U_j<q)/q and its upper
    counterpart. This is not a refinement of the correlation matrix -- it is a
    different statistic, and it is the one on which model families separate:
    every ELLIPTICAL copula (Gaussian, Student-t at any nu) is radially
    symmetric and therefore has lambda_L == lambda_U identically, while the
    CLIC data has lambda_L = 0.167 against lambda_U = 0.069 at q = 0.005, with
    disjoint bootstrap CIs. A model can match the whole correlation matrix and
    still miss this completely, so it belongs in the paper.
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import mplhep as hep
    _HAS_HEP = True
except ImportError:                                          # pragma: no cover
    hep = None
    _HAS_HEP = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qpce import QPCE, QPCEConfig, load_dataset, make_train_test
from qpce.metrics import spearman_matrix

MC_C, SIM_C, HW_C = "#000000", "#1f77b4", "#d62728"
MC_L, SIM_L, HW_L = "MC data", "QPCE (Sim)", "QPCE (IBM)"


def style(name="ROOT"):
    """mplhep style if available, otherwise a close matplotlib approximation.

    The fallback exists so the script runs in environments without mplhep; it
    is NOT pixel-identical.  If mplhep is installed the mplhep path is used.
    """
    if _HAS_HEP:
        hep.style.use(getattr(hep.style, name, hep.style.ROOT))
        plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight",
                             "legend.frameon": False})
    else:
        plt.rcParams.update({
            "font.family": "serif", "font.size": 11, "axes.linewidth": 0.9,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False, "figure.dpi": 300,
            "savefig.bbox": "tight", "errorbar.capsize": 0})


def _hist_with_errors(x, bins):
    """Density histogram with POISSON errors.

    A generative-model comparison is a counting experiment: each bin holds N
    independent draws, so the density estimate carries sqrt(N) and the plot
    must show it.  v1 drew bare steps, which makes a 500-germ QPU sample look
    as certain as a 20000-sample simulation.  Returns (density, sigma_density,
    raw counts) -- the raw counts are needed to propagate the ratio error.
    """
    cnt, edges = np.histogram(x, bins=bins)
    w = np.diff(edges)
    N = max(len(x), 1)
    dens = cnt / (N * w)
    err = np.sqrt(cnt) / (N * w)
    return dens, err, cnt


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


# ---------------------------------------------------------------- heatmaps
def fig_corr(Y, title, out, name):
    C = spearman_matrix(Y)
    n = len(C)
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel("pixel"); ax.set_ylabel("pixel")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("correlation")
    save(fig, out, name)
    return C


# --------------------------------------------------------------- marginals
def fig_marginal(series, k, bins, out, name, logy=False, xlabel="intensity"):
    """Density panel + ratio-to-MC panel, both with Poisson error bars.

    Main panel: the reference and the QPU are drawn as ERRORBAR points (they
    are finite samples), the noiseless simulation as a step band (it can be
    made arbitrarily large, so its error is negligible by construction and
    drawing it as points would misrepresent that).

    Ratio panel: ERRORBARS ONLY, no connecting steps.  For r = n/d with both
    Poisson,  sigma_r / r = sqrt(1/N_n + 1/N_d)  in raw counts, so the bars
    widen exactly where the statistics run out -- which is the tails, which is
    the region the figure exists to show.
    """
    ctr = 0.5 * (bins[1:] + bins[:-1])
    xer = 0.5 * np.diff(bins)
    ref_d = ref_c = None
    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(5.0, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    for (lab, Y, col) in series:
        v = Y[:, k] if Y.ndim == 2 else np.asarray(Y).ravel()
        d, e, c = _hist_with_errors(v, bins)
        if lab == MC_L:
            ref_d, ref_c, ref_e = d, c, e
        # EVERY series gets BOTH: the step outline shows the binning, the
        # vertical bars show the statistical uncertainty.  Vertical only --
        # a horizontal bar would redraw the bin width the step already shows.
        a0.step(ctr, d, where="mid", color=col, lw=1.3, alpha=0.75)
        m = c > 0
        a0.errorbar(ctr[m], d[m], yerr=e[m], fmt="o", ms=3.0, lw=1.0,
                    capsize=0, color=col, label=lab)

    # ratio panel: the MC reference carries its OWN error, drawn as a band at
    # 1.  Without it the figure implies the denominator is exact, which at
    # 1400 events per pixel it is not -- in the tails the reference is the
    # dominant uncertainty, and a model point sitting outside a naive band can
    # be perfectly consistent once it is included.
    ok0 = (ref_d > 0) & (ref_c > 0)
    rel = np.zeros_like(ref_d)
    rel[ok0] = 1.0 / np.sqrt(ref_c[ok0])
    a1.fill_between(ctr[ok0], 1 - rel[ok0], 1 + rel[ok0], step="mid",
                    color=MC_C, alpha=0.16, lw=0, label=MC_L)
    for (lab, Y, col) in series:
        if lab == MC_L:
            continue
        v = Y[:, k] if Y.ndim == 2 else np.asarray(Y).ravel()
        d, e, c = _hist_with_errors(v, bins)
        good = (ref_d > 0) & (ref_c > 0) & (c > 0)
        r = np.full_like(d, np.nan)
        re = np.full_like(d, np.nan)
        r[good] = d[good] / ref_d[good]
        # ONLY the numerator's error here: the denominator's is the shaded
        # band, so including it in the bars too would double-count it.
        re[good] = r[good] / np.sqrt(c[good])
        # RATIO PANEL: points with vertical bars ONLY -- no step outline.
        # A ratio is not a distribution: the step implies a binned density and
        # invites the eye to read area, which is meaningless here. Points also
        # keep the panel readable where consecutive bins swing across 1.
        a1.errorbar(ctr[good], r[good], yerr=re[good], fmt="o", ms=3.0,
                    lw=1.0, capsize=0, color=col)

    a0.set_ylabel("density")
    if logy:
        a0.set_yscale("log")
        pos = [d for _, Y, _ in series
               for d in [_hist_with_errors(
                   Y[:, k] if Y.ndim == 2 else np.asarray(Y).ravel(), bins)[0]]]
        m = min(x[x > 0].min() for x in pos if (x > 0).any())
        a0.set_ylim(0.5 * m, None)
    else:
        a0.set_ylim(0, None)
    a0.legend(fontsize=14)
    a1.axhline(1.0, color=MC_C, lw=0.9)
    a1.set_ylim(0.4, 1.6)
    a1.set_ylabel(f"ratio", fontsize=18)
    a1.set_xlabel(xlabel)
    save(fig, out, name)


# ---------------------------------------------------------- tail dependence
def _lam(Y, q, upper=False, U=None):
    U = np.argsort(np.argsort(Y, 0), 0) / (len(Y) + 1.0) if U is None else U
    iu = np.triu_indices(Y.shape[1], 1)
    ind = ((U > 1 - q) if upper else (U < q)).astype(float)
    return float(np.mean([(ind[:, i] * ind[:, j]).mean() / q
                          for i, j in zip(*iu)]))


def fig_tail_dependence(series, out, name, qs=(0.25, 0.15, 0.10, 0.05,
                                               0.02, 0.01), n_boot=120):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for lab, Y, col in series:
        U = np.argsort(np.argsort(Y, 0), 0) / (len(Y) + 1.0)
        for ax, up in ((a0, False), (a1, True)):
            v = np.array([_lam(Y, q, up, U) for q in qs])
            lo = np.zeros_like(v); hi = np.zeros_like(v)
            for j, q in enumerate(qs):
                b = [_lam(Y[rng.integers(0, len(Y), len(Y))], q, up)
                     for _ in range(n_boot)]
                lo[j], hi[j] = np.percentile(b, [16, 84])
            ax.errorbar(qs, v, yerr=[v - lo, hi - v], fmt="o-", ms=4, lw=1.3,
                        color=col, label=lab)
    for ax, t in ((a0, r"lower tail  $\lambda_L(q)$"),
                  (a1, r"upper tail  $\lambda_U(q)$")):
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("quantile $q$"); 
    a0.set_ylabel(r"$\lambda(q)$")
    a0.legend(fontsize=14)
    save(fig, out, name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--data", default="data/cal_shower_img_8q.npy")
    ap.add_argument("--n-gen", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--hep-style", default="ROOT",
                    help="mplhep style: ROOT, ATLAS, CMS, LHCb")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    style(a.hep_style)

    rp = lambda p: p if os.path.exists(p) else os.path.join(_ROOT, p)
    out = a.outdir or os.path.join(_ROOT, "plots",
                                   os.path.basename(a.run.rstrip("/")))
    os.makedirs(out, exist_ok=True)

    X, _ = load_dataset(rp(a.data), 4000, 7)
    Yd = make_train_test(X, 0.35, 7)["Y_test"]
    Ys = QPCE.load(os.path.join(rp(a.run), "params.npz"),
                    QPCEConfig()).generate(a.n_gen, seed=a.seed)
    series = [(MC_L, Yd, MC_C), (SIM_L, Ys, SIM_C)]
    n = Yd.shape[1]
    iu = np.triu_indices(n, 1)
    Cd = fig_corr(Yd, MC_L, out, "corr_mc")
    Cs = fig_corr(Ys, SIM_L, out, "corr_simulation")
    print(f"mean |rho_S|  MC {np.abs(Cd[iu]).mean():.4f}   "
          f"sim {np.abs(Cs[iu]).mean():.4f}", end="")
    print()

    for k in range(n):
        lo = min(Y[:, k].min() for _, Y, _ in series)
        hi = max(Y[:, k].max() for _, Y, _ in series)
        b = np.linspace(lo, hi, 41)
        fig_marginal(series, k, b, out, f"marginal_pixel_{k:02d}")
        fig_marginal(series, k, b, out, f"tail_pixel_{k:02d}", logy=True)

    Es = [(lab, Y.sum(1)[:, None], c) for lab, Y, c in series]
    lo = min(e[:, 0].min() for _, e, _ in Es)
    hi = max(e[:, 0].max() for _, e, _ in Es)
    b = np.linspace(lo, hi, 51)
    fig_marginal(Es, 0, b, out, "total_energy",
                 xlabel=r"total energy")
    fig_marginal(Es, 0, b, out, "total_energy_tail", logy=True,
                 xlabel=r"total energy")

    fig_tail_dependence(series, out, "tail_dependence")
    print(f"\n{'series':<16}{'lam_L(0.05)':>13}{'lam_U(0.05)':>13}"
          f"{'Lambda':>10}")
    for lab, Y, _ in series:
        L, U = _lam(Y, 0.05), _lam(Y, 0.05, True)
        print(f"{lab:<16}{L:>13.4f}{U:>13.4f}{L - U:>+10.4f}")
    print(f"\n[saved] {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
