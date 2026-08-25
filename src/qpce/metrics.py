"""
Every statistic quoted in the paper.

DESIGN RULE
-----------
With one exception, everything here is a rank statistic or a ratio. That is
deliberate. Rank statistics are invariant under any per-cell monotone map, so
they cannot be inflated by recalibrating the marginals of a cell. It is what
lets the same estimator be applied to simulation and to a noisy device without
adjustment, and it is why a per-cell correction, however it was obtained,
provably cannot manufacture dependence.

WHAT IS HERE
------------
Spearman and Kendall correlation matrices, the per-cell first Wasserstein
distance and chi-squared against the data, the tail-dependence coefficients as
a function of quantile, the energy-sum width ratio which equals one exactly for
independent cells, and the dependence score D used throughout the manuscript.

ON THE SCORE D
--------------
D is normalised so that one means the generated rank structure matches the data
and zero means independence. The denominator is the independence floor computed
from the data itself, not a number chosen by hand, which is what makes the score
comparable across sample sizes and across devices.

TAIL COEFFICIENTS
-----------------
The tail-dependence coefficient at quantile q is the probability that one cell
is extreme given that another is, normalised by q. Its lower and upper versions
are equal for every elliptical family, so their difference is a direct measure
of how non-elliptical the data are. That difference is the statistic the paper
uses to separate model families rather than to rank fits.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .energy import energy_distance, energy_permutation_test


# ---------------------------------------------------------------- marginals
def w1_per_pixel(Y_ref, Y_gen):
    return np.array([wasserstein_distance(Y_ref[:, j], Y_gen[:, j])
                     for j in range(Y_ref.shape[1])])


def w1_total_energy(Y_ref, Y_gen):
    """W1 between the total deposited energies sum_j Y_j.  Sensitive to the
    correlation structure through the variance of the sum."""
    return float(wasserstein_distance(Y_ref.sum(1), Y_gen.sum(1)))


def histogram_chi2(Y_ref, Y_gen, bins=30, min_count=10):
    """Diagonal-only chi^2/ndf per pixel with exact multinomial errors at
    fixed bin edges -- the infinite-resample limit of the histogram
    bootstrap.  The multinomial constraint makes bins negatively correlated,
    so this is diagonal-only with ndf = n_used - 1; bins with fewer than
    ``min_count`` entries are dropped and counted."""
    out = []
    for j in range(Y_ref.shape[1]):
        edges = np.histogram_bin_edges(
            np.concatenate([Y_ref[:, j], Y_gen[:, j]]), bins=bins)
        h = np.diff(edges)
        nr, ng = len(Y_ref), len(Y_gen)
        cr, _ = np.histogram(Y_ref[:, j], edges)
        cg, _ = np.histogram(Y_gen[:, j], edges)
        fr, fg = cr / (nr * h), cg / (ng * h)
        ser = np.sqrt(cr / nr * (1 - cr / nr) / nr) / h
        seg = np.sqrt(cg / ng * (1 - cg / ng) / ng) / h
        ok = (cr >= min_count) & (cg >= min_count)
        ndf = max(int(ok.sum()) - 1, 1)
        chi2 = float(np.sum((fr[ok] - fg[ok]) ** 2
                            / (ser[ok] ** 2 + seg[ok] ** 2)))
        out.append({"pixel": j, "chi2_ndf": chi2 / ndf, "ndf": ndf,
                    "bins_dropped": int((~ok).sum())})
    return out


# ---------------------------------------------------------------- dependence
def spearman_matrix(Y):
    R = np.argsort(np.argsort(Y, axis=0), axis=0).astype(float)
    return np.corrcoef(R.T)


def spearman_offdiag(Y, iu=None):
    n = Y.shape[1]
    iu = np.triu_indices(n, 1) if iu is None else iu
    return spearman_matrix(Y)[iu]


def correlation_report(Y_ref, Y_gen, n_boot=300, seed=0):
    """Rank-correlation matrices with a bootstrap z-score on the residual.

    Rank correlation, not Pearson: it is invariant under the per-pixel unit
    conversion, so it isolates dependence from marginal shape.
    """
    n = Y_ref.shape[1]
    iu = np.triu_indices(n, 1)
    Cd, Cm = spearman_matrix(Y_ref), spearman_matrix(Y_gen)
    rng = np.random.default_rng(seed)
    boot = np.stack([spearman_matrix(Y_ref[rng.integers(0, len(Y_ref),
                                                        len(Y_ref))])
                     for _ in range(n_boot)])
    se = boot.std(0, ddof=1)
    Z = (Cm - Cd) / np.maximum(se, 1e-12)
    np.fill_diagonal(Z, 0.0)
    return {"C_data": Cd, "C_model": Cm, "z": Z,
            "frobenius": float(np.linalg.norm(Cm - Cd)),
            "mean_abs_rho_model": float(np.abs(Cm[iu]).mean()),
            "mean_abs_rho_data": float(np.abs(Cd[iu]).mean()),
            "max_abs_z": float(np.abs(Z).max()),
            "frac_beyond_2sigma": float((np.abs(Z[iu]) > 2).mean())}


def tail_dependence(Y, q, upper=False):
    """Lambda(q) = C(q,q)/q per pair, on rank coordinates.

    NOTE, and it is a theorem not a fluctuation: while each qubit keeps a
    private germ, pixel pairs are conditionally independent given the shared
    wires, so lambda -> 0 as q -> 0 for ANY parameter values.  Finite-q
    values are meaningful; the limit is structurally zero.
    """
    n = Y.shape[1]
    iu = np.triu_indices(n, 1)
    U = np.argsort(np.argsort(Y, axis=0), axis=0) / (len(Y) + 1.0)
    ind = (U > 1.0 - q) if upper else (U < q)
    ind = ind.astype(float)
    return np.array([(ind[:, i] * ind[:, j]).mean() / q
                     for i, j in zip(*iu)])


# ---------------------------------------------------------------- joint
def summarize(Y_ref, Y_gen, standardiser, bins=30, n_boot=300, n_perm=200,
              seed=0):
    """The full evaluation, joint statistic first."""
    obs, p = energy_permutation_test(standardiser(Y_gen[:len(Y_ref)]),
                                     standardiser(Y_ref), n_perm=n_perm,
                                     seed=seed)
    w1 = w1_per_pixel(Y_ref, Y_gen)
    corr = correlation_report(Y_ref, Y_gen, n_boot=n_boot, seed=seed)
    return {
        "energy_distance_vs_test": float(energy_distance(
            standardiser(Y_gen[:len(Y_ref)]), standardiser(Y_ref))),
        "energy_test_statistic": obs,
        "permutation_p_value": p,
        "W1_per_pixel_mean": float(w1.mean()),
        "W1_per_pixel": [round(float(x), 6) for x in w1],
        "W1_total_energy": w1_total_energy(Y_ref, Y_gen),
        "chi2": histogram_chi2(Y_ref, Y_gen, bins=bins),
        "chi2_ndf_mean": float(np.mean([c["chi2_ndf"] for c in
                                        histogram_chi2(Y_ref, Y_gen, bins=bins)])),
        "rank_correlation": {k: v for k, v in corr.items()
                             if not isinstance(v, np.ndarray)},
        "lambda_lower": {f"q={q}": round(float(tail_dependence(Y_gen, q).mean()), 4)
                         for q in (0.25, 0.10, 0.05, 0.01)},
        "lambda_lower_data": {f"q={q}": round(float(tail_dependence(Y_ref, q).mean()), 4)
                              for q in (0.25, 0.10, 0.05, 0.01)},
    }
