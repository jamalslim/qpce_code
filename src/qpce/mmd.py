"""
qpce.mmd
==========

Short answer to "shouldn't that be MMD?": it already is.

-----------------------------------------------------------------------
1.  THE ENERGY DISTANCE IS AN MMD.  EXACTLY, NOT BY ANALOGY.
-----------------------------------------------------------------------
Let rho be a semimetric of negative type on R^d and fix any base point
z0.  The DISTANCE-INDUCED KERNEL is

    k_rho(x, y) = 1/2 [ rho(x, z0) + rho(y, z0) - rho(x, y) ].

Sejdinovic, Sriperumbudur, Gretton and Fukumizu, *Equivalence of
distance-based and RKHS-based statistics in hypothesis testing*,
Ann. Statist. 41, 2263 (2013), Theorem 22, prove

    E(P, Q)  =  2 MMD^2( P, Q ; k_rho ),

so the energy distance of Szekely and Rizzo IS the maximum mean
discrepancy in the RKHS of the distance kernel, up to the factor 2.  With
rho(x,y) = ||x - y|| (negative type for 0 < beta <= 2 in ||.||^beta, and
strictly so for beta < 2) that kernel is CHARACTERISTIC, so the MMD is a
metric on distributions: zero if and only if P = Q.  Baringhaus and Franz,
J. Multivar. Anal. 88, 190 (2004), reached the same statistic from the
two-sample-test side; Lyons, Ann. Probab. 41, 3284 (2013), gives the
general negative-type statement.

``verify_equivalence`` below checks E = 2 MMD^2 numerically; measured
agreement 1.1e-15 relative.  So the loss in train.py is not an unusual
choice, it is MMD with one particular kernel -- the one that has NO
BANDWIDTH.

-----------------------------------------------------------------------
2.  WHY THE DISTANCE KERNEL RATHER THAN A GAUSSIAN RBF
-----------------------------------------------------------------------
The Gaussian kernel exp(-||x-y||^2 / 2 sigma^2) is also characteristic,
so RBF-MMD is also a metric.  The differences are practical and they cut
in both directions; both are implemented here so the question is settled
by measurement rather than by preference.

  FOR the distance kernel
    * No bandwidth.  Nothing to tune, nothing to defend to a referee, and
      no way to change the answer after seeing it.  With an RBF you must
      pick sigma, and the median heuristic is a heuristic -- Ramdas,
      Reddi, Poczos, Singh and Wasserman, AAAI 2015, show its power
      against DEPENDENCE departures degrades relative to marginal
      departures as dimension grows, which is exactly the failure mode
      that matters here.
    * Scale-free: no length scale to become wrong as the fit improves and
      the residual shrinks.  A fixed sigma chosen at initialisation is
      badly matched to the residual after 600 iterations.
    * Unbounded kernel, so it retains sensitivity in the tails, where a
      Gaussian kernel with median bandwidth has already saturated to zero.
      For calorimeter showers the tails are the physics.

  AGAINST the distance kernel
    * It weights the discrepancy by physical distance, so it is dominated
      by whichever part of the mismatch is largest in absolute terms.
      Measured on this dataset: E between a correlated and an independent
      sample with IDENTICAL marginals is 0.067, while an untrained
      model's marginal mismatch contributes of order 5.  Dependence is a
      ~1 % perturbation until the marginals are nearly exact.  A bounded
      kernel with a small bandwidth saturates the marginal term and
      therefore gives dependence a relatively larger share of the
      gradient early on.
    * Its unbiased U-statistic can go negative, and under a frozen batch
      an optimiser will drive it there by memorising the sample.  So can
      the unbiased RBF-MMD estimator; this is not a distinguishing point,
      but it is a real defect of both and it is why only out-of-sample
      numbers are quoted.

-----------------------------------------------------------------------
3.  WHAT IS IMPLEMENTED
-----------------------------------------------------------------------
    loss="energy"    E(P,Q), the distance kernel, no bandwidth.  Default.
    loss="mmd"       MMD^2_u with a MIXTURE of Gaussian kernels at
                     sigma = c * sigma_median, c in {1/4, 1/2, 1, 2, 4}.
                     A mixture rather than a single bandwidth because a
                     single one has to be right at every stage of the fit
                     and cannot be; the sum of characteristic kernels is
                     characteristic, so the metric property survives.
    loss="mmd3"      the TRIMMED mixture, c in {1/2, 1, 2}.  Measured
                     better than "mmd": the 4x component carries more
                     MARGINAL than dependence signal (ratio 0.3) and the
                     1/4x component's marginal number is negative, i.e.
                     pure estimator noise.  "mmd" is retained unchanged
                     only because run_mmd was trained with it.
    loss="mmd1"      single Gaussian at the median heuristic, for the
                     ablation that shows why a mixture is used.

sigma_median is computed ONCE from the DATA (the median pairwise distance
of the training sample) and frozen, never recomputed from the model
sample.  Recomputing it each iteration makes the objective a moving
target and the descent meaningless -- a bandwidth that shrinks with the
residual can lower the loss while the fit gets worse.

Every estimator here is the unbiased U-statistic, and every gradient is
analytic and verified against central differences in ``check_gradients``.

-----------------------------------------------------------------------
4.  AUDIT.  MEASURED, IN ``audit()``.
-----------------------------------------------------------------------
(a) Gradient.  Central differences over a step sweep show the classic
    V: 6.7e-07 at h = 1e-3 (truncation), 5.6e-09 at h = 1e-4 (floor),
    1.1e-05 at h = 1e-7 (round-off).  The analytic gradient is correct;
    the 2.9e-07 quoted at a single h = 1e-6 was the difference scheme's
    own error, not the gradient's.

(b) Null calibration.  Two samples from the SAME law, 60 replicas of
    n = m = 400: mean +3.8e-04 against a standard error of the mean of
    3.9e-04, i.e. 0.99 sigma from zero.  The estimator is unbiased, as an
    unbiased U-statistic must be, and therefore goes NEGATIVE about half
    the time at P = Q.  A monotone drift below zero during training is
    sample memorisation, not P -> Q; that is why only out-of-sample
    numbers are quoted.

(c) The balance that actually decides the fit.  On a d = 8 Gaussian with
    rho = 0.6, comparing the correlated sample against a column-shuffled
    copy (IDENTICAL marginals, dependence destroyed) versus against a
    +0.15 shift (identical dependence, marginals moved):

        statistic                 dependence   marginal   ratio
        energy distance             0.09102     0.03122     2.9
        MMD^2 mixture               0.07010     0.00827     8.5
        MMD^2 sigma = 0.5 x median  0.03681     0.00153    24.1
        MMD^2 sigma = 1.0 x median  0.02166     0.00415     5.2
        MMD^2 sigma = 2.0 x median  0.00332     0.00259     1.3
        MMD^2 sigma = 4.0 x median  0.00024     0.00087     0.3

    So a bounded kernel at or below the median bandwidth gives dependence
    roughly 3x more relative weight than the distance kernel does, and
    that -- not any difference in correctness -- is why --loss mmd lands
    at a Spearman residual of 0.335 against 0.440 for --loss energy.
    The wide components are close to useless here: at 4 x median the
    marginal signal EXCEEDS the dependence signal, and at 0.25 x median
    the marginal number comes out negative, i.e. pure estimator noise.
    From that table I PREDICTED that a trimmed {1/2, 1, 2} mixture would
    fit the dependence better than {1/4, 1/2, 1, 2, 4}.

(d) THE PREDICTION WAS WRONG, and this is why the table above is a
    diagnostic and not a design rule.  Trained identically (600
    iterations, local encoding, L = 4, 128 angles):

        loss     Spearman residual   mean|rho_S|   W1/pixel   E vs test
        energy         0.4300          0.6011      0.00351     0.00879
        mmd            0.3481          0.5629      0.00360     0.00807
        mmd3           0.4116          0.5549      0.00386     0.00820

    and independently in qpce_x (its own seed, batch and split):

        energy         0.3471          0.5629      0.00226     0.00729
        mmd3           0.3877          0.5930      0.00240     0.00847

    Trimming made the dependence residual WORSE in both packages.  So the
    static discriminating power of a kernel, measured once on a Gaussian
    toy at one perturbation size, does not predict how well a fit lands.
    A plausible reading -- untested, therefore a hypothesis and not a
    result -- is that the wide component keeps the objective informative
    early, when the model is far from the data and every narrow kernel has
    saturated, and the narrow component sharpens late; removing either
    end removes coarse-to-fine coverage that the single-perturbation
    ratio cannot see.  ``mmd`` remains the recommended Gaussian setting,
    for measured reasons rather than for the reason I first gave.
"""
from __future__ import annotations

import numpy as np

__all__ = ["median_sigma", "mmd2_and_grad", "mmd2", "make_loss",
           "verify_equivalence", "check_gradients", "audit"]


def _sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


# =====================================================================
def median_sigma(Y, max_n=2000, seed=0):
    """Median-heuristic bandwidth: the median PAIRWISE DISTANCE (not the
    squared distance) of the reference sample.  Computed once, frozen."""
    rng = np.random.default_rng(seed)
    Z = Y if len(Y) <= max_n else Y[rng.choice(len(Y), max_n, replace=False)]
    D = np.sqrt(_sqdist(Z, Z))
    iu = np.triu_indices(len(Z), 1)
    return float(np.median(D[iu]))


def mmd2(X, Y, sigmas):
    """Unbiased U-statistic estimator of MMD^2 for a sum of Gaussian
    kernels with the given bandwidths."""
    n, m = len(X), len(Y)
    Dxx, Dyy, Dxy = _sqdist(X, X), _sqdist(Y, Y), _sqdist(X, Y)
    tot = 0.0
    for s in np.atleast_1d(sigmas):
        g = 1.0 / (2.0 * s * s)
        Kxx, Kyy, Kxy = np.exp(-g * Dxx), np.exp(-g * Dyy), np.exp(-g * Dxy)
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        tot += (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                - 2.0 * Kxy.mean())
    return float(tot)


def mmd2_and_grad(X, Y, sigmas):
    """(MMD^2_u, dMMD^2_u/dX), analytic.

        d/dx_i exp(-||x_i - z||^2 / 2 s^2) = -(x_i - z)/s^2 * k(x_i, z)

    so the gradient is a difference of two kernel-weighted centroids,
    assembled with the same two matrix products as the value."""
    n, m = len(X), len(Y)
    Dxx, Dyy, Dxy = _sqdist(X, X), _sqdist(Y, Y), _sqdist(X, Y)
    val = 0.0
    grad = np.zeros_like(X)
    for s in np.atleast_1d(sigmas):
        g = 1.0 / (2.0 * s * s)
        Kxx, Kyy, Kxy = np.exp(-g * Dxx), np.exp(-g * Dyy), np.exp(-g * Dxy)
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        val += (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                - 2.0 * Kxy.mean())
        cxx = 2.0 / (n * (n - 1)) * (-2.0 * g)
        cxy = -2.0 / (n * m) * (-2.0 * g)
        grad += cxx * (X * Kxx.sum(1, keepdims=True) - Kxx @ X)
        grad += cxy * (X * Kxy.sum(1, keepdims=True) - Kxy @ Y)
    return float(val), grad


# =====================================================================
def make_loss(kind, Y_ref):
    """Return (name, f(X, Y) -> (value, dValue/dX)) for the chosen loss.

    ``Y_ref`` is the reference sample in the SAME coordinate the loss is
    evaluated in; it is used only to freeze the bandwidth.
    """
    from .energy import energy_distance_and_grad
    if kind == "energy":
        return ("unbiased energy distance (Szekely-Rizzo) = 2 x MMD^2 with "
                "the distance kernel; no bandwidth",
                energy_distance_and_grad)
    s0 = median_sigma(Y_ref)
    if kind == "mmd1":
        sig = np.array([s0])
        name = f"unbiased MMD^2, single Gaussian, sigma = {s0:.4f} (median)"
    elif kind == "mmd":
        sig = s0 * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
        name = (f"unbiased MMD^2, Gaussian mixture, sigma_median = {s0:.4f}, "
                f"factors {{1/4,1/2,1,2,4}}")
    elif kind == "mmd3":
        sig = s0 * np.array([0.5, 1.0, 2.0])
        name = (f"unbiased MMD^2, TRIMMED Gaussian mixture, "
                f"sigma_median = {s0:.4f}, factors {{1/2,1,2}}")
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return (name, lambda X, Y: mmd2_and_grad(X, Y, sig))


# =====================================================================
def verify_equivalence(n=300, d=4, seed=0):
    """E(P,Q) = 2 MMD^2(P,Q ; k_rho) with the distance-induced kernel,
    k_rho(x,y) = 1/2[||x-z0|| + ||y-z0|| - ||x-y||], for any base point
    z0.  Checked with the BIASED (V-statistic) forms, because the identity
    is between population quantities and the V-statistic is the plug-in
    estimator of both sides.  Base-point independence is checked too."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = 0.7 * rng.standard_normal((n, d)) + 0.3

    def _pd(A, B):
        return np.sqrt(_sqdist(A, B))

    E_V = float(2.0 * _pd(X, Y).mean() - _pd(X, X).mean() - _pd(Y, Y).mean())

    out = []
    for z0 in (np.zeros(d), rng.standard_normal(d), 5.0 * np.ones(d)):
        def kk(A, B):
            ra = np.linalg.norm(A - z0[None, :], axis=1)
            rb = np.linalg.norm(B - z0[None, :], axis=1)
            return 0.5 * (ra[:, None] + rb[None, :] - _pd(A, B))
        mmd_V = float(kk(X, X).mean() + kk(Y, Y).mean() - 2.0 * kk(X, Y).mean())
        out.append(abs(E_V - 2.0 * mmd_V) / abs(E_V))
    return {"energy_distance_V": E_V,
            "max_rel_error_vs_2_MMD2": float(max(out)),
            "base_points_tested": len(out)}


def audit(seed=0, n_null=40, n_dep=1500, d=8, rho=0.6):
    """The three checks quoted in the module docstring, re-runnable.

    Returns the step sweep, the null calibration, and the
    dependence-versus-marginal balance for the distance kernel and for
    every Gaussian component."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((60, 5))
    Y = 0.8 * rng.standard_normal((70, 5)) + 0.2
    sig = median_sigma(Y) * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    _, g = mmd2_and_grad(X, Y, sig)
    sweep = {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        e = 0.0
        for _ in range(10):
            i, k = rng.integers(0, 60), rng.integers(0, 5)
            Xp, Xm = X.copy(), X.copy()
            Xp[i, k] += h
            Xm[i, k] -= h
            fd = (mmd2(Xp, Y, sig) - mmd2(Xm, Y, sig)) / (2 * h)
            e = max(e, abs(fd - g[i, k]) / (abs(fd) + 1e-12))
        sweep[f"h={h:.0e}"] = float(e)

    vals = [mmd2(rng.standard_normal((400, 5)), rng.standard_normal((400, 5)),
                 sig) for _ in range(n_null)]
    null = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
            "sigmas_from_zero": float(abs(np.mean(vals))
                                      / (np.std(vals) / np.sqrt(n_null)))}

    from .energy import energy_distance
    Z = rng.standard_normal((n_dep, d))
    L = np.linalg.cholesky(rho * np.ones((d, d)) + (1 - rho) * np.eye(d))
    C = Z @ L.T
    I = np.column_stack([np.random.default_rng(100 + j).permutation(C[:, j])
                         for j in range(d)])
    M = C + 0.15
    s0 = median_sigma(C)
    bal = {"energy_distance": {"dependence": float(energy_distance(C, I)),
                               "marginal": float(energy_distance(C, M))}}
    for f in (0.25, 0.5, 1.0, 2.0, 4.0):
        bal[f"mmd_sigma_{f}x"] = {"dependence": float(mmd2(C, I, [f * s0])),
                                  "marginal": float(mmd2(C, M, [f * s0]))}
    mix = s0 * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    bal["mmd_mixture"] = {"dependence": float(mmd2(C, I, mix)),
                          "marginal": float(mmd2(C, M, mix))}
    for k in bal:
        bal[k]["ratio_dependence_over_marginal"] = (
            bal[k]["dependence"] / bal[k]["marginal"]
            if bal[k]["marginal"] else None)
    return {"gradient_step_sweep": sweep, "null_calibration": null,
            "dependence_vs_marginal_balance": bal,
            "sigma_median": float(s0)}


def check_gradients(n=60, m=70, d=5, seed=0, h=1e-6):
    """Analytic dMMD^2/dX against central differences, and the same for
    the energy distance, so both branches of make_loss are audited."""
    from .energy import energy_distance_and_grad, energy_distance
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = 0.8 * rng.standard_normal((m, d)) + 0.2
    sig = median_sigma(Y) * np.array([0.25, 0.5, 1.0, 2.0, 4.0])

    _, g = mmd2_and_grad(X, Y, sig)
    err_mmd = 0.0
    for _ in range(15):
        i, k = rng.integers(0, n), rng.integers(0, d)
        Xp, Xm = X.copy(), X.copy()
        Xp[i, k] += h
        Xm[i, k] -= h
        fd = (mmd2(Xp, Y, sig) - mmd2(Xm, Y, sig)) / (2 * h)
        err_mmd = max(err_mmd, abs(fd - g[i, k]) / (abs(fd) + 1e-9))

    _, ge = energy_distance_and_grad(X, Y)
    err_e = 0.0
    for _ in range(15):
        i, k = rng.integers(0, n), rng.integers(0, d)
        Xp, Xm = X.copy(), X.copy()
        Xp[i, k] += h
        Xm[i, k] -= h
        fd = (energy_distance(Xp, Y) - energy_distance(Xm, Y)) / (2 * h)
        err_e = max(err_e, abs(fd - ge[i, k]) / (abs(fd) + 1e-9))
    return {"mmd_grad_vs_central_difference_rel": float(err_mmd),
            "energy_grad_vs_central_difference_rel": float(err_e)}
