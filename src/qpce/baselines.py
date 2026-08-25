"""
qpce.baselines
================

Classical comparators.  A generative-QML paper is judged against the model a
domain expert would reach for first, and on this dataset that is a
parametric copula with empirical marginals.

Reported honestly: on the 8-cell CLIC data the Gaussian copula is hard to
beat, because the empirical dependence is close to elliptical -- the regime
in which Gaussian and Student-t copulas are near-optimal, and in which no
near-term quantum generative model should be expected to win.  A claim of
quantum benefit has to be made on non-elliptical, tail-asymmetric targets.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class _EmpiricalMarginals:
    """Inverse empirical CDF per pixel, used ONLY by the classical
    baselines.  QPCE-F itself never uses this."""

    def __init__(self, Y):
        self.sorted = np.sort(np.asarray(Y, float), axis=0)
        self.N = len(Y)
        self.pos = np.arange(1, self.N + 1) / (self.N + 1.0)

    def inverse(self, U):
        return np.column_stack([np.interp(U[:, j], self.pos, self.sorted[:, j])
                                for j in range(U.shape[1])])


class GaussianCopulaBaseline:
    """Gaussian copula, n(n-1)/2 correlation parameters, plus empirical
    marginals.  Exact ancestral sampling by Cholesky."""

    def fit(self, Y):
        self.marg = _EmpiricalMarginals(Y)
        R = np.argsort(np.argsort(Y, axis=0), axis=0) / (len(Y) + 1.0)
        Z = norm.ppf(np.clip(R, 1e-6, 1 - 1e-6))
        self.R = np.corrcoef(Z.T)
        self.L = np.linalg.cholesky(self.R)
        return self

    def sample(self, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((n_samples, self.R.shape[0])) @ self.L.T
        return self.marg.inverse(norm.cdf(g))

    @property
    def n_params(self):
        n = self.R.shape[0]
        return n * (n - 1) // 2


class IndependenceBaseline:
    """Perfect marginals, zero dependence.  The empirical realisation of the
    no-entanglement floor: it separates marginal fidelity from dependence
    fidelity in every metric."""

    def fit(self, Y):
        self.Y = np.asarray(Y, float)
        return self

    def sample(self, n_samples, seed=555):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            rng.permutation(self.Y[:, j])[:n_samples]
            for j in range(self.Y.shape[1])])

    @property
    def n_params(self):
        return 0
