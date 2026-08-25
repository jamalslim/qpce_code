"""
The verification protocol. This is the part of the code that makes the paper
an experiment rather than a fit.

THE PROBLEM IT SOLVES
---------------------
A generative model that matches a target tells you nothing about which of its
components did the work. In a model with a trained classical decoder the
question cannot even be posed, because switching the circuit off leaves a
network that still fits the data. Here the model is the circuit, so the control
is a parameter setting of the model itself.

THE THREE MEASUREMENTS
----------------------
1. Independence floor. With the couplers set to zero and the shared germ held
   fixed, the cells are provably independent whatever the remaining angles.
   Any dependence measured in that configuration is finite-sample noise, and it
   gives the ceiling below which a null result must sit.

2. Couplers off, shared wire frozen. The trained model evaluated in the
   configuration of point 1. If the measured dependence does not collapse to
   the floor, something other than the couplers is creating correlation and the
   attribution claim fails.

3. Couplers off, shared wire released. This isolates the shared-latent channel,
   which is dependence the model produces with zero entangling gates. It is not
   a failure, it is the factor model working as designed, and reporting it is
   what turns the attribution into a decomposition rather than a slogan.

WHAT IT DOES NOT CLAIM
----------------------
These are fixed-basis measurements. They locate the origin of the dependence
inside the circuit. They are not an entanglement witness, because a separable
device reproducing the same expectation values would return the same numbers.
Certifying the state needs more than one measurement setting, which is what the
Bell script does.
"""
from __future__ import annotations

import numpy as np

from .energy import energy_distance
from .metrics import spearman_matrix


def independence_floor(Y_data, standardiser, n_reps=20, seed=101, split=True):
    """F = E(independence law, data), by independent column permutation.

    ``split`` scores a permuted half against the OTHER half: the U-statistic
    is unbiased only for independent samples, so shuffling a sample and
    scoring it against itself is biased.
    """
    rng = np.random.default_rng(seed)
    Z = standardiser(Y_data)
    N = len(Z)
    h = N // 2
    vals = []
    for _ in range(n_reps):
        idx = rng.permutation(N)
        A = Z[idx[:h]]
        B = Z[idx[h:2 * h]] if split else A
        A = np.column_stack([rng.permutation(A[:, j])
                             for j in range(A.shape[1])])
        vals.append(energy_distance(A, B))
    return float(np.mean(vals)), float(np.std(vals))


def effect_size(e_model, floor):
    d = 1.0 - e_model / floor if floor > 0 else np.nan
    return float(d)


def strip_test(circuit, n_samples=20000, seed=31337):
    """Largest |off-diagonal rank correlation| with every coupler removed.
    In simulation this VERIFIES Proposition 1, which is proved; its
    independent content is on hardware, where it excludes crosstalk."""
    E = np.random.default_rng(seed).uniform(0, 1, (n_samples, circuit.n_wires))
    Z = circuit.no_entanglement_reference(E)
    C = spearman_matrix(Z)
    np.fill_diagonal(C, 0.0)
    return float(np.abs(C).max())


class CertificationReport:
    def __init__(self, Y_data, standardiser, n_score=1400, seed=7,
                 Y_score=None):
        """Y_data sets the independence FLOOR; Y_score is what the model is
        scored against.

        These must be allowed to differ.  v1 used one array for both and
        train.py passed Y_train, so the headline effect size D was computed
        IN-SAMPLE.  The floor is a property of the data and is best estimated
        on the larger (training) split, but the model score has to be against
        held-out data or it is not a generalisation claim.  Pass
        Y_score=Y_test.
        """
        self.Y = Y_data
        self.Y_score = Y_data if Y_score is None else Y_score
        self.in_sample = Y_score is None
        self.std = standardiser
        self.n_score = int(n_score)
        self.seed = seed
        self.floor, self.floor_sd = independence_floor(Y_data, standardiser,
                                                       seed=seed)

    def score(self, Y_gen):
        e = energy_distance(self.std(Y_gen[:self.n_score]),
                            self.std(self.Y_score[:self.n_score]))
        return {"energy": float(e), "D": effect_size(e, self.floor),
                "sigma_below_floor": float((self.floor - e)
                                           / (self.floor_sd + 1e-15))}

    # ---- controls -----------------------------------------------------
    def product_state_control(self, circuit, readout, n_samples=20000, seed=9):
        """A separable device with classically precomputed single-qubit
        angles reproduces <Z> exactly, hence every score."""
        E = np.random.default_rng(seed).uniform(0, 1, (n_samples,
                                                       circuit.n_wires))
        Z = circuit.expectations(E)
        Z_sep = np.cos(np.arccos(np.clip(Z, -1, 1)))     # product-state prep
        out = self.score(readout.to_intensity(Z_sep))
        out["max_abs_deviation_in_Z"] = float(np.abs(Z - Z_sep).max())
        out["entanglement_present"] = False
        return out

    def monotone_control(self, circuit, factors=(1.0, 0.1, 1e-4, 1e-8),
                         n_samples=20000, seed=9):
        """Rank statistics under global depolarising <Z> -> f<Z>."""
        E = np.random.default_rng(seed).uniform(0, 1, (n_samples,
                                                       circuit.n_wires))
        Z = circuit.expectations(E)
        iu = np.triu_indices(circuit.n, 1)
        base = spearman_matrix(Z)[iu]
        return {f"f={f:g}": {
            "max_rank_corr_deviation": float(np.abs(
                spearman_matrix(f * Z)[iu] - base).max()),
            "entanglement_present": False} for f in factors}

    def classical_control(self, n_samples=20000, seed=0):
        """Gaussian copula with the data's own marginals: no quantum
        resource, and the number the model has to beat."""
        from .baselines import GaussianCopulaBaseline
        gc = GaussianCopulaBaseline().fit(self.Y)
        return {**self.score(gc.sample(n_samples, seed=seed)),
                "n_params": gc.n_params}

    def full(self, circuit, readout, Y_gen):
        return {
            "floor": {"F": self.floor, "sd": self.floor_sd,
                      "estimator": "split-half column permutation, "
                                   "non-overlapping"},
            "model": self.score(Y_gen),
            "strip_test_max_offdiag_rank_corr": strip_test(circuit),
            "controls": {
                "product_state_surrogate":
                    self.product_state_control(circuit, readout),
                "monotone_depolarising": self.monotone_control(circuit),
                "classical_gaussian_copula": self.classical_control(),
            },
            "interpretation": (
                "D measures the fraction of available dependence captured "
                "within this model family.  It is invariant under monotone "
                "per-pixel distortion, is reproduced exactly by a separable "
                "product-state device, and is exceeded by a parametric "
                "classical copula.  It is not an entanglement witness."),
        }
