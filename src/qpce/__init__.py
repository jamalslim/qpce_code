"""
qpce, the quantum polynomial chaos expansion.

The circuit alone generates calorimeter shower images: the pixel marginal
pdfs and the pixel correlations.  No copula coordinate, no probability
integral transform, no rank recalibration, no classical decoder.

    quantum_features.py  the interferometric circuit with germ re-uploading
    energy.py            the loss, unbiased energy distance
    adjoint.py           exact gradients at three state evolutions
    training.py          L-BFGS on a frozen germ batch, fresh-germ validation
    certify.py           the verification protocol and its null controls
    metrics.py           rank statistics, tails, the dependence score
    data.py              dataset and the 2n unit-conversion constants
    config.py            coupler graphs, edge colouring, resource cost
    baselines.py         classical reference densities
    model.py             end-to-end orchestration
    observables.py       the single-setting Pauli-Z readout
    plots.py             figure helpers
    mmd.py               kernel two-sample statistics
    progress.py          optional progress bar

One circuit, one loss, one training script (scripts/train.py).
"""
from .config import (QPCEConfig, ring_skip2_edges, COUPLER_GRAPHS,  # noqa: F401
                     grid2x4_edges, ladder_edges, ring_edges, path_edges)  # noqa: F401,E501
from .quantum_features import InterferometricCircuit       # noqa: F401
from .data import load_dataset, make_train_test, LinearReadout, Standardiser  # noqa: F401,E501
from .energy import energy_distance, energy_permutation_test  # noqa: F401
from .training import EnergyTrainer                        # noqa: F401
from .model import QPCE                                   # noqa: F401
from .certify import CertificationReport                   # noqa: F401

__version__ = "1.0.0"
__all__ = ["QPCEConfig", "ring_skip2_edges", "COUPLER_GRAPHS",
           "grid2x4_edges", "ladder_edges", "ring_edges", "path_edges", "InterferometricCircuit",
           "load_dataset", "make_train_test", "LinearReadout", "Standardiser",
           "energy_distance", "energy_permutation_test", "EnergyTrainer",
           "QPCE", "CertificationReport"]
