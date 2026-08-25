"""
Dataset loading and the only classical constants the deployed model holds.

Each cell needs two numbers to carry a dimensionless expectation value, which
lives in [-1, 1], into physical intensity units. Those 2n constants are an
offset and a scale computed once from the training split. They are not fitted,
they are not updated during training, and they are the complete inventory of
classical numbers inside the generator.

The split is a shuffled 65/35 train/test partition at a fixed seed. Everything
reported in the paper is computed on the held-out part, on germs freshly drawn
after training, so no number is quoted on data the optimiser could see.
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(path: str, n_total_samples: int, seed: int):
    """Load the 8-cell CLIC shower dataset (Zenodo 10.5281/zenodo.16027525).

    A missing file is a HARD ERROR.  Silently substituting a synthetic
    surrogate would let a whole study train on the wrong distribution with
    only a log line to warn you.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            "Get cal_shower_img_8q.npy from Zenodo doi:10.5281/zenodo.16027525 "
            "and place it at data/cal_shower_img_8q.npy, or pass --data PATH."
        )
    rng = np.random.default_rng(seed)
    X_full = np.load(path, allow_pickle=True)
    n_pixels = min(8, X_full.shape[1])
    n_use = min(n_total_samples, X_full.shape[0])
    idx = rng.choice(X_full.shape[0], size=n_use, replace=False)
    return X_full[idx, :n_pixels].astype(float), path


def make_train_test(X_all: np.ndarray, test_size: float, seed: int):
    """65/35 split.  No transform of any kind is fitted here."""
    idx = np.arange(X_all.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return {"Y_train": X_all[tr], "Y_test": X_all[te],
            "idx_train": tr, "idx_test": te}


class LinearReadout:
    """Y_k = m_k + s_k <Z_k>.  Two stored numbers per pixel, zero fitted."""

    def __init__(self, Y_train: np.ndarray, pad: float = 0.02):
        lo, hi = Y_train.min(axis=0), Y_train.max(axis=0)
        span = hi - lo
        self.lo = lo - pad * span
        self.hi = hi + pad * span
        self.m = 0.5 * (self.lo + self.hi)
        self.s = 0.5 * (self.hi - self.lo)

    def to_intensity(self, Z):
        return self.m + self.s * Z

    def d_intensity_d_Z(self, Z):
        return np.broadcast_to(self.s[None, :], np.shape(Z))

    @property
    def n_stored(self) -> int:
        return int(2 * len(self.m))

    def save(self):
        return {"readout_m": self.m, "readout_s": self.s}

    @classmethod
    def from_saved(cls, d):
        obj = cls.__new__(cls)
        obj.m, obj.s = np.asarray(d["readout_m"]), np.asarray(d["readout_s"])
        obj.lo, obj.hi = obj.m - obj.s, obj.m + obj.s
        return obj


class Standardiser:
    """z-scoring used INSIDE the loss so that no pixel dominates the energy
    distance by having a larger dynamic range.  Two more stored numbers per
    pixel; it never touches the generated images, only the loss coordinate."""

    def __init__(self, Y_train: np.ndarray):
        self.mu = Y_train.mean(axis=0)
        self.sd = Y_train.std(axis=0)

    def __call__(self, Y):
        return (Y - self.mu) / self.sd

    def scale(self):
        return 1.0 / self.sd
