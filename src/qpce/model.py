"""
qpce.model
============

End-to-end QPCE, the quantum polynomial chaos expansion.

    germs eps, eps' ~ U(0,1)^{2n}          i.i.d., no data in the circuit
      -> interferometer with germ re-uploading   (quantum_features)
      -> <Z_k>, one computational-basis setting  (observables)
      -> Y_k = m_k + s_k <Z_k>                   (data.LinearReadout)

Trained by one loss, the energy distance (energy.py, training.py).
Evaluated afterwards on fresh germs (metrics.py, certify.py).

No copula coordinate.  No probability integral transform.  No rank
recalibration.  No inverse PIT.  No classical decoder.  The only classical
numbers in the deployed model are 2n unit-conversion constants, and none of
them is fitted.
"""
from __future__ import annotations

import json
import numpy as np

from .config import QPCEConfig, ring_skip2_edges
from .quantum_features import InterferometricCircuit
from .data import LinearReadout, Standardiser
from .training import EnergyTrainer
from . import metrics as M
from .certify import CertificationReport


class QPCE:
    def __init__(self, config: QPCEConfig = None):
        self.cfg = config or QPCEConfig()
        self.circuit = None
        self.readout = None
        self.std = None
        self.trainer = None
        self.train_report = None

    # ------------------------------------------------------------------
    def build(self, Y_train, edges=None):
        cfg = self.cfg
        self.circuit = InterferometricCircuit(
            cfg.n_qubits, cfg.n_blocks, edges or cfg.edges, seed=cfg.seed,
            walls=cfg.walls, dither=cfg.dither, w_min=cfg.w_min,
            wf_init=cfg.wf_init, shared_germ=cfg.shared_germ)
        self.readout = LinearReadout(Y_train, pad=cfg.readout_pad)
        self.std = Standardiser(Y_train)
        return self

    def fit(self, Y_train, verbose=True, **kw):
        if self.circuit is None:
            self.build(Y_train)
        self.trainer = EnergyTrainer(self.circuit, self.readout, self.std,
                                     self.cfg, verbose=verbose)
        self.train_report = self.trainer.fit(Y_train, **kw)
        return self

    # ------------------------------------------------------------------
    def generate(self, n_samples, seed=None):
        """Unconditional generation: fresh germs -> circuit -> intensities.
        Nothing is resampled from the training set at generation time."""
        seed = self.cfg.eval_seed if seed is None else seed
        E = np.random.default_rng(seed).uniform(0, 1,
                                                (n_samples, self.circuit.n_wires))
        return self.readout.to_intensity(self.circuit.expectations(E))

    def evaluate(self, Y_test, n_samples=None, seed=None):
        cfg = self.cfg
        Y_gen = self.generate(n_samples or cfg.n_eval, seed)
        out = M.summarize(Y_test, Y_gen, self.std, bins=cfg.n_hist_bins,
                          n_boot=cfg.n_bootstrap, n_perm=cfg.n_perm,
                          seed=cfg.seed)
        r, S = self.circuit.purity(
            np.random.default_rng(cfg.seed).uniform(0, 1, (4000,
                                                           self.circuit.n_wires)))
        out["mean_bloch_radius"] = float(r.mean())
        out["mean_linear_entropy"] = float(S.mean())
        return out, Y_gen

    def certify(self, Y_data, Y_gen=None, verbose=True, Y_score=None):
        Y_gen = self.generate(self.cfg.n_eval) if Y_gen is None else Y_gen
        rep = CertificationReport(Y_data, self.std, seed=self.cfg.seed,
                                  Y_score=Y_score)
        out = rep.full(self.circuit, self.readout, Y_gen)
        if verbose:
            m, c = out["model"], out["controls"]
            print(f"[certify] scored {'IN-SAMPLE' if rep.in_sample else 'HELD-OUT'}")
            print(f"[certify] E {m['energy']:.4e} | floor "
                  f"{out['floor']['F']:.4e} | D {m['D']:.4f} | strip "
                  f"{out['strip_test_max_offdiag_rank_corr']:.4f}")
            print(f"[control] product-state D "
                  f"{c['product_state_surrogate']['D']:.4f} | Gaussian copula D "
                  f"{c['classical_gaussian_copula']['D']:.4f}")
        return out

    # ------------------------------------------------------------------
    def n_parameters(self):
        return {"quantum_angles": int(self.circuit.n_params),
                "classical_trainable": 0,
                "classical_stored": int(self.readout.n_stored),
                "chaos_order_per_germ": self.cfg.n_blocks,
                "coupler_edges": len(self.circuit.edges),
                "measurement_settings": 1}

    # ---- checkpoint ---------------------------------------------------
    def save(self, path):
        d = {g: getattr(self.circuit, g) for g in self.circuit.groups}
        d.update({"edges": np.array(self.circuit.edges),
                  "walls": self.circuit.walls, "dither": self.circuit.dither,
                  "theta": self.circuit.theta, "wf": self.circuit.wf,
                  "bf": self.circuit.bf,
                  "shared_germ": self.circuit.n_shared,
                  "a": self.circuit.a, "w": self.circuit.w,
                  "std_mu": self.std.mu, "std_sd": self.std.sd})
        d.update(self.readout.save())
        np.savez(path, **d)

    @classmethod
    def load(cls, path, config: QPCEConfig = None):
        ck = np.load(path, allow_pickle=True)
        cfg = config or QPCEConfig()
        edges = [tuple(int(x) for x in e) for e in ck["edges"]]
        cfg.walls = str(ck["walls"])
        cfg.dither = bool(ck["dither"])
        # back-compatible: checkpoints written before the shared germ have no
        # "a" array, and load as ordinary private-latent models.
        cfg.shared_germ = int(ck["shared_germ"]) if "shared_germ" in ck.files \
            else 0
        cfg.n_blocks = int(ck["beta"].shape[0])
        cfg.n_qubits = int(ck["beta"].shape[1])
        obj = cls(cfg)
        obj.circuit = InterferometricCircuit(cfg.n_qubits, cfg.n_blocks, edges,
                                             seed=cfg.seed, walls=cfg.walls,
                                             dither=cfg.dither,
                                             w_min=cfg.w_min,
                                             shared_germ=cfg.shared_germ)
        for g in ("beta", "phi", "theta", "wf", "bf", "a", "w"):
            if g in ck:
                setattr(obj.circuit, g, ck[g])
        obj.readout = LinearReadout.from_saved(ck)
        obj.std = Standardiser.__new__(Standardiser)
        obj.std.mu, obj.std.sd = ck["std_mu"], ck["std_sd"]
        return obj
