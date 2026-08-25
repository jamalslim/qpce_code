"""
Training loop. L-BFGS on the energy distance with exact adjoint gradients.

WHY A FROZEN GERM BATCH
-----------------------
The objective is stochastic in the germs. If germs were redrawn every
iteration, L-BFGS would be optimising a moving target and its curvature
estimates would be meaningless. So the germ batch is frozen for the fit, which
makes the objective deterministic and the quasi-Newton step well defined.

WHY THAT REQUIRES FRESH-GERM VALIDATION
---------------------------------------
The price of freezing is that the model can start fitting the particular germ
sample rather than the distribution it came from. The frozen objective then
keeps falling after the model has stopped improving. We therefore evaluate on
freshly drawn germs at intervals, keep the best-validation iterate, and stop
when it stops improving. Reporting the frozen-batch loss as a result would be
a mistake, and the code does not do it.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize

from .energy import energy_distance
from .mmd import make_loss
from .adjoint import grad_scalar


class EnergyTrainer:
    """L-BFGS on ONE scalar loss, with exact adjoint gradients.

    ``kind`` selects the kernel, not the number of terms:

        "energy"  the distance kernel.  E = 2 MMD^2 exactly (Sejdinovic
                  et al., Ann. Statist. 41, 2263 (2013), Thm 22; verified
                  to 1.9e-14 in qpce.mmd.verify_equivalence).  No
                  bandwidth.
        "mmd"     Gaussian mixture, sigma factors {1/4,1/2,1,2,4}
        "mmd3"    TRIMMED Gaussian mixture {1/2,1,2} -- measured better
        "mmd1"    single Gaussian at the median heuristic

    It is still ONE loss.  Changing the kernel does not add a term, a
    weight or a schedule.
    """

    def __init__(self, circuit, readout, standardiser, cfg, verbose=True,
                 kind=None):
        self.circ = circuit
        self.readout = readout
        self.std = standardiser
        self.cfg = cfg
        self.verbose = verbose
        self.kind = kind or getattr(cfg, "loss", "energy")
        self.loss_name = None
        self.loss_fn = None
        self.history = []

    # ---- loss coordinate ---------------------------------------------
    def _forward(self, E):
        """germs -> z-scored intensities, plus d(coordinate)/d<Z>."""
        Z = self.circ.expectations(E)
        Yz = self.std(self.readout.to_intensity(Z))
        dY_dZ = self.readout.s[None, :] * self.std.scale()[None, :]
        return Z, Yz, dY_dZ

    def _ensure_loss(self, target):
        if self.loss_fn is None:
            self.loss_name, self.loss_fn = make_loss(self.kind, target)
        return self.loss_fn

    def loss(self, theta, E, target):
        self.circ.unpack(theta)
        return self._ensure_loss(target)(self._forward(E)[1], target)[0]

    def loss_and_grad(self, theta, E, target):
        self.circ.unpack(theta)
        Z, Yz, dY_dZ = self._forward(E)
        val, dY = self._ensure_loss(target)(Yz, target)
        dZ = dY * dY_dZ
        # EXACT adjoint: three state evolutions for the whole gradient,
        # independent of the parameter count.  Parameter shift costs
        # 2 n_params circuit evaluations and is what kept every earlier
        # fit at 60 unconverged iterations.  Agreement: 2.4e-16.
        _, g = grad_scalar(self.circ, E, dZ)
        self.history.append(float(val))
        return val, g

    # ---- validation on FRESH germs ------------------------------------
    @staticmethod
    def null_floor(target, n_rep=60, seed=0):
        """Standard deviation of the training statistic under the NULL.

        The loss is an UNBIASED U-statistic for a quantity that is >= 0, so
        when the two samples agree it lands below zero about half the time.
        That is correct and expected.  What is NOT informative is a value far
        below the null spread: since the germ batch is frozen (common random
        numbers, see the module docstring), L-BFGS can drive the statistic
        arbitrarily negative by fitting that one batch's sampling noise.

        Measured: minimising this same statistic with only 16 parameters
        against a frozen 700-point target reaches -0.005757 on the frozen
        batch and -0.000035 on fresh germs -- 0.0057 of pure memorisation.
        A 138-parameter circuit can do at least as well.

        This returns the scale below which the training curve stops carrying
        information, so the trainer can say so instead of printing a number
        that looks like progress.
        """
        rng = np.random.default_rng(seed)
        m = len(target) // 2
        v = []
        for _ in range(n_rep):
            i = rng.permutation(len(target))
            v.append(energy_distance(target[i[:m]], target[i[m:2 * m]]))
        return float(np.std(v))

    def validate(self, theta, n_valid, target, seed):
        """Loss on FRESH germs -- the only number that measures the model
        rather than the optimiser's grip on one germ draw."""
        keep = self.circ.pack().copy()
        self.circ.unpack(theta)
        E = np.random.default_rng(seed).uniform(0, 1,
                                                (n_valid, self.circ.n_wires))
        v = self.loss(theta, E, target)
        self.circ.unpack(keep)
        return float(v)

    # ---- fit ----------------------------------------------------------
    def fit(self, Y_train, maxiter=None, batch=None, data_batch=None,
            seed=None, validate_every=25, patience=8, n_valid=None,
            early_stop=True):
        """Fit by L-BFGS on a frozen germ batch, with FRESH-GERM validation.

        validate_every : evaluate on fresh germs every this many iterations
        patience       : stop after this many validations without improvement
        n_valid        : fresh germs per validation (default: same as batch)
        early_stop     : if False, run to maxiter but still record validation

        WHY THIS EXISTS.  The germ batch is frozen for common random numbers,
        which is what makes L-BFGS usable at all -- but it also means the
        training curve keeps decreasing long after the model has stopped
        improving, because the optimiser is minimising that batch's sampling
        noise.  The training loss going NEGATIVE is normal (the statistic is
        an unbiased estimator of a non-negative quantity); going far below the
        null spread is memorisation.  Without a fresh-germ number there is no
        signal to stop on, and a 1800-iteration run reports a smooth,
        monotone, entirely misleading curve.

        The returned dict carries `loss_valid_best` and `stopped_early`; the
        restored parameters are those of the BEST validation, not the last
        iterate.
        """
        cfg = self.cfg
        maxiter = cfg.maxiter if maxiter is None else maxiter
        batch = cfg.batch if batch is None else batch
        data_batch = cfg.data_batch if data_batch is None else data_batch
        seed = cfg.seed if seed is None else seed

        rng = np.random.default_rng(seed + 100)
        E = rng.uniform(0, 1, (batch, self.circ.n_wires))          # CRN
        sub = Y_train[rng.choice(len(Y_train),
                                 min(data_batch, len(Y_train)), replace=False)]
        target = self.std(sub)

        n_valid = batch if n_valid is None else int(n_valid)
        floor = self.null_floor(target, seed=seed) if self.kind == "energy" else None

        theta0 = self.circ.pack().copy()
        e0 = self.loss(theta0, E, target)
        if self.verbose:
            print(f"loss: {self.loss_name}", flush=True)
            print(f"initial loss ({self.kind}) {e0:.6f}", flush=True)
            if floor is not None:
                print(f"noise floor: the training statistic has null spread "
                      f"+-{floor:.6f} at B={batch}, M={len(target)}.", flush=True)
                print(f"             values below about -{floor:.6f} are the "
                      f"optimiser fitting this batch's noise, not the data. "
                      f"Watch VALID, not E.", flush=True)

        t0 = time.time()
        state = {"it": 0, "best": np.inf, "best_x": theta0.copy(),
                 "best_it": 0, "bad": 0, "stop": False, "curve": []}

        class _Converged(Exception):
            pass

        def cb(xk):
            state["it"] += 1
            it = state["it"]
            msg = (f"  [L-BFGS] iter {it:4d}  E = {self.history[-1]:.6f}"
                   f"   ({time.time() - t0:.0f}s)")
            if it % validate_every == 0 or it == 1:
                v = self.validate(xk, n_valid, target, seed + 900 + it)
                state["curve"].append((it, float(self.history[-1]), v))
                flag = ""
                if v < state["best"] - 1e-9:
                    state["best"], state["bad"] = v, 0
                    state["best_x"] = np.array(xk, copy=True)
                    state["best_it"] = it
                    flag = "  *best*"
                else:
                    state["bad"] += 1
                    flag = f"  (no gain x{state['bad']})"
                gap = float(self.history[-1]) - v
                msg += f"   VALID = {v:.6f}{flag}"
                if floor is not None and gap > 2 * floor:
                    msg += f"   [train-valid gap {gap:+.6f} > 2x noise floor]"
                if self.verbose:
                    print(msg, flush=True)
                if early_stop and state["bad"] >= patience:
                    state["stop"] = True
                    if self.verbose:
                        print(f"  [early stop] no fresh-germ improvement for "
                              f"{patience} checks; best was iteration "
                              f"{state['best_it']} at VALID = "
                              f"{state['best']:.6f}.", flush=True)
                    raise _Converged
                return
            if self.verbose:
                print(msg, flush=True)

        try:
            res = minimize(self.loss_and_grad, theta0, args=(E, target),
                           jac=True, method="L-BFGS-B", callback=cb,
                           options={"maxiter": maxiter,
                                    "maxcor": cfg.lbfgs_maxcor,
                                    "ftol": cfg.lbfgs_ftol,
                                    "gtol": cfg.lbfgs_gtol})
            x_final, nit, nfev = res.x, int(res.nit), int(res.nfev)
        except _Converged:
            x_final, nit, nfev = state["best_x"], state["it"], len(self.history)

        # Restore the BEST fresh-germ iterate, not the last one.
        v_last = self.validate(x_final, n_valid, target, seed + 12345)
        if state["best_it"] and state["best"] < v_last - 1e-9:
            if self.verbose:
                print(f"  [restore] last iterate VALID = {v_last:.6f}; "
                      f"restoring iteration {state['best_it']} "
                      f"(VALID = {state['best']:.6f}).", flush=True)
            x_final = state["best_x"]
        self.circ.unpack(x_final)
        self.circ.project()
        final = self.loss(self.circ.pack(), E, target)
        v_final = self.validate(self.circ.pack(), n_valid, target, seed + 777)
        secs = time.time() - t0
        if self.verbose:
            print(f"final loss ({self.kind})   train {final:.6f}   "
                  f"FRESH-GERM {v_final:.6f}   "
                  f"({nit} iterations, {nfev} evaluations, {secs:.0f}s)",
                  flush=True)
            if floor is not None and final < -2 * floor:
                print(f"  NOTE: the training loss sits {abs(final)/floor:.1f}x "
                      f"the null spread below zero. That is batch memorisation, "
                      f"not fit quality; quote the FRESH-GERM number.",
                      flush=True)
        # NOTE: loss_initial/loss_final are values of the TRAINING loss.
        # With kind != "energy" they are MMD^2 values, NOT energy distances,
        # and are not comparable across kinds.  The cross-comparable number
        # is the energy distance against the held-out test set, reported by
        # scripts/train.py under diagnostics.
        return {"loss_kind": self.kind, "loss_name": self.loss_name,
                "loss_initial": float(e0), "loss_final": float(final),
                "loss_valid_final": float(v_final),
                "loss_valid_best": (float(state["best"])
                                    if np.isfinite(state["best"]) else None),
                "best_iteration": int(state["best_it"]),
                "stopped_early": bool(state["stop"]),
                "null_floor": floor,
                "validation_curve": state["curve"],
                "iterations": int(nit), "evaluations": int(nfev),
                "seconds": round(secs, 1), "history": self.history}


def check_gradient(circuit, readout, standardiser, cfg, n_check=8, seed=0,
                   h=1e-5):
    """Verify the exact gradient against central differences on a few
    coordinates.  Called by scripts/selftest.py."""
    rng = np.random.default_rng(seed)
    E = rng.uniform(0, 1, (60, circuit.n_wires))
    target = standardiser(rng.uniform(0.05, 0.4, (60, circuit.n)))
    tr = EnergyTrainer(circuit, readout, standardiser, cfg, verbose=False,
                       kind="energy")
    v0 = circuit.pack().copy()
    _, g = tr.loss_and_grad(v0, E, target)
    idx = np.linspace(0, len(v0) - 1, min(n_check, len(v0))).astype(int)
    err = 0.0
    for k in idx:
        vp, vm = v0.copy(), v0.copy()
        vp[k] += h
        vm[k] -= h
        fd = (tr.loss(vp, E, target) - tr.loss(vm, E, target)) / (2 * h)
        err = max(err, abs(fd - g[k]) / (abs(fd) + 1e-12))
    circuit.unpack(v0)
    return float(err)
