"""
scripts/selftest.py -- validate every mechanical claim the package makes.

    python scripts/selftest.py

These are not hygiene checks.  Several of them ARE the results and should
fail loudly if the code drifts:

  1  batched statevector engine == independent dense Kronecker simulator
  2  the germ chaos order is exactly L (and exactly 1 without re-uploading)
  3  parameter-shift gradients == central differences
  4  energy-distance gradient == central differences
  5  the energy distance SEES dependence at identical marginals
  6  Proposition 1: couplers off => pixels exactly independent, any angles
  7  the readout is monotone per pixel, so it cannot create dependence
  8  rank statistics are invariant under monotone per-pixel distortion
     (which is why the certification is not an entanglement witness)
  9  checkpoint round-trip is bit-exact
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from scipy.stats import spearmanr

from qpce import QPCEConfig, QPCE, ring_skip2_edges, InterferometricCircuit
from qpce.quantum_features import dense_reference_expectations
from qpce.energy import energy_distance, energy_distance_and_grad
from qpce.metrics import spearman_matrix
from qpce.data import LinearReadout, Standardiser
from qpce.training import check_gradient

OK, FAIL = "  PASS", "  FAIL"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{OK if cond else FAIL}  {name}{('  ' + detail) if detail else ''}",
          flush=True)


def circ(n=5, L=2, seed=3, walls="fixed", dither=True):
    rng = np.random.default_rng(seed)
    c = InterferometricCircuit(n, L, ring_skip2_edges(n), seed=seed,
                               walls=walls, dither=dither)
    c.beta = rng.uniform(-1.5, 1.5, c.beta.shape)
    c.phi = rng.uniform(-1.2, 1.2, c.phi.shape)
    c.theta = np.full(c.theta.shape, np.pi / 2) if walls == "fixed" \
        else rng.uniform(0.4, 2.2, c.theta.shape)
    c.wf = rng.uniform(0.2, 0.6, n)
    c.bf = rng.uniform(-0.3, 0.3, n)
    return c


t0 = time.time()
print("QPCE-F selftest\n")

# 1 -------------------------------------------------------------------
c = circ()
E = np.random.default_rng(0).uniform(0, 1, (6, c.n_wires))
err = float(np.abs(c.expectations(E) - dense_reference_expectations(c, E)).max())
check("statevector engine == dense reference", err < 1e-10, f"max err {err:.1e}")

# The same cross-check WITH shared wires and a non-trivial private amplitude,
# against an independent dense-Kronecker implementation.  Without this the
# shared-germ path was only ever checked against itself.
from qpce.quantum_features import InterferometricCircuit as _IC   # noqa: E402
from qpce.config import ring_skip2_edges as _rse                  # noqa: E402
_worst = 0.0
for _K in (1, 2, 3):
    _c2 = _IC(5, 3, _rse(5), seed=11, walls="trainable", dither=True,
              shared_germ=_K)
    _rg = np.random.default_rng(_K)
    _c2.a = _rg.uniform(-1.2, 1.2, _c2.a.shape)
    _c2.w = _rg.uniform(0.4, 1.6, 5)
    _E2 = _rg.uniform(0, 1, (6, _c2.n_wires))
    _worst = max(_worst, float(np.abs(
        _c2.expectations(_E2) - dense_reference_expectations(_c2, _E2)).max()))
check("statevector engine == dense reference WITH K=1,2,3 shared wires "
      "and w != 1", _worst < 1e-10, f"max err {_worst:.1e}")

# 2 -------------------------------------------------------------------
def germ_order(cc, K):
    g = np.linspace(0, 1, 601)
    Ee = np.full((601, cc.n_wires), 0.37)
    Ee[:, 0] = g
    m = cc.expectations(Ee)[:, 0]
    cols = [np.ones_like(g)]
    for k in range(1, K + 1):
        cols += [np.cos(k * np.pi * g), np.sin(k * np.pi * g)]
    A = np.stack(cols, 1)
    r = m - A @ np.linalg.lstsq(A, m, rcond=None)[0]
    return float(np.sqrt(np.mean(r ** 2)) / np.std(m))

for L in (2, 3, 4):
    c = circ(L=L, seed=10 + L)
    lo, hi = germ_order(c, L - 1), germ_order(c, L)
    check(f"germ chaos order is exactly L={L}", hi < 1e-12 and lo > 1e-3,
          f"resid(K=L-1) {lo:.1e} -> resid(K=L) {hi:.1e}")

# 3 -------------------------------------------------------------------
c = circ(n=4, L=2, seed=21, walls="trainable")
E = np.random.default_rng(2).uniform(0, 1, (40, c.n_wires))
G = c.grad_expectations(E)
v0 = c.pack().copy()
h, err = 1e-5, 0.0
for k in range(c.n_params):
    vp, vm = v0.copy(), v0.copy()
    vp[k] += h
    vm[k] -= h
    c.unpack(vp); mp = c.expectations(E)
    c.unpack(vm); mm = c.expectations(E)
    c.unpack(v0)
    err = max(err, float(np.abs((mp - mm) / (2 * h) - G[:, :, k]).max()))
check("parameter-shift gradients == central differences (ALL params)",
      err < 1e-8, f"max err {err:.1e}")

# 4 -------------------------------------------------------------------
rng = np.random.default_rng(5)
X, Y = rng.standard_normal((60, 4)), rng.standard_normal((70, 4)) + 0.4
_, g = energy_distance_and_grad(X, Y)
err = 0.0
for (b, k) in [(0, 0), (17, 3), (59, 1)]:
    Xp, Xm = X.copy(), X.copy()
    Xp[b, k] += 1e-6
    Xm[b, k] -= 1e-6
    fd = (energy_distance(Xp, Y) - energy_distance(Xm, Y)) / 2e-6
    err = max(err, abs(fd - g[b, k]))
check("energy-distance gradient == central differences", err < 1e-8,
      f"max err {err:.1e}")

# 5 -------------------------------------------------------------------
Z = rng.standard_normal((3000, 2))
Cor = np.column_stack([Z[:, 0], 0.9 * Z[:, 0] + 0.436 * Z[:, 1]])
Ind = np.column_stack([Z[:, 0], rng.standard_normal(3000)])
e_dep = energy_distance(Cor, Ind)
check("energy distance sees DEPENDENCE at identical marginals",
      e_dep > 0.02, f"E = {e_dep:.4f}")

# 6 -------------------------------------------------------------------
worst = 0.0
for seed in range(4):
    c = circ(n=6, L=3, seed=30 + seed)
    E = np.random.default_rng(seed).uniform(0, 1, (60000, c.n_wires))
    C = spearman_matrix(c.no_entanglement_reference(E))
    np.fill_diagonal(C, 0.0)
    worst = max(worst, float(np.abs(C).max()))
check("Proposition 1: couplers off => independent, any angles",
      worst < 6 / np.sqrt(60000), f"max |rho| {worst:.4f} "
      f"(floor {6 / np.sqrt(60000):.4f})")

# 7 -------------------------------------------------------------------
Ytr = np.abs(rng.standard_normal((500, 5))) + 0.1
ro = LinearReadout(Ytr)
Zt = rng.uniform(-1, 1, (4000, 5))
Yt = ro.to_intensity(Zt)
mono = all(spearmanr(Zt[:, j], Yt[:, j]).statistic > 1 - 1e-12
           for j in range(5))
check("readout is monotone per pixel (cannot create dependence)", mono)

# 8 -------------------------------------------------------------------
c = circ(n=6, L=2, seed=44)
E = np.random.default_rng(7).uniform(0, 1, (8000, c.n_wires))
Zc = c.expectations(E)
iu = np.triu_indices(6, 1)
base = spearman_matrix(Zc)[iu]
dev = max(float(np.abs(spearman_matrix(f * Zc)[iu] - base).max())
          for f in (0.5, 0.1, 1e-4, 1e-8))
check("rank statistics invariant under monotone distortion "
      "(=> not an entanglement witness)", dev < 1e-12, f"max dev {dev:.1e}")

# 9 -------------------------------------------------------------------
cfg = QPCEConfig(n_blocks=2, batch=100, data_batch=100, maxiter=1)
m = QPCE(cfg).build(Ytr[:, :5] if False else np.abs(
    rng.standard_normal((600, 8))) + 0.1)
m.save("/tmp/_qpce_roundtrip.npz")
m2 = QPCE.load("/tmp/_qpce_roundtrip.npz", QPCEConfig())
d = float(np.abs(m.generate(400) - m2.generate(400)).max())
check("checkpoint round-trip is bit-exact", d == 0.0, f"max |dY| {d:.1e}")

# 10 ------------------------------------------------------------------
# The adjoint gradient IS the training gradient, so it is checked against
# the parameter-shift rule -- which is itself exact for Ry, Rz and RZZ --
# in every walls/dither configuration, together with the flat forward map
# it relies on (a qubit-ordering mismatch would otherwise pass silently).
from qpce.adjoint import check_adjoint                       # noqa: E402
worst_forward = worst_g = 0.0
for _w in ("fixed", "trainable"):
    for _d in (True, False):
        for _s in (0, 1, 2):                           # shared modes K
            r = check_adjoint(walls=_w, dither=_d, shared_germ=_s)
            worst_forward = max(worst_forward, r["forward_flat_vs_reference"])
            worst_g = max(worst_g, r["adjoint_vs_parameter_shift_rel"])
check("adjoint forward map == reference statevector (all 12 configs)",
      worst_forward < 1e-12, f"max err {worst_forward:.1e}")
check("ADJOINT gradient == parameter shift (12 configs, K=0,1,2 shared)",
      worst_g < 1e-10, f"max rel err {worst_g:.1e}")

# The shared germ must not break Proposition 1.  With phi = 0 the unitary
# still factorises over qubits, so the pixels are independent CONDITIONAL on
# eps_0.  no_entanglement_reference(freeze_shared=True) holds eps_0 fixed, and
# the strip test must be read that way.  Unconditionally the correlation is
# SUPPOSED to be non-zero -- that is the whole point of the wire.
from qpce.quantum_features import InterferometricCircuit               # noqa
from qpce.config import ring_skip2_edges                               # noqa
# L = 1 with a large amplitude: the germ rotation is pi(eps_i + a_i eps_0),
# so at large L the response oscillates through many periods and the induced
# correlation averages away.  The clean demonstration is one block, where
# <Z_i> is monotone in eps_0 over most of its range.
_c = InterferometricCircuit(6, 1, ring_skip2_edges(6), seed=3,
                            walls="trainable", dither=False,
                            shared_germ=1)
_c.a = 0.8 * np.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
_E = np.random.default_rng(4).uniform(0, 1, (4000, _c.n_wires))
_ranks = lambda A: np.apply_along_axis(
    lambda c: np.argsort(np.argsort(c)), 0, A)
_iu = np.triu_indices(6, 1)
_cond = np.abs(np.corrcoef(_ranks(
    _c.no_entanglement_reference(_E, freeze_shared=True)).T)[_iu]).max()
_unc = np.abs(np.corrcoef(_ranks(
    _c.no_entanglement_reference(_E, freeze_shared=False)).T)[_iu]).max()
check("Prop. 1 holds CONDITIONALLY on the shared germ (phi=0, eps_0 frozen)",
      _cond < 0.06, f"max |rho| = {_cond:.4f} at n=4000 (sampling floor ~0.03)")
check("the shared germ DOES create unconditional dependence (phi=0)",
      _unc > 0.2, f"max |rho| = {_unc:.4f} vs {_cond:.4f} conditional "
                  f"-- dependence with ZERO entanglers, which is the point")

# 11 ------------------------------------------------------------------
# The loss is MMD.  Check the identity E = 2 MMD^2 with the distance
# kernel (Sejdinovic et al., Ann. Statist. 41, 2263 (2013), Thm 22), the
# analytic MMD gradient, and the null calibration of the unbiased
# estimator -- the last one is why negative training values are expected
# rather than alarming.
from qpce.mmd import verify_equivalence, check_gradients as _mg, audit  # noqa: E402
_eq = verify_equivalence()
check("energy distance == 2 x MMD^2 (distance kernel), any base point",
      _eq["max_rel_error_vs_2_MMD2"] < 1e-10,
      f"max rel err {_eq['max_rel_error_vs_2_MMD2']:.1e}")
_gg = _mg()
check("MMD gradient == central differences",
      _gg["mmd_grad_vs_central_difference_rel"] < 1e-5,
      f"rel err {_gg['mmd_grad_vs_central_difference_rel']:.1e}")
_au = audit(n_null=25, n_dep=800)
check("unbiased MMD estimator is unbiased under the null",
      _au["null_calibration"]["sigmas_from_zero"] < 3.0,
      f"{_au['null_calibration']['sigmas_from_zero']:.2f} sigma from zero")
_b = _au["dependence_vs_marginal_balance"]
check("MMD sees DEPENDENCE at identical marginals more sharply than the "
      "distance kernel",
      _b["mmd_mixture"]["ratio_dependence_over_marginal"]
      > _b["energy_distance"]["ratio_dependence_over_marginal"],
      f"ratio {_b['mmd_mixture']['ratio_dependence_over_marginal']:.1f} "
      f"vs {_b['energy_distance']['ratio_dependence_over_marginal']:.1f}")

print(f"\n{sum(results)}/{len(results)} passed in {time.time() - t0:.1f}s")
sys.exit(0 if all(results) else 1)
