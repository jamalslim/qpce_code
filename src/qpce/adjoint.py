"""
Exact gradients at the cost of three state evolutions.

Training a circuit with a few hundred angles by finite differences or by the
parameter-shift rule costs work proportional to the number of parameters. The
adjoint method costs a constant number of passes instead, because the loss here
is an expectation of a diagonal observable, so a single backward sweep collects
every derivative at once.

The implementation is the standard reverse-mode construction. Evolve forward
and keep the final state, apply the observable, then walk the gates backwards
applying each inverse to both the state and the adjoint vector while reading
off one derivative per gate. Memory stays flat because we never store the
intermediate states, we recompute them by inverse evolution.

CORRECTNESS
-----------
Speed is worthless if the gradient is wrong, and a sign error here produces a
model that trains to something plausible but wrong. The self-test therefore
checks the adjoint against the parameter-shift rule, which is exact and
independent, across twelve configurations. Agreement is at the 1e-15 level.
Do not change this file without rerunning that test.
"""
from __future__ import annotations

import numpy as np

__all__ = ["grad_scalar", "check_adjoint"]


# ---------------------------------------------------------------------
#  flat batched primitives, psi : (B, 2^n), qubit 0 most significant
# ---------------------------------------------------------------------
def _view(psi, n, i):
    return psi.reshape(psi.shape[0], 1 << i, 2, 1 << (n - 1 - i))


def _ry(psi, n, i, ang, sign=+1.0):
    v = _view(psi, n, i)
    a = v[:, :, 0, :].copy()
    b = v[:, :, 1, :].copy()
    t = sign * np.asarray(ang, float) * 0.5
    c, s = np.cos(t), np.sin(t)
    if c.ndim:
        c, s = c[:, None, None], s[:, None, None]
    v[:, :, 0, :] = c * a - s * b
    v[:, :, 1, :] = s * a + c * b


def _rz(psi, n, i, ang, sign=+1.0):
    v = _view(psi, n, i)
    e = np.exp(-1j * (sign * np.asarray(ang, float) * 0.5))
    if e.ndim:
        e = e[:, None, None]
    v[:, :, 0, :] *= e
    v[:, :, 1, :] *= np.conj(e)


def _rzz(psi, n, i, j, ang, sign=+1.0):
    if i > j:
        i, j = j, i
    v = psi.reshape(psi.shape[0], 1 << i, 2, 1 << (j - i - 1), 2,
                    1 << (n - 1 - j))
    t = sign * float(ang) * 0.5
    ep, em = np.exp(-1j * t), np.exp(1j * t)
    v[:, :, 0, :, 0, :] *= ep
    v[:, :, 1, :, 1, :] *= ep
    v[:, :, 0, :, 1, :] *= em
    v[:, :, 1, :, 0, :] *= em


def _dY(psi, n, i):
    """(-i Y / 2) psi, in place."""
    v = _view(psi, n, i)
    a = v[:, :, 0, :].copy()
    b = v[:, :, 1, :].copy()
    v[:, :, 0, :] = -0.5 * b
    v[:, :, 1, :] = 0.5 * a


def _dZ(psi, n, i):
    v = _view(psi, n, i)
    v[:, :, 0, :] *= -0.5j
    v[:, :, 1, :] *= 0.5j


def _dZZ(psi, n, i, j):
    if i > j:
        i, j = j, i
    v = psi.reshape(psi.shape[0], 1 << i, 2, 1 << (j - i - 1), 2,
                    1 << (n - 1 - j))
    v[:, :, 0, :, 0, :] *= -0.5j
    v[:, :, 1, :, 1, :] *= -0.5j
    v[:, :, 0, :, 1, :] *= 0.5j
    v[:, :, 1, :, 0, :] *= 0.5j


def _z_table(n):
    idx = np.arange(1 << n)
    return np.stack([1.0 - 2.0 * ((idx >> (n - 1 - k)) & 1)
                     for k in range(n)], axis=1)


# ---------------------------------------------------------------------
def _germ_angle(c, E, i):
    """pi * (eps_i + a_i eps_0) -- the re-uploaded germ rotation on qubit i.

    With shared_germ off this is exactly the published pi*eps_i, so the
    adjoint below is unchanged for every existing checkpoint."""
    if not getattr(c, "shared_germ", False):
        return np.pi * E[:, i]
    ang = np.pi * c.w[i] * E[:, i]
    for m in range(c.n_shared):
        ang = ang + np.pi * c.a[m, i] * E[:, 2 * c.n + m]
    return ang


def _forward(c, E):
    n, L = c.n, c.L
    psi = np.zeros((E.shape[0], 1 << n), dtype=complex)
    psi[:, 0] = 1.0
    for l in range(L):
        for i in range(n):
            _ry(psi, n, i, _germ_angle(c, E, i))
        for i in range(n):
            _rz(psi, n, i, c.beta[l, i])
        for e, (i, j) in enumerate(c.edges):
            _rzz(psi, n, i, j, c.phi[l, e])
        for i in range(n):
            _ry(psi, n, i, c.theta[l, i])
    if c.dither:
        for i in range(n):
            _ry(psi, n, i, np.pi * (c.wf[i] * E[:, n + i] + c.bf[i]))
    return psi


def expectations_flat(c, E):
    psi = _forward(c, E)
    return (psi.real ** 2 + psi.imag ** 2) @ _z_table(c.n)


def grad_scalar(c, E, C):
    """Exact gradient of  Loss = sum_{b,k} C[b,k] <Z_k>_b  with respect to
    every trainable gate angle, packed in circuit.groups order.

    Returns (expectations, gradient).  Three state evolutions total.
    """
    n, L = c.n, c.L
    psi = _forward(c, E)
    Z = _z_table(n)
    expz = (psi.real ** 2 + psi.imag ** 2) @ Z
    lam = (C @ Z.T) * psi                      # |lambda> = O|psi>, O diagonal

    g = {"beta": np.zeros_like(c.beta), "phi": np.zeros_like(c.phi),
         "theta": np.zeros_like(c.theta), "wf": np.zeros_like(c.wf),
         "bf": np.zeros_like(c.bf),
         "a": np.zeros((getattr(c, "n_shared", 0), n)),
         "w": np.zeros(n)}

    if c.dither:
        for i in range(n - 1, -1, -1):
            ang = np.pi * (c.wf[i] * E[:, n + i] + c.bf[i])
            mu = psi.copy()
            _dY(mu, n, i)
            per_b = 2.0 * np.real(np.einsum("bd,bd->b", np.conj(lam), mu))
            g["wf"][i] += np.pi * float(per_b @ E[:, n + i])
            g["bf"][i] += np.pi * float(per_b.sum())
            _ry(psi, n, i, ang, sign=-1.0)
            _ry(lam, n, i, ang, sign=-1.0)

    for l in range(L - 1, -1, -1):
        for i in range(n - 1, -1, -1):                      # mixing wall
            mu = psi.copy()
            _dY(mu, n, i)
            g["theta"][l, i] += 2.0 * float(
                np.real(np.vdot(lam.ravel(), mu.ravel())))
            _ry(psi, n, i, c.theta[l, i], sign=-1.0)
            _ry(lam, n, i, c.theta[l, i], sign=-1.0)
        for e in range(len(c.edges) - 1, -1, -1):           # couplers
            i, j = c.edges[e]
            mu = psi.copy()
            _dZZ(mu, n, i, j)
            g["phi"][l, e] += 2.0 * float(
                np.real(np.vdot(lam.ravel(), mu.ravel())))
            _rzz(psi, n, i, j, c.phi[l, e], sign=-1.0)
            _rzz(lam, n, i, j, c.phi[l, e], sign=-1.0)
        for i in range(n - 1, -1, -1):                      # local phases
            mu = psi.copy()
            _dZ(mu, n, i)
            g["beta"][l, i] += 2.0 * float(
                np.real(np.vdot(lam.ravel(), mu.ravel())))
            _rz(psi, n, i, c.beta[l, i], sign=-1.0)
            _rz(lam, n, i, c.beta[l, i], sign=-1.0)
        for i in range(n - 1, -1, -1):                      # germ, re-uploaded
            # SHARED GERM.  The rotation is Ry(pi(eps_i + a_i eps_0)), so
            #     d(angle)/d a_i = pi eps_0
            # and a_i appears once PER BLOCK -- the contributions accumulate
            # over l, exactly as wf/bf accumulate over the single dither layer.
            # Same three-evolution cost as before: no extra circuit runs.
            if getattr(c, "shared_germ", False):
                mu = psi.copy()
                _dY(mu, n, i)
                per_b = 2.0 * np.real(np.einsum("bd,bd->b", np.conj(lam), mu))
                for m in range(c.n_shared):
                    g["a"][m, i] += np.pi * float(per_b @ E[:, 2 * n + m])
                g["w"][i] += np.pi * float(per_b @ E[:, i])
            ang = _germ_angle(c, E, i)
            _ry(psi, n, i, ang, sign=-1.0)
            _ry(lam, n, i, ang, sign=-1.0)

    return expz, np.concatenate([g[k].ravel() for k in c.groups])


# ---------------------------------------------------------------------
def check_adjoint(n=5, L=2, batch=13, seed=0, walls="trainable",
                  dither=True, shared_germ=0):
    """Self-audit, run by scripts/selftest.py.

    1. the flat forward map reproduces InterferometricCircuit.expectations
       to machine precision (guards against a qubit-ordering mismatch);
    2. the adjoint gradient reproduces the parameter-shift gradient, which
       is itself exact for Ry, Rz and RZZ.
    """
    from .config import ring_skip2_edges
    from .quantum_features import InterferometricCircuit
    rng = np.random.default_rng(seed)
    c = InterferometricCircuit(n, L, ring_skip2_edges(n), seed=seed,
                               walls=walls, dither=dither,
                               shared_germ=shared_germ)
    E = rng.uniform(0, 1, (batch, c.n_wires))
    C = rng.standard_normal((batch, n))

    ref = c.expectations(E)
    fwd = expectations_flat(c, E)
    r_fwd = float(np.abs(ref - fwd).max())

    _, g_adj = grad_scalar(c, E, C)
    g_ps = np.einsum("bn,bnp->p", C, c.grad_expectations(E))
    denom = np.abs(g_ps).max() + 1e-12
    r_grad = float(np.abs(g_adj - g_ps).max() / denom)
    return {"forward_flat_vs_reference": r_fwd,
            "adjoint_vs_parameter_shift_rel": r_grad,
            "n": n, "L": L, "n_params": int(c.n_params)}
