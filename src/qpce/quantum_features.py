"""
The interferometric circuit, which is the whole generative model.

WHAT THIS IS
------------
One circuit turns latent randomness into calorimeter cell intensities. There
is no classical decoder anywhere after it. The latent variables, called germs
after the usual terminology of polynomial chaos, enter as rotation angles and
are re-uploaded at the head of every block. That single design choice is what
makes the model an expansion rather than a black box.

WHY RE-UPLOADING MATTERS
------------------------
Encode a germ once and every expectation value is a trigonometric polynomial
of degree one in it, no matter how deep the circuit runs afterwards. Degree
one cannot bend into a skewed calorimeter marginal. Re-upload the same germ at
each of the L blocks and the reachable degree becomes exactly L. Circuit depth
therefore plays the role that the truncation order plays in a classical chaos
expansion. We verify the claim numerically rather than trusting it, by sweeping
one germ and checking that the Fourier content above harmonic L sits at machine
precision.

BLOCK STRUCTURE
---------------
Each block is germ rotations, then a layer of single-qubit Z phases, then the
two-qubit RZZ couplers on the chosen graph, then a wall of Y rotations. The
couplers are the only place where dependence between cells can be created, and
setting their angles to zero is what the verification protocol exploits.

TWO KINDS OF GERM WIRE
----------------------
Each qubit reads its own private germ, and every qubit additionally reads one
shared germ with a trainable weight. The shared wire is a deliberate factor
model. It supplies the collective mode of a shower, which is rank one and
carries most of the correlation, using single-qubit rotations alone and no
extra two-qubit gates. Removing the couplers while leaving that wire free is
how we measure the shared channel on its own.

INDEXING CONVENTION
-------------------
Statevectors are returned with shape (batch, 2, 2, ..., 2). Qubit q lives on
axis q+1, and qubit 0 is the most significant bit of the flat index. Get this
wrong and every correlation silently transposes, so it is stated here once and
respected everywhere.
"""
from __future__ import annotations

import numpy as np


# =====================================================================
# batched statevector primitives
# qubit q lives on tensor axis 1+q; axis 0 is the batch
# =====================================================================

def zero_state(batch: int, n_qubits: int) -> np.ndarray:
    psi = np.zeros((batch,) + (2,) * n_qubits, dtype=np.complex128)
    psi[(slice(None),) + (0,) * n_qubits] = 1.0
    return psi


def _coeff(x, ndim_rest):
    x = np.asarray(x)
    return x if x.ndim == 0 else x.reshape(x.shape + (1,) * ndim_rest)


def apply_ry(psi, q, angle) -> None:
    s = np.moveaxis(psi, 1 + q, -1)
    half = np.asarray(angle) / 2.0
    c = _coeff(np.cos(half), s.ndim - 2)
    sn = _coeff(np.sin(half), s.ndim - 2)
    s0 = s[..., 0].copy()
    s1 = s[..., 1]
    s[..., 0] = c * s0 - sn * s1
    s[..., 1] = sn * s0 + c * s1


def apply_rz(psi, q, angle) -> None:
    s = np.moveaxis(psi, 1 + q, -1)
    half = np.asarray(angle) / 2.0
    s[..., 0] = _coeff(np.exp(-1j * half), s.ndim - 2) * s[..., 0]
    s[..., 1] = _coeff(np.exp(+1j * half), s.ndim - 2) * s[..., 1]


def apply_rzz(psi, i, j, angle) -> None:
    """exp(-i angle Z_i Z_j / 2)."""
    a_i, a_j = 1 + i, 1 + j
    half = np.asarray(angle) / 2.0
    ph_eq, ph_ne = np.exp(-1j * half), np.exp(+1j * half)
    for bi in (0, 1):
        for bj in (0, 1):
            sl = [slice(None)] * psi.ndim
            sl[a_i] = bi
            sl[a_j] = bj
            ph = ph_eq if bi == bj else ph_ne
            if np.ndim(ph) > 0:
                view = psi[tuple(sl)]
                psi[tuple(sl)] = view * ph.reshape((-1,) + (1,) * (view.ndim - 1))
            else:
                psi[tuple(sl)] *= ph


def expect_z_all(psi) -> np.ndarray:
    """<Z_q> for every qubit: (B, n)."""
    p = psi.real ** 2 + psi.imag ** 2
    n = psi.ndim - 1
    out = np.empty((psi.shape[0], n))
    for q in range(n):
        axes = tuple(a for a in range(1, n + 1) if a != 1 + q)
        pq = p.sum(axis=axes)
        out[:, q] = pq[:, 0] - pq[:, 1]
    return out


def bloch_vectors(psi) -> np.ndarray:
    """Reduced Bloch vector of every qubit: (B, n, 3).  Used for the
    marginal-support bound |<Z_k>| <= |r_k|."""
    n = psi.ndim - 1
    out = np.empty((psi.shape[0], n, 3))
    for q in range(n):
        s = np.moveaxis(psi, 1 + q, -1)
        ax = tuple(range(1, s.ndim - 1))
        a0, a1 = s[..., 0], s[..., 1]
        cross = (np.conj(a0) * a1).sum(axis=ax)
        p0 = (a0.real ** 2 + a0.imag ** 2).sum(axis=ax)
        p1 = (a1.real ** 2 + a1.imag ** 2).sum(axis=ax)
        out[:, q, 0] = 2.0 * cross.real
        out[:, q, 1] = 2.0 * cross.imag
        out[:, q, 2] = p0 - p1
    return out


# =====================================================================
# the circuit
# =====================================================================

class InterferometricCircuit:
    """Parameter carrier and forward map.  All parameters are gate angles;
    the classical trainable count is exactly zero.

        beta  : (L, n)     Rz phases                        [always]
        phi   : (L, |E|)   RZZ couplers                     [always]
        theta : (L, n)     mixing walls                     [walls="trainable"]
        wf,bf : (n,), (n,) final private dither             [dither=True]
    """

    def __init__(self, n_qubits, n_blocks, edges, seed=7, walls="fixed",
                 dither=True, w_min=0.15, wf_init=0.2, shared_germ=0,
                 a_init=0.15):
        rng = np.random.default_rng(seed)
        self.n = int(n_qubits)
        self.L = int(n_blocks)
        self.edges = [tuple(sorted(e)) for e in edges]
        self.walls = walls
        self.dither = bool(dither)
        self.w_min = float(w_min)
        self.beta = 0.05 * rng.standard_normal((self.L, self.n))
        self.phi = 0.05 * rng.standard_normal((self.L, len(self.edges)))
        self.theta = np.full((self.L, self.n), np.pi / 2)      # published value
        self.wf = np.full(self.n, float(wf_init))
        self.bf = np.zeros(self.n)

        # ---- SHARED GERMS + TRAINABLE PRIVATE AMPLITUDE ------------------
        #
        #     Ry( pi ( sum_m a[m,i] eps_0m  +  w[i] eps_i ) )   on qubit i,
        #                                                        every block
        #
        # `shared_germ = K` adds K extra latents eps_0m ~ U(0,1), each fed to
        # EVERY qubit with its own trainable per-qubit amplitude a[m,i].  `w`
        # is the amplitude of the PRIVATE germ, previously hard-wired to 1.
        #
        # WHY K > 1.  The CLIC rank-correlation matrix needs more than one
        # global mode:
        #     eigenvalues     5.128  1.551  0.662  0.297 ...
        #     cumulative      64.1%  83.5%  91.8%  95.5% ...
        # A single shared wire can carry only the leading mode; the second
        # (19.4 %, eigenvector -0.27 -0.18 +0.01 +0.32 +0.71 +0.51 -0.02
        # -0.15) has no carrier at all.  K = 2 reaches 83.5 % of the available
        # dependence, K = 3 reaches 91.8 %.
        #
        # WHY w IS NOW TRAINABLE.  With the private amplitude pinned at 1 the
        # model cannot turn its own noise down, so the only way to raise the
        # shared fraction is to raise a -- and the induced correlation is
        # NON-MONOTONE in a, because pi(a eps_0 + eps_i) wraps:
        #     a  = 0.3  0.5  0.8  1.2  1.8  3.0
        #     rho= .039 .066 .051 .166 .430 .364     <- peaks then falls
        # The K=1 fit ran into exactly this: it converged to a = 1.74, 1.78,
        # 1.62, past the peak, and stalled at mean |rho_S| = 0.400.  Making w
        # trainable lets the shared mode dominate by SHRINKING the private
        # part instead of over-rotating the shared one, which is the direction
        # that actually works.
        #
        # COST: (K+1) n single-qubit rotation coefficients, K extra U(0,1)
        # wires.  ZERO two-qubit gates, zero SWAPs, zero two-qubit depth.
        #
        # THIS IS NOT CLASSICAL STRUCTURE.  a and w are rotation angles on the
        # quantum circuit, in the same class as beta, theta and wf; the eps_0m
        # are independent U(0,1) noise wires re-uploaded to every qubit rather
        # than to one, and the ansatz already re-uploads n such wires.
        # Classical trainable parameters: still ZERO.
        #
        # WHAT IT DOES COST: with phi = 0 the pixels are independent only
        # CONDITIONAL on the shared wires, so Proposition 1 must be restated
        # as conditional independence.  See no_entanglement_reference.
        self.n_shared = int(shared_germ)
        self.shared_germ = self.n_shared > 0
        # PRIVATE-GERM AMPLITUDE.  Active only when shared wires are present.
        #
        # With shared wires the angle on qubit i is
        #     pi ( w_i eps_i + sum_m a_{m,i} eps_m^shared )
        # and w_i is what lets the optimiser BALANCE private against shared
        # variance.  Without it the private term is pinned at amplitude 1 and
        # the only way to raise the correlation is to raise |a|, which pushes
        # the rotation into the oscillatory regime (the K=1 fit reached
        # |a| = 1.78, i.e. 5.6 rad) where the induced correlation washes out
        # again.  w_i is therefore not cosmetic: it is the degree of freedom
        # that sets the shared variance FRACTION, which is exactly the
        # quantity the target's rank-1 mode fixes at 64 %.
        #
        # w is fixed at 1 when n_shared = 0, so every private-latent
        # checkpoint and every published number is unaffected.
        self.w = np.ones(self.n)
        if self.shared_germ:
            sgn = np.sign(rng.standard_normal((self.n_shared, self.n)))
            self.a = float(a_init) * sgn
        else:
            self.a = np.zeros((0, self.n))

    # ---- trainable groups, in pack() order ----------------------------
    @property
    def groups(self):
        g = ["beta", "phi"]
        if self.walls == "trainable":
            g.append("theta")
        if self.dither:
            g += ["wf", "bf"]
        if self.shared_germ:
            g += ["a", "w"]
        return g

    @property
    def n_params(self) -> int:
        return int(sum(getattr(self, g).size for g in self.groups))

    @property
    def n_wires(self) -> int:
        """Germ wires eps (n), dither wires eps' (n), and -- when
        shared_germ is on -- ONE extra wire eps_0 appended at the end."""
        return 2 * self.n + self.n_shared

    # ---- forward ------------------------------------------------------
    def statevector(self, E, shift=None, entanglers=True):
        """E[:, :n] are the germs eps; E[:, n:] the dither wires eps'.
        ``shift = (group, l, idx, delta)`` adds a rigid angle offset, which
        is how the exact parameter-shift gradient is taken."""
        n, L = self.n, self.L
        psi = zero_state(E.shape[0], n)
        for l in range(L):
            for i in range(n):                                 # germ, re-uploaded
                if self.shared_germ:
                    ang = np.pi * self.w[i] * E[:, i]          # private
                    for m in range(self.n_shared):             # + global modes
                        ang = ang + np.pi * self.a[m, i] * E[:, 2 * n + m]
                else:
                    ang = np.pi * E[:, i]
                # A shift on "germ" is a RIGID angle offset at block l on
                # qubit i -- the same convention as "dither".  a_i enters the
                # angle as pi a_i eps_0, so the chain rule factor pi*eps_0 is
                # applied in grad_expectations, and because a_i appears in
                # EVERY block the shifts are summed over l.  Shifting a_i
                # itself would not be a rigid offset (the induced angle change
                # is sample-dependent), which is why the naive version
                # disagreed with the adjoint.
                if shift and shift[0] == "germ" and shift[1] == l \
                        and shift[2] == i:
                    ang = ang + shift[3]
                apply_ry(psi, i, ang)
            for i in range(n):
                b = self.beta[l, i]
                if shift and shift[0] == "beta" and shift[1] == l and shift[2] == i:
                    b = b + shift[3]
                apply_rz(psi, i, b)
            if entanglers:
                for e, (i, j) in enumerate(self.edges):
                    p = self.phi[l, e]
                    if shift and shift[0] == "phi" and shift[1] == l and shift[2] == e:
                        p = p + shift[3]
                    apply_rzz(psi, i, j, p)
            for i in range(n):                                 # mixing wall
                t = self.theta[l, i]
                if shift and shift[0] == "theta" and shift[1] == l and shift[2] == i:
                    t = t + shift[3]
                apply_ry(psi, i, t)
        if self.dither:                                        # last gate
            for i in range(n):
                a = np.pi * (self.wf[i] * E[:, n + i] + self.bf[i])
                if shift and shift[0] == "dither" and shift[2] == i:
                    a = a + shift[3]
                apply_ry(psi, i, a)
        return psi

    def expectations(self, E, shift=None, entanglers=True):
        return expect_z_all(self.statevector(E, shift, entanglers))

    def no_entanglement_reference(self, E, freeze_shared=True):
        """<Z_i> with every coupler removed.

        By Proposition 1 the pixels are then exactly independent, for any
        values of the other parameters.  With a SHARED GERM that statement
        becomes conditional: independence holds given eps_0, because eps_0
        enters every qubit.  `freeze_shared=True` holds eps_0 at its median so
        the strip test measures what it is meant to measure -- residual
        coupling from the entanglers -- rather than the deliberate classical
        common factor.  Pass False to see the unconditional value, which is
        SUPPOSED to be non-zero when shared_germ is on."""
        if self.shared_germ and freeze_shared:
            E = np.array(E, copy=True)
            E[:, 2 * self.n:2 * self.n + self.n_shared] = 0.5
        return self.expectations(E, entanglers=False)

    def purity(self, E, entanglers=True):
        """Per-qubit Bloch radius |r| and linear entropy S = 1 - |r|^2."""
        r = np.linalg.norm(bloch_vectors(self.statevector(E, None, entanglers)),
                           axis=2)
        return r, np.clip(1.0 - r ** 2, 0.0, 1.0)

    # ---- packing ------------------------------------------------------
    def pack(self):
        return np.concatenate([getattr(self, g).ravel() for g in self.groups])

    def unpack(self, v):
        k = 0
        for g in self.groups:
            shape = getattr(self, g).shape
            size = int(np.prod(shape))
            setattr(self, g, v[k:k + size].reshape(shape).copy())
            k += size

    def project(self):
        """Hard projection |w_i| >= w_min after each step.  A soft penalty is
        defeated by the correlation gradient, which always favours a smaller
        dither."""
        if self.dither:
            s = np.where(self.wf >= 0, 1.0, -1.0)
            self.wf = s * np.maximum(np.abs(self.wf), self.w_min)

    # ---- exact gradients ----------------------------------------------
    def grad_expectations(self, E):
        """d<Z>/d(angle) for every trainable parameter, exact.

        Ry, Rz and RZZ all have generators with eigenvalues +-1/2, so the
        +-pi/2 parameter-shift rule is exact for every gate in this circuit.
        Cost: two circuit evaluations per parameter -- the same as the
        central finite difference it replaces, without the O(eps^2) bias
        (measured relative error 1.2e-11 against 5.7e-5).

        Returns (B, n, n_params) in pack() order.
        """
        n, L, nE = self.n, self.L, len(self.edges)
        G = np.zeros((E.shape[0], n, self.n_params))
        k = 0
        h = np.pi / 2
        for g in self.groups:
            if g == "beta":
                for l in range(L):
                    for i in range(n):
                        G[:, :, k] = 0.5 * (
                            self.expectations(E, shift=("beta", l, i, +h))
                            - self.expectations(E, shift=("beta", l, i, -h)))
                        k += 1
            elif g == "phi":
                for l in range(L):
                    for e in range(nE):
                        G[:, :, k] = 0.5 * (
                            self.expectations(E, shift=("phi", l, e, +h))
                            - self.expectations(E, shift=("phi", l, e, -h)))
                        k += 1
            elif g == "theta":
                for l in range(L):
                    for i in range(n):
                        G[:, :, k] = 0.5 * (
                            self.expectations(E, shift=("theta", l, i, +h))
                            - self.expectations(E, shift=("theta", l, i, -h)))
                        k += 1
            elif g == "wf":
                for i in range(n):
                    dm = 0.5 * (self.expectations(E, shift=("dither", 0, i, +h))
                                - self.expectations(E, shift=("dither", 0, i, -h)))
                    G[:, :, k] = dm * (np.pi * E[:, n + i:n + i + 1])
                    k += 1
            elif g == "bf":
                for i in range(n):
                    dm = 0.5 * (self.expectations(E, shift=("dither", 0, i, +h))
                                - self.expectations(E, shift=("dither", 0, i, -h)))
                    G[:, :, k] = dm * np.pi
                    k += 1
            elif g in ("a", "w"):
                # a[m,i] and w[i] both scale the SAME germ rotation on qubit i,
                # and both act in EVERY block, so the rigid germ shifts are
                # summed over l and the chain rule supplies pi*eps_0m or
                # pi*eps_i respectively.
                dm_i = []
                for i in range(n):
                    d = np.zeros((E.shape[0], n))
                    for l in range(L):
                        d += 0.5 * (
                            self.expectations(E, shift=("germ", l, i, +h))
                            - self.expectations(E, shift=("germ", l, i, -h)))
                    dm_i.append(d)
                if g == "a":
                    for m in range(self.n_shared):
                        for i in range(n):
                            G[:, :, k] = dm_i[i] * (
                                np.pi * E[:, 2 * n + m:2 * n + m + 1])
                            k += 1
                else:
                    for i in range(n):
                        G[:, :, k] = dm_i[i] * (np.pi * E[:, i:i + 1])
                        k += 1
        return G

    # ---- hardware ------------------------------------------------------
_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron1(gate, q, n):
    ops = [gate if k == q else _I2 for k in range(n)]
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def dense_reference_expectations(circuit, E):
    """Brute-force <Z_i> via dense 2^n operators, O(B 4^n).  Small n only;
    used by scripts/selftest.py to validate the batched engine."""
    from scipy.linalg import expm
    n, L = circuit.n, circuit.L

    def ry(a):
        return np.array([[np.cos(a / 2), -np.sin(a / 2)],
                         [np.sin(a / 2), np.cos(a / 2)]], complex)

    out = np.empty((E.shape[0], n))
    for b in range(E.shape[0]):
        psi = np.zeros(2 ** n, complex)
        psi[0] = 1.0
        for l in range(L):
            for i in range(n):
                ang = np.pi * circuit.w[i] * E[b, i]
                for m in range(getattr(circuit, "n_shared", 0)):
                    ang += np.pi * circuit.a[m, i] * E[b, 2 * n + m]
                psi = _kron1(ry(ang), i, n) @ psi
            for i in range(n):
                psi = _kron1(np.diag([np.exp(-1j * circuit.beta[l, i] / 2),
                                      np.exp(1j * circuit.beta[l, i] / 2)]),
                             i, n) @ psi
            for e, (i, j) in enumerate(circuit.edges):
                ZZ = _kron1(_Z, i, n) @ _kron1(_Z, j, n)
                psi = expm(-1j * circuit.phi[l, e] / 2 * ZZ) @ psi
            for i in range(n):
                psi = _kron1(ry(circuit.theta[l, i]), i, n) @ psi
        if circuit.dither:
            for i in range(n):
                a = np.pi * (circuit.wf[i] * E[b, n + i] + circuit.bf[i])
                psi = _kron1(ry(a), i, n) @ psi
        for i in range(n):
            out[b, i] = float((psi.conj() @ _kron1(_Z, i, n) @ psi).real)
    return out
