"""
Configuration and coupler topologies, with the hardware cost of each.

The coupler graph is the single most consequential choice in the model. It sets
where dependence can be created, it sets the two-qubit depth through the
chromatic index of the graph, and it decides whether the circuit runs on real
hardware at all.

Two graphs matter here. The dense ring-plus-skip-2 graph is the natural first
choice, since it maximises reachable dependence at fixed depth. It is also
4-regular, and a 4-regular graph cannot be a subgraph of a lattice whose
vertices have degree at most 3, so it does not embed on present devices and
routing it inflates the depth several fold. The path is the alternative used
for deployment. It embeds natively, and being bipartite its edges colour in two
classes, so its two-qubit depth reaches the theoretical floor of 2L.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


def ring_skip2_edges(n: int) -> List[Tuple[int, int]]:
    """Circular nearest-neighbour ring plus skip-2 chords, |E| = 2n for
    n >= 5.  The ring mirrors nearest-neighbour lateral shower correlation;
    the skip-2 chords give direct couplers for next-to-nearest and
    cross-block anti-correlations a pure ring cannot reach at small L.

    Returned in ring-then-skip order.  phi is indexed by POSITION in this
    list, so a checkpoint is only meaningful together with the edge list it
    was trained with -- model.load() always restores both.
    """
    E = [(i, (i + 1) % n) for i in range(n)]
    E += [(i, (i + 2) % n) for i in range(n)]
    return [tuple(sorted(e)) for e in E]


def grid2x4_edges(n: int = 8) -> List[Tuple[int, int]]:
    """The 2x4 square-lattice block: 10 edges, the maximum any 8-qubit region
    of a square lattice can offer (verified exhaustively over all 8409
    connected 8-vertex subgraphs of a 5x5 patch).  Bipartite with Delta = 3,
    so by Koenig chi' = 3: THREE parallel RZZ layers per block, zero SWAPs.

        0 - 1 - 2 - 3
        |   |   |   |
        4 - 5 - 6 - 7
    """
    if n != 8:
        raise ValueError("grid2x4 is defined for n = 8")
    E = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [tuple(sorted(e)) for e in E]


def ladder_edges(n: int = 8) -> List[Tuple[int, int]]:
    """The same 10 edges relabelled so consecutive logical indices are rungs.
    In the CLIC 8-cell shower the intensity is unimodal in the cell index, so
    the pixel ordering is NOT exchangeable and the labelling matters."""
    if n != 8:
        raise ValueError("ladder is defined for n = 8")
    E = [(2 * k, 2 * k + 1) for k in range(4)]
    E += [(2 * k, 2 * k + 2) for k in range(3)]
    E += [(2 * k + 1, 2 * k + 3) for k in range(3)]
    return [tuple(sorted(e)) for e in E]


def ring_edges(n: int = 8) -> List[Tuple[int, int]]:
    """The n-cycle.  chi' = 2 for even n, so TWO parallel RZZ layers per
    block.  Native on a square lattice (the perimeter of the 2x4 block) but
    NOT on heavy-hex, whose girth is 12.  The causal cone of <Z_i> is +-L
    sites, which at n = 8 wraps the whole register from L = 4."""
    return [tuple(sorted((i, (i + 1) % n))) for i in range(n)]


def path_edges(n: int = 8) -> List[Tuple[int, int]]:
    """The (n-1)-edge path: the maximum realisable on heavy-hex, whose girth
    12 forces every 8-vertex subgraph to be a forest (at most 7 edges,
    verified exhaustively over all 1014 connected 8-vertex subgraphs).
    chi' = 2, so TWO parallel RZZ layers per block and zero SWAPs on Heron.
    The cone reaches +-L sites, so long-range pixel pairs need L >~ 6."""
    return [(i, i + 1) for i in range(n - 1)]


# Coupler graphs selectable with `--coupler-graph`.
#
# WHY THIS SWITCH EXISTS.  The published C8(1,2) graph is the 4-antiprism: 16
# edges, 4-regular.  chi' = 4, so its FLOOR is 4 parallel RZZ layers per block.
# But it is not a subgraph of any IBM device -- at most 10 of its 16 edges are
# native on a square lattice and at most 7 on heavy-hex -- so it must be
# routed, and routing serialises the layer it was supposed to parallelise:
# measured, 24.5 two-qubit layers per block instead of 4, a factor of 6.
#
# A graph the hardware already has compiles with ZERO SWAPs and hits chi'
# exactly.  Measured at L = 4: C8(1,2) routed is 316 pulses / 98 layers /
# 6.8 us; path8 native is 28 pulses / 8 layers / 0.7 us.
#
# Proposition 1 is untouched: it holds for ANY coupler graph, since phi = 0
# still factorises the unitary.  The certification floor is a property of the
# DATA, not of the ansatz, so every graph is scored against the same F.
COUPLER_GRAPHS = {
    "ring_skip2": ring_skip2_edges,   # 16 edges, chi'=4, published, NOT native
    "grid2x4": grid2x4_edges,         # 10 edges, chi'=3, native on square
    "ladder": ladder_edges,           # 10 edges, chi'=3, native on square
    "ring8": ring_edges,              #  8 edges, chi'=2, native on square
    "path8": path_edges,              #  7 edges, chi'=2, native on heavy-hex
}


@dataclass
class QPCEConfig:
    # ---- dataset ------------------------------------------------------
    # 8-cell CLIC calorimeter showers, desyqml/clic v1.0.0,
    # Zenodo doi:10.5281/zenodo.16027525
    data_path: str = "data/cal_shower_img_8q.npy"
    output_dir: str = "run"
    seed: int = 7
    test_size: float = 0.35          # 65/35 split: 2600 train, 1400 test
    n_total_samples: int = 4000

    # ---- circuit ------------------------------------------------------
    n_qubits: int = 8
    n_blocks: int = 4                # L: interferometer blocks AND chaos order
    # Coupler graph.  "ring_skip2" is the published C8(1,2) and requires
    # routing on every IBM device; the others are hardware-native and compile
    # with zero SWAPs.  See COUPLER_GRAPHS above.
    coupler_graph: str = "ring_skip2"
    # Defaults are the MEASURED-BEST configuration, not the published one.
    # walls="fixed", dither=True reproduces manuscript Eq. 3 exactly and is
    # one flag away; measured, it costs mean |rho_S| 0.363 against 0.601 and
    # energy distance 0.0587 against 0.0088.  The |w| >= w_min projection is
    # the expensive half: the unconstrained optimum sits at E = 0.0104 and
    # re-imposing the floor throws it to E = 0.0520.
    walls: str = "trainable"         # "fixed" = published Ry(pi/2)
    dither: bool = False
    # One extra latent fed to EVERY qubit with a trainable per-qubit
    # amplitude.  Costs n single-qubit rotations per block and ZERO two-qubit
    # gates.  See QPCECircuit.__init__ for why the CLIC data needs it.
    shared_germ: int = 0             # True = published final private dither
    w_min: float = 0.15              # dither floor, projected each step
    wf_init: float = 0.2

    # ---- readout ------------------------------------------------------
    # Y_k = m_k + s_k <Z_k>.  m, s are set from the training range with this
    # relative padding; they are unit-conversion constants, not fitted.
    readout_pad: float = 0.02

    # ---- training -----------------------------------------------------
    # ONE loss.  This selects the KERNEL, not the number of terms.
    #   "energy" distance kernel, no bandwidth (E = 2 MMD^2 exactly)
    #   "mmd"    Gaussian mixture {1/4,1/2,1,2,4} x sigma_median
    #   "mmd3"   trimmed mixture {1/2,1,2} -- measured better
    #   "mmd1"   single Gaussian at the median heuristic
    loss: str = "energy"
    batch: int = 1500                # germ draws per loss evaluation (CRN)
    data_batch: int = 1500           # training images per loss evaluation
    maxiter: int = 600               # L-BFGS iterations.  60 was NOT converged:
                                     # dependence is a ~1% perturbation of the
                                     # loss until the marginals are nearly exact
                                     # (the correlated-vs-independent energy gap
                                     # is 0.067; initial marginal mismatch ~5).
    lbfgs_maxcor: int = 15
    lbfgs_ftol: float = 1e-9
    lbfgs_gtol: float = 1e-7

    # ---- evaluation ---------------------------------------------------
    n_eval: int = 20000
    n_bootstrap: int = 300
    n_perm: int = 200                # permutation replicas for the energy test
    n_hist_bins: int = 30
    eval_seed: int = 31415

    # ---- bookkeeping --------------------------------------------------
    @property
    def edges(self) -> List[Tuple[int, int]]:
        try:
            return COUPLER_GRAPHS[self.coupler_graph](self.n_qubits)
        except KeyError:
            raise ValueError(
                f"unknown coupler_graph {self.coupler_graph!r}; "
                f"choose from {sorted(COUPLER_GRAPHS)}")

    @property
    def chromatic_index(self) -> int:
        """chi'(G): the number of PARALLEL RZZ layers per block, hence the
        two-qubit depth floor.  Every gate in the diagonal block commutes, so
        a proper edge colouring is a valid schedule."""
        E = [tuple(sorted(e)) for e in self.edges]
        for k in range(1, 9):
            col = {}

            def ok(e, c):
                return all(col.get(f) != c for f in E
                           if f != e and set(f) & set(e))

            def bt(i):
                if i == len(E):
                    return True
                for c in range(k):
                    if ok(E[i], c):
                        col[E[i]] = c
                        if bt(i + 1):
                            return True
                        del col[E[i]]
                return False

            if bt(0):
                return k
        return -1

    @property
    def n_beta(self) -> int:
        return self.n_blocks * self.n_qubits

    @property
    def n_phi(self) -> int:
        return self.n_blocks * len(self.edges)

    @property
    def n_theta(self) -> int:
        return self.n_blocks * self.n_qubits if self.walls == "trainable" else 0

    @property
    def n_dither(self) -> int:
        return 2 * self.n_qubits if self.dither else 0

    @property
    def n_quantum_parameters(self) -> int:
        return self.n_beta + self.n_phi + self.n_theta + self.n_dither

    @property
    def n_classical_trainable(self) -> int:
        return 0

    @property
    def n_classical_stored(self) -> int:
        return 2 * self.n_qubits            # the readout pair (m_k, s_k)

    @property
    def measurement_settings(self) -> int:
        return 1                            # one computational-basis setting

    @property
    def chaos_order(self) -> int:
        return self.n_blocks                # degree in each germ

    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def summary(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "blocks_L": self.n_blocks,
            "chaos_order_in_each_germ": self.chaos_order,
            "shared_germ": self.shared_germ,
            "coupler_graph": self.coupler_graph,
            "coupler_edges": len(self.edges),
            "chromatic_index": self.chromatic_index,
            "two_qubit_layers_native": self.n_blocks * self.chromatic_index,
            "loss": self.loss,
            "walls": self.walls,
            "dither": self.dither,
            "quantum_parameters": self.n_quantum_parameters,
            "classical_trainable": self.n_classical_trainable,
            "classical_stored": self.n_classical_stored,
            "measurement_settings": self.measurement_settings,
        }


def edge_colour(E, kmax=8):
    """Proper edge colouring of the coupler graph, by backtracking.

    Returns the chromatic index and the colour classes. Each class is a set of
    edges sharing no qubit, so all its RZZ gates execute in one parallel layer.
    The two-qubit depth of a block is therefore the chromatic index, not the
    number of edges, and for a path that number is two.
    """
    E = [tuple(sorted(e)) for e in E]
    for k in range(1, kmax + 1):
        col = {}

        def ok(e, c):
            return all(col.get(f) != c for f in E if f != e and set(f) & set(e))

        def bt(i):
            if i == len(E):
                return True
            for c in range(k):
                if ok(E[i], c):
                    col[E[i]] = c
                    if bt(i + 1):
                        return True
                    del col[E[i]]
            return False

        if bt(0):
            layers = [[e for e in E if col[e] == c] for c in range(k)]
            return k, [l for l in layers if l]
    raise RuntimeError("edge colouring failed")


def fidelity_estimate(two_q, eplg=6.2e-3, n=8, readout=1.2e-2, eps1q=2.5e-4,
                      one_q=4 * 3 * 8):
    """Crude circuit fidelity from gate and readout error rates.

    Multiplicative model, two-qubit gates times single-qubit gates times
    readout. It is an order-of-magnitude budget used when comparing coupler
    graphs, not a calibrated prediction of any particular device.
    """
    import math
    return (math.exp(two_q * math.log1p(-eplg))
            * math.exp(one_q * math.log1p(-eps1q))
            * math.exp(n * math.log1p(-readout)))
