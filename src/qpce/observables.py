"""
qpce.observables
==================

Measured-observable specification: one single-body Z per qubit, extracted
from ONE computational-basis setting.

    O = { Z_i }_{i=0}^{n-1}

All operators commute and are simultaneously measurable from one shot, so a
generated shower costs one circuit execution and S shots, independent of n.
The published QPCE measured 72 Pauli operators across two settings; QPCE-F
needs exactly n scalars because the readout target is one intensity per
pixel and there is no feature bank to feed.

Pauli-string convention is Qiskit little-endian (rightmost character is
qubit 0), matching the qubit-to-pixel map q_i <-> p_i.
"""
from __future__ import annotations


def readout_observables(n_qubits: int):
    obs = []
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "Z"
        obs.append("".join(s[::-1]))
    return obs


def observable_count(n_qubits: int) -> int:
    return n_qubits


def measurement_settings() -> int:
    return 1
