# QPCE, the quantum polynomial chaos expansion

Reference implementation for the manuscript. One interferometric circuit is the
entire generative model. It produces calorimeter shower images, the density of
every cell and the correlations between cells together, with no classical
decoder and no fitted classical parameters.

## The model

A classical polynomial chaos expansion writes a random output as a truncated
series in the randomness that drives it, and stores the coefficients explicitly,
which is why its cost grows quickly with dimension and order. QPCE keeps the
first idea and discards the second. The randomness, called germs, is still the
only input. The coefficients are never stored, because they are the amplitudes
of one unitary circuit, so exponentially many of them are carried by a few
hundred gate angles.

Two mechanisms make this work. A germ encoded once contributes trigonometric
degree one to every expectation value, whatever the depth. Re-uploaded at the
head of each of `L` blocks it contributes exactly degree `L`, so circuit depth is
the truncation order of the expansion. Separately, every qubit reads a common
germ with a trainable weight, which supplies the collective mode of a shower,
rank one and carrying most of the correlation, using single-qubit rotations and
no additional two-qubit gates.

Readout is fixed. Cell `i` is the Pauli-Z expectation of qubit `i`, shifted and
scaled by two stored unit-conversion constants computed once from the training
split. Those `2n` numbers are the only classical numbers in the model.

## Verification

Because the model is the circuit, its control is a setting of the model itself.
`scripts/certify.py` runs three measurements.

With the couplers at zero and the shared germ frozen, the cells are provably
independent whatever the remaining angles, which gives the floor a null result
must fall below. Evaluating the trained model in that configuration must
collapse its dependence to that floor. Releasing the shared wire while the
couplers stay at zero then isolates the shared-latent channel, dependence
produced with no entangling gates at all. Reporting both makes the attribution
a decomposition rather than an assertion.

These are fixed-basis measurements, so they locate the origin of the dependence
inside the circuit rather than certifying the state. Certification of the state
needs more than one measurement setting, which is what
`scripts/bell_generative_advantage.py` provides.

## Install

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    python scripts/selftest.py

Four packages are needed. scikit-learn is load-bearing rather than incidental,
since it provides the train and test split that defines every held-out quantity.
matplotlib draws figures and affects no number. `requirements-lock.txt` pins the
exact versions the published numbers were produced with, and
`requirements-optional.txt` adds mplhep for the HEP figure style and tqdm for a
progress bar, neither of which changes a result.

## Reproducing the paper

One command.

    python scripts/reproduce.py

It recomputes the numbers from the shipped checkpoint, writes
`outputs/numbers.tex`, and compares each value against the macro table printed
in the submitted manuscript, which ships as `outputs/manuscript_reference.tex`.
It names anything that disagrees and exits non-zero if anything does. Expected
result, **67 of 67 simulation numbers reproduced exactly**. The device
measurements reported in the manuscript come from the archived hardware run and
are listed as skipped.

The steps individually:

    python scripts/train.py --blocks 6 --graph path --shared-germ 1 --out outputs/checkpoint
    python scripts/score.py outputs/checkpoint
    python scripts/certify.py --load outputs/checkpoint/params.npz
    python scripts/paper_numbers.py
    python scripts/tail_asymmetry.py --run outputs/checkpoint
    python scripts/bell_generative_advantage.py
    python scripts/plot_paper.py outputs/checkpoint

`train.py` retrains from scratch and will not land on the shipped checkpoint bit
for bit, since L-BFGS on a non-convex objective is sensitive to floating-point
ordering. Every number in the paper is computed from the shipped checkpoint,
which is why `reproduce.py` uses it.

At eight qubits and depth six the model is exactly simulable, so every quantity
here, the outputs, the training gradients and the reference floors alike, is
computed exactly rather than approximated.

## Layout

    src/qpce/quantum_features.py   the circuit, statevector and expectation values
    src/qpce/energy.py             the loss, unbiased energy distance
    src/qpce/adjoint.py            exact gradients at three state evolutions
    src/qpce/training.py           L-BFGS, frozen germ batch, fresh-germ validation
    src/qpce/certify.py            the verification protocol
    src/qpce/metrics.py            rank statistics, tails, the dependence score
    src/qpce/data.py               dataset and the 2n unit conversions
    src/qpce/config.py             coupler graphs, edge colouring, resource cost
    src/qpce/baselines.py          classical reference densities
    src/qpce/model.py              end-to-end orchestration
    src/qpce/plots.py              figure helpers

    scripts/reproduce.py                   recompute and check every number
    scripts/train.py                       train
    scripts/score.py                       held-out scoring with intervals
    scripts/certify.py                     verification protocol
    scripts/paper_numbers.py               regenerate numbers.tex
    scripts/tail_asymmetry.py              tail coefficients and the corollary check
    scripts/bell_generative_advantage.py   measurement-setting certification
    scripts/advantage_investigation.py     light cone, anticoncentration, tail probes
    scripts/plot_paper.py                  publication figures
    scripts/selftest.py                    twenty correctness gates

    data/                          CLIC eight-cell shower dataset, 4000 events
    outputs/checkpoint/            the deployed model, 154 angles
    outputs/                       everything the scripts compute
    plots/                         every figure

Source and generated artefacts never mix. Numbers land in `outputs/`, figures in
`plots/`, and both directories can be deleted and rebuilt.

## Data

The eight-cell CLIC electromagnetic calorimeter shower dataset, ~4000 Geant4
events.

## Citation

If you use this code, please cite the manuscript. See `CITATION.cff`.
