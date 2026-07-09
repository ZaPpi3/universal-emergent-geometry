# Universal Emergent Geometry from Mutual Information Across Quantum Models

Paul Jarvis, Independent Researcher, United Kingdom (mrpaulwjarvis@gmail.com)

This repository is the full data/code/paper release accompanying the manuscript
`main.tex` / `main.pdf`. It studies mutual-information-induced graph Laplacians
as a diagnostic of quantum phase structure across the XXZ chain, the
transverse-field Ising model (TFIM), a classical spin-glass control, and free
fermions, using exact diagonalization up to N=20.

## Contents

```
main.tex, main.pdf      the paper
code/                    all analysis code
  lib/                    Hamiltonians, entanglement/MI, Laplacian, DMRG utils, plot style
  run_all.py              9 of the paper's 10 experiments, run in dependency order
  run_pin_sensitivity.py  the 10th, kept separate - named directly in main.tex Sec. II.B
  make_figures.py         builds all figures/*.png from data/ (run last)
  tests/                  pytest correctness suite (see below)
data/                   CSV/JSON/NPZ outputs of run_all.py / run_pin_sensitivity.py
                         (checked into the repo so the paper's figures/tables/tests
                         are reproducible without re-running the full sweep)
figures/                PNG figures included in the paper
```

`run_all.py` is a merge of what were originally 9 separate one-script-per-experiment
files (`run_main_sweep.py`, `run_finite_size.py`, `run_large_n_scaling.py`,
`run_large_n_scaling_pbc.py`, `run_scaling_fits.py`, `run_delta_sweep.py`,
`run_central_charge_fit.py`, `run_scaling_collapse.py`, `run_dmrg_crosscheck.py`),
combined into one file purely to cut down the file count for this release - no
computation changed. Each stage is a separate, namespaced function
(`stage_main_sweep()`, `stage_finite_size()`, ...); see the module docstring in
`run_all.py` for the full mapping and rationale.

## Requirements

```
pip install -r requirements.txt
```

`quimb` (the DMRG cross-check library) is only needed for the DMRG stage of
`run_all.py` (Table I) and the `@pytest.mark.slow` test; if it isn't
installed, `run_all.py` skips that one stage with a message and still
completes everything else - including all N=20 results, which use exact
sparse diagonalization and have no tensor-network dependency.

## Reproducing the results

The data/ and figures/ directories are already populated, so the paper can be
read and checked as-is. To regenerate everything from scratch:

```
cd code
python run_all.py               # 9 experiments; DMRG stage needs quimb, else skipped
python run_pin_sensitivity.py   # 10th experiment, kept separate (see above)
python make_figures.py          # run last: builds figures/ from data/
```

## Tests

```
cd code
pytest -v                   # all tests
pytest -v -m "not slow"     # skip the DMRG cross-check (requires quimb)
```

Two files:
- `tests/test_pipeline.py` (26 tests) - Hamiltonian Hermiticity (OBC & PBC),
  the spin-glass and free-fermion consistency conditions (including the h=0
  classical limit of the TFIM), Laplacian spectral bounds and invariants
  (trace, embedding orthonormality), Lanczos residual convergence, exact
  recovery of known fitting forms on synthetic data, and DMRG/exact-ED
  agreement.
- `tests/test_paper_numbers.py` (18 tests) - checks every quantitative claim
  made in the paper's text and tables directly against the corresponding data
  file, so that a future change to the pipeline (or a transcription slip in
  the manuscript) that silently altered a reported number would fail this
  suite rather than pass unnoticed.

## Data availability

All numerical results in the paper are reproducible from this release. See
the "Data Availability" section of `main.tex` / `main.pdf` for details.
