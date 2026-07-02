# Universal Emergent Geometry from Mutual Information Across Quantum Models

This repository hosts the standalone manuscript source code, numerical exact diagonalization engines, and underlying datasets for evaluating how smooth, coordinate-invariant geometric structures emerge universally from many-body quantum ground states.

By processing the ground states of four completely different physical systems across finite-size runs ($N = 6, 8, 10$), this framework demonstrates that entanglement-induced geometry functions as a universal macro-state of quantum information, revealing distinct and predictable phase indicators entirely in the spectral domain.

## 📁 Repository Architecture

*   `main.tex` — Flagship typeset LaTeX manuscript source formatted under the single-column PRL template.
*   `main.pdf` — Flagship typeset pdf manuscript formatted under the single-column PRL template.
*   `code/` — Advanced numerical calculation and state compilation modules.
    *   `manybody_mi_compiler.py` — The full exact diagonalization execution engine. It constructs the multi-model Hilbert space actions, computes reduced density matrices via partial traces, and extracts true von Neumann mutual information.
*   `release/` — Complete open-source reproducibility bundle containing the 12-matrix dataset and 4 core structural charts:
    *   **Invariant Material Datasets (.csv)**
        *   `xxz_N6.csv`, `xxz_N8.csv`, `xxz_N10.csv` — Finite-size critical geometry logs.
        *   `tfim_N6.csv`, `tfim_N8.csv`, `tfim_N10.csv` — Discrete symmetry-breaking records.
        *   `spinglass_N6.csv`, `spinglass_N8.csv`, `spinglass_N10.csv` — Disordered noise baseline matrix sets.
        *   `freefermion_N6.csv`, `freefermion_N8.csv`, `freefermion_N10.csv` — Flat integrable system scaling logs.
    *   **Diagnostic Visualization Panels (.png)**
        *   `universality_lambda1.png` — Lowest non-zero eigenvalue $\lambda_1$ scaling trajectories.
        *   `universality_gap.png` — Geometric spectral gap $\Delta_{\text{gap}}$ transition indicators.
        *   `universality_mi_decay.png` — Spatial correlation decay exponent $\alpha$ mapping parameters.
        *   `universality_embeddings.png` — Low-dimensional spectral embedding coordinate charts.

## 🧠 Geometric Universality Classes

The extraction engine maps quantum matter into four explicit geometric phases:
1.  **Continuous Critical Geometry (XXZ)**: Smooth eigenvalue minima and nearly circular embedding charts, reflecting maximal coordinate smoothness and conformal invariance.
2.  **Symmetry-Breaking Geometry (TFIM)**: Sharp, abrupt spectral drops and warped, crumpled topology transformations near the critical phase transition.
3.  **Disorder-Dominated Geometry (Spin Glass)**: Complete spectral collapse to the flat identity matrix ($L = \mathbb{I}$), acting as a strict control proving the extractor cannot create geometry from noise.
4.  **Integrable Flat Geometry (Free Fermions)**: Rigidly invariant, non-deforming lattice structures locked by local conservation laws.

## 🚀 Quickstart Reproducibility Loop

To re-run the many-body ground-state integrations from scratch, ensure you have a standard Python scientific stack (`numpy`, `scipy`, `matplotlib`) configured, and execute the numerical core:

```bash
python code/manybody_mi_compiler.py
```

## 📜 Open-Source Licensing

This project is archived openly under the terms of the **MIT License**. All manuscript assets and numerical datasets can be redistributed, audited, or expanded with proper scientific attribution.
