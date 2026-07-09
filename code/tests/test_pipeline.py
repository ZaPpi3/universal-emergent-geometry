"""
Correctness tests for the MI-Laplacian pipeline. Run with:
    pytest code/tests/test_pipeline.py -v
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from lib import hamiltonians as H
from lib import entanglement as E
from lib import laplacian as L


# ---------------------------------------------------------------- fixtures
N_SMALL = 6


# ---------------------------------------------------------------- Hamiltonians
def test_xxz_hermitian():
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    diff = (Hs - Hs.T)
    assert np.max(np.abs(diff.toarray() if hasattr(diff, "toarray") else diff)) < 1e-10


def test_tfim_hermitian():
    Hs = H.build_tfim(N_SMALL, h=1.0, pin=1e-6)
    diff = (Hs - Hs.T)
    assert np.max(np.abs(diff.toarray() if hasattr(diff, "toarray") else diff)) < 1e-10


def test_xxz_pbc_hermitian():
    Hs = H.build_xxz(N_SMALL, delta=1.0, pbc=True)
    diff = (Hs - Hs.T)
    assert np.max(np.abs(diff.toarray() if hasattr(diff, "toarray") else diff)) < 1e-10


def test_tfim_pbc_hermitian():
    Hs = H.build_tfim(N_SMALL, h=1.0, pin=1e-6, pbc=True)
    diff = (Hs - Hs.T)
    assert np.max(np.abs(diff.toarray() if hasattr(diff, "toarray") else diff)) < 1e-10


def test_free_fermion_pbc_hermitian():
    Hs = H.build_free_fermion(N_SMALL, t=1.0, pbc=True)
    diff = (Hs - Hs.T)
    assert np.max(np.abs(diff.toarray() if hasattr(diff, "toarray") else diff)) < 1e-10


def test_free_fermion_equals_xxz_delta0_pbc():
    """Same equivalence as test_free_fermion_equals_xxz_delta0, but under PBC --
    exercises the separate wraparound-bond code path in both builders."""
    Hxxz = H.build_xxz(N_SMALL, delta=0.0, pbc=True).toarray()
    Hff = H.build_free_fermion(N_SMALL, t=1.0, pbc=True).toarray()
    assert np.max(np.abs(Hxxz - Hff)) < 1e-10


def test_free_fermion_equals_xxz_delta0():
    """Free-fermion (XX) chain is exactly the XXZ chain at Delta=0."""
    Hxxz = H.build_xxz(N_SMALL, delta=0.0).toarray()
    Hff = H.build_free_fermion(N_SMALL, t=1.0).toarray()
    assert np.max(np.abs(Hxxz - Hff)) < 1e-10


def test_free_fermion_scale_invariance():
    """Ground-state MI must be invariant under the overall energy scale t
    (an overall positive rescaling of H does not change its eigenvectors)."""
    Hs1 = H.build_free_fermion(N_SMALL, t=0.3)
    Hs2 = H.build_free_fermion(N_SMALL, t=1.7)
    e1, psi1 = E.ground_state_sparse(Hs1)
    e2, psi2 = E.ground_state_sparse(Hs2)
    I1, _ = E.mutual_information_matrix(psi1, N_SMALL)
    I2, _ = E.mutual_information_matrix(psi2, N_SMALL)
    assert np.max(np.abs(I1 - I2)) < 1e-8


# ---------------------------------------------------------------- spin glass control
def test_spin_glass_zero_mutual_information():
    diag = H.build_spin_glass_diag(N_SMALL, seed=42)
    e0, psi = E.ground_state_diag(diag)
    # ground state must be an exact computational basis state
    assert np.isclose(np.max(np.abs(psi)), 1.0)
    assert np.isclose(np.sum(psi ** 2), 1.0)
    I, S1 = E.mutual_information_matrix(psi, N_SMALL)
    assert np.max(np.abs(I)) < 1e-12
    assert np.max(np.abs(S1)) < 1e-12


def test_spin_glass_laplacian_is_identity():
    diag = H.build_spin_glass_diag(N_SMALL, seed=7)
    e0, psi = E.ground_state_diag(diag)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    lap = L.normalized_laplacian(I)
    assert np.allclose(lap, np.eye(N_SMALL))


# ---------------------------------------------------------------- entropy/MI properties
def test_entropy_nonnegative():
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    e0, psi = E.ground_state_sparse(Hs)
    S1 = E.single_site_entropies(psi, N_SMALL)
    assert np.all(S1 >= -1e-10)
    assert np.all(S1 <= np.log(2) + 1e-8)  # single qubit entropy bounded by ln 2


def test_mutual_information_nonnegative_and_symmetric():
    Hs = H.build_tfim(N_SMALL, h=1.0, pin=1e-6)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    assert np.all(I >= -1e-10)
    assert np.allclose(I, I.T)
    assert np.allclose(np.diag(I), 0.0)


def test_statevector_normalized():
    Hs = H.build_xxz(N_SMALL, delta=0.5)
    e0, psi = E.ground_state_sparse(Hs)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_lanczos_residual_small():
    """||H psi - E0 psi|| should be tiny - checks that the reported ground
    state actually solves the eigenproblem, not merely that it's normalized."""
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    e0, psi = E.ground_state_sparse(Hs)
    resid = Hs.dot(psi) - e0 * psi
    assert np.linalg.norm(resid) < 1e-8


def test_tfim_h0_is_classical_product_state():
    """At h=0, the TFIM Hamiltonian (-sum ZZ - pin*Z_0) is diagonal in the
    computational basis, exactly like the spin-glass control: the ground
    state should be an unentangled product state, with the pin field alone
    responsible for selecting one of the two ferromagnetic branches."""
    Hs = H.build_tfim(N_SMALL, h=0.0, pin=1e-6)
    e0, psi = E.ground_state_sparse(Hs)
    I, S1 = E.mutual_information_matrix(psi, N_SMALL)
    assert np.max(np.abs(I)) < 1e-8
    assert np.max(np.abs(S1)) < 1e-8


# ---------------------------------------------------------------- Laplacian spectral properties
def test_laplacian_eigenvalues_in_range():
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    lap = L.normalized_laplacian(I)
    sq = L.spectral_quantities(lap)
    # normalized graph Laplacian: eigenvalues in [0, 2], lambda0 ~ 0
    assert sq["evals"][0] < 1e-8
    assert np.all(sq["evals"] > -1e-8)
    assert np.all(sq["evals"] < 2.0 + 1e-8)


def test_laplacian_symmetric():
    Hs = H.build_tfim(N_SMALL, h=0.5, pin=1e-6)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    lap = L.normalized_laplacian(I)
    assert np.allclose(lap, lap.T)


def test_laplacian_trace_equals_N():
    """trace(L) = N - sum_i (norm_adj)_ii = N exactly, since I_ii = 0 by
    construction (no self-loops), for any MI matrix - including the
    isolated-node (spin-glass) case where L = Id."""
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    lap = L.normalized_laplacian(I)
    assert abs(np.trace(lap) - N_SMALL) < 1e-10


def test_spectral_embedding_orthonormal():
    """v1, v2 are eigenvectors of a real symmetric matrix at distinct
    (non-degenerate here) eigenvalues, so they must be orthonormal."""
    Hs = H.build_xxz(N_SMALL, delta=1.0)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N_SMALL)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    v1, v2 = sq["v1"], sq["v2"]
    assert abs(np.dot(v1, v1) - 1.0) < 1e-10
    assert abs(np.dot(v2, v2) - 1.0) < 1e-10
    assert abs(np.dot(v1, v2)) < 1e-10


# ---------------------------------------------------------------- decay-exponent fit
def test_mi_decay_exponent_nan_when_insufficient_data():
    I = np.zeros((6, 6))
    alpha, r, Iavg, r2 = L.mi_decay_exponent(I)
    assert np.isnan(alpha)


def test_mi_decay_exponent_recovers_known_power_law():
    """Synthetic MI matrix I(r) = r^-2 exactly; the fit should recover alpha=2."""
    N = 12
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                r = abs(i - j)
                I[i, j] = r ** (-2.0)
    alpha, r, Iavg, r2 = L.mi_decay_exponent(I)
    assert abs(alpha - 2.0) < 1e-6
    assert r2 > 0.999


def test_mi_decay_exponent_pbc_distance_folding():
    """Under pbc=True, pair distance must be folded to the shorter arc,
    r = min(|i-j|, N-|i-j|); build I(r) = r^-2 in terms of the *folded*
    distance and check the fit still recovers alpha=2 - this would fail
    if the wraparound folding were missing or wrong."""
    N = 12
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                r = min(abs(i - j), N - abs(i - j))
                I[i, j] = r ** (-2.0)
    alpha, r, Iavg, r2 = L.mi_decay_exponent(I, pbc=True)
    assert abs(alpha - 2.0) < 1e-6
    assert r2 > 0.999
    assert r.max() <= N // 2


# ---------------------------------------------------------------- cross-validation vs DMRG
@pytest.mark.slow
def test_dmrg_matches_exact_ed():
    from lib import dmrg_utils as D
    N = 10
    Hs = H.build_xxz(N, delta=1.0)
    e_ed, psi_ed = E.ground_state_sparse(Hs)
    mpo = D.xxz_mpo(N, delta=1.0)
    e_dmrg, psi_dmrg, chi, converged = D.dmrg_ground_state(mpo)
    assert converged
    assert abs(e_ed - e_dmrg) < 1e-5


# ---------------------------------------------------------------- fits.py
def test_fit_power_law_recovers_exact_power_law():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from fits import fit_power_law
    N = np.array([6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
    y = 3.0 * N ** (-1.7)
    fit = fit_power_law(N, y)
    assert abs(fit["b"] - 1.7) < 1e-6
    assert fit["r2"] > 1 - 1e-10


def test_fit_log_corrected_recovers_exact_form():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from fits import fit_log_corrected
    N = np.array([6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
    a, c = 2.0, -0.5
    y = (a / N) * (1 + c / np.log(N))
    fit = fit_log_corrected(N, y)
    assert abs(fit["c"] - c) < 1e-6
    assert fit["r2"] > 1 - 1e-10


def test_fit_saturating_recovers_exact_form():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from fits import fit_saturating
    N = np.array([6, 8, 10, 12, 14, 16, 18, 20], dtype=float)
    yinf, a, b = 0.3, 2.0, 1.2
    y = yinf + a * N ** (-b)
    fit = fit_saturating(N, y)
    assert abs(fit["yinf"] - yinf) < 1e-5
    assert abs(fit["b"] - b) < 1e-4


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
