"""
Ground states, reduced density matrices, entropies, and mutual-information
matrices for spin-1/2 chains, computed exactly from the dense statevector.

All entropies use the natural log (nats), so that the Calabrese-Cardy
central-charge fit (stage_central_charge_fit() in run_all.py, S = (c/3) ln N
+ const for the PBC half-chain cut used in the paper) uses the literature
coefficient directly without a base-conversion factor. The same convention
also supports an OBC fit with the c/6 coefficient, though the paper does not
use one - OBC half-chain entropy shows an N mod 4 parity oscillation that
corrupts a naive fit, so PBC data is used instead (see
stage_central_charge_fit() in run_all.py).
"""
import numpy as np
import scipy.sparse.linalg as spla

_EPS = 1e-14


def ground_state_sparse(H, tol=1e-10, maxiter=5000):
    """Lanczos ground state of a sparse (or dense) Hermitian matrix H.
    Returns (E0, psi) with psi a dense complex/float numpy array, ||psi||=1.
    """
    dim = H.shape[0]
    if dim <= 4:
        # ARPACK requires k < dim; fall back to dense for tiny systems.
        Hd = np.asarray(H.todense()) if hasattr(H, "todense") else np.asarray(H)
        evals, evecs = np.linalg.eigh(Hd)
        return evals[0], evecs[:, 0]
    evals, evecs = spla.eigsh(H, k=1, which="SA", tol=tol, maxiter=maxiter)
    psi = evecs[:, 0]
    psi = psi / np.linalg.norm(psi)
    return evals[0], psi


def ground_state_diag(diag):
    """Ground state of a diagonal Hamiltonian: an exact computational-basis
    state at the (lexicographically first) minimizing index. This avoids
    ARPACK ever returning an arbitrary superposition across the exact
    degeneracy of a classical Hamiltonian.
    """
    idx = int(np.argmin(diag))
    psi = np.zeros(diag.shape[0], dtype=np.float64)
    psi[idx] = 1.0
    return diag[idx], psi


def _reshape_tensor(psi, N):
    return psi.reshape((2,) * N)


def reduced_rho_single(psi, N, i):
    t = _reshape_tensor(psi, N)
    t = np.moveaxis(t, i, 0).reshape(2, -1)
    rho = t @ t.conj().T
    return rho


def reduced_rho_pair(psi, N, i, j):
    assert i != j
    t = _reshape_tensor(psi, N)
    rest = [k for k in range(N) if k not in (i, j)]
    perm = [i, j] + rest
    t = np.transpose(t, perm).reshape(4, -1)
    rho = t @ t.conj().T
    return rho


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals.real, 0.0, None)
    evals = evals[evals > _EPS]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log(evals)))


def single_site_entropies(psi, N):
    return np.array([von_neumann_entropy(reduced_rho_single(psi, N, i)) for i in range(N)])


def mutual_information_matrix(psi, N):
    """Full N x N mutual information matrix I_ij = S_i + S_j - S_ij (I_ii = 0)."""
    S1 = single_site_entropies(psi, N)
    I = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            Sij = von_neumann_entropy(reduced_rho_pair(psi, N, i, j))
            val = S1[i] + S1[j] - Sij
            val = max(val, 0.0)  # MI is non-negative in exact arithmetic; guard tiny -eps noise
            I[i, j] = I[j, i] = val
    return I, S1


def half_chain_entropy(psi, N):
    """Entropy of the bipartition [0..N//2-1] | [N//2..N-1] (contiguous half-chain),
    used for the Calabrese-Cardy central-charge fit.
    """
    half = N // 2
    t = _reshape_tensor(psi, N)
    perm = list(range(half)) + list(range(half, N))
    t = np.transpose(t, perm).reshape(2 ** half, -1)
    rho = t @ t.conj().T
    return von_neumann_entropy(rho)
