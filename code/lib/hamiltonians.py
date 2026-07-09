"""
Sparse Hamiltonian constructors for the four models studied in the paper:
XXZ chain, transverse-field Ising model (TFIM), classical spin glass, and
free fermions (XX chain).

All spin (non-classical) Hamiltonians are built as sparse CSR matrices over
the full 2^N Hilbert space using bond-term embedding: each nearest-neighbour
bond operator is a 4x4 sparse block, embedded via kron(I_left, block, I_right)
and summed incrementally to keep peak memory at O(2^N) rather than O(N * 2^N).

Ground states are obtained via sparse Lanczos (scipy.sparse.linalg.eigsh),
which only requires matrix-vector products and therefore scales far beyond
what dense diagonalization allows - this is what lets us reach N=20 (and
beyond) with *no truncation approximation*, unlike DMRG or dense ED.
"""
import numpy as np
import scipy.sparse as sp

# Single-qubit Pauli matrices (sparse)
_SX = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_SY = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_SZ = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_ID2 = sp.identity(2, format="csr", dtype=complex)


def _embed_bond(block4, i, N, pbc_wrap=False):
    """Embed a 4x4 two-site operator `block4` acting on sites (i, i+1 mod N)
    into the full 2^N-dim Hilbert space via kron(I_left, block, I_right).

    For the wraparound bond of a PBC chain (i = N-1, site N-1 & site 0), we
    instead build the term as a sum of matched single-site operators using
    the same block decomposition, since sites 0 and N-1 are not adjacent in
    the tensor-product ordering.
    """
    if not pbc_wrap:
        left = sp.identity(2 ** i, format="csr", dtype=complex)
        right = sp.identity(2 ** (N - i - 2), format="csr", dtype=complex)
        return sp.kron(sp.kron(left, block4, format="csr"), right, format="csr")
    else:
        raise ValueError("wraparound bonds must be built term-by-term; use _embed_wraparound")


def _embed_single_pair(op_a, op_b, i, j, N):
    """Embed op_a (acting on site i) (x) op_b (acting on site j), i<j, into 2^N space."""
    assert i < j
    left = sp.identity(2 ** i, format="csr", dtype=complex)
    mid = sp.identity(2 ** (j - i - 1), format="csr", dtype=complex)
    right = sp.identity(2 ** (N - j - 1), format="csr", dtype=complex)
    return sp.kron(sp.kron(sp.kron(sp.kron(left, op_a, format="csr"), mid, format="csr"),
                            op_b, format="csr"), right, format="csr")


def _bonds(N, pbc):
    bonds = [(i, i + 1) for i in range(N - 1)]
    if pbc:
        bonds.append((N - 1, 0))
    return bonds


def build_xxz(N, delta, pbc=False):
    """H = sum_bonds [ Sx_i Sx_j + Sy_i Sy_j + delta * Sz_i Sz_j ]
    using Pauli (not spin-1/2-normalized) operators, matching the paper's
    convention H = sum (sx sx + sy sy + delta sz sz).
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    block = (sp.kron(_SX, _SX, format="csr") + sp.kron(_SY, _SY, format="csr")
             + delta * sp.kron(_SZ, _SZ, format="csr"))
    for (i, j) in _bonds(N, pbc):
        if j == i + 1:
            H = H + _embed_bond(block, i, N)
        else:
            # wraparound PBC bond
            H = H + (_embed_single_pair(_SX, _SX, 0, N - 1, N)
                      + _embed_single_pair(_SY, _SY, 0, N - 1, N)
                      + delta * _embed_single_pair(_SZ, _SZ, 0, N - 1, N))
    H.eliminate_zeros()
    return H.real.astype(np.float64) if np.allclose(H.imag.data, 0) else H


def build_tfim(N, h, pbc=False, pin=1e-6):
    """H = -sum_bonds Sz_i Sz_j - h * sum_i Sx_i - pin * Sz_0

    `pin` is a tiny explicit symmetry-breaking longitudinal field on site 0
    that lifts the (near-)exact two-fold Z2 degeneracy in the ordered phase
    (h<1). Without it, ARPACK returns an essentially arbitrary cat-state
    superposition there, which has spuriously large/ill-defined entanglement.
    Physically this mimics the infinitesimal symmetry-breaking field always
    present in a real ordered phase. We verify results are insensitive to
    the exact value of `pin` (robustness test, see run_pin_sensitivity.py).
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    zz = sp.kron(_SZ, _SZ, format="csr")
    for (i, j) in _bonds(N, pbc):
        if j == i + 1:
            H = H - _embed_bond(zz, i, N)
        else:
            H = H - _embed_single_pair(_SZ, _SZ, 0, N - 1, N)
    for i in range(N):
        left = sp.identity(2 ** i, format="csr", dtype=complex)
        right = sp.identity(2 ** (N - i - 1), format="csr", dtype=complex)
        H = H - h * sp.kron(sp.kron(left, _SX, format="csr"), right, format="csr")
    if pin != 0.0:
        left = sp.identity(1, format="csr", dtype=complex)
        right = sp.identity(2 ** (N - 1), format="csr", dtype=complex)
        H = H - pin * sp.kron(sp.kron(left, _SZ, format="csr"), right, format="csr")
    H.eliminate_zeros()
    return H.real.astype(np.float64) if np.allclose(H.imag.data, 0) else H


def build_free_fermion(N, t=1.0, pbc=False):
    """XX chain: H = -t * sum_bonds (Sx_i Sx_j + Sy_i Sy_j).
    Equivalent (via Jordan-Wigner) to free fermions, and identical to the
    XXZ chain at delta=0 up to the overall scale t. Ground-state entanglement
    is invariant under the overall positive scale t, so t is *not* expected
    to change any MI-derived quantity - this is used as a built-in
    consistency check, not a genuine physical sweep parameter.
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    block = t * (sp.kron(_SX, _SX, format="csr") + sp.kron(_SY, _SY, format="csr"))
    for (i, j) in _bonds(N, pbc):
        if j == i + 1:
            H = H + _embed_bond(block, i, N)
        else:
            H = H + t * (_embed_single_pair(_SX, _SX, 0, N - 1, N)
                          + _embed_single_pair(_SY, _SY, 0, N - 1, N))
    H.eliminate_zeros()
    return H.real.astype(np.float64) if np.allclose(H.imag.data, 0) else H


def build_spin_glass_diag(N, seed, J0=1.0, pbc=False):
    """Classical spin-glass Hamiltonian, diagonal in the computational basis:
    H = sum_{i<j} J_ij sigma^z_i sigma^z_j, J_ij ~ N(0, J0^2), all-to-all.

    Returns the length-2^N diagonal (real) as a numpy array, and does NOT
    build a sparse operator (unnecessary - it's already diagonal).
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** N
    # sigma^z eigenvalues for each basis state, all sites: +1 for bit 0, -1 for bit 1
    bits = ((np.arange(dim)[:, None] >> np.arange(N)[None, :]) & 1)
    z = 1 - 2 * bits  # (dim, N), +-1
    idx_i, idx_j = np.triu_indices(N, k=1)
    Jij = rng.normal(0.0, J0, size=idx_i.shape[0])
    diag = (z[:, idx_i] * z[:, idx_j] * Jij[None, :]).sum(axis=1)
    return diag
