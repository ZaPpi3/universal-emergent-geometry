"""
DMRG ground states via quimb, built with the *same* Pauli-matrix convention
as lib.hamiltonians (not quimb's built-in MPO_ham_* helpers, which use a
different normalization). Used purely as an independent cross-check of the
exact sparse-ED results - not as the primary computational engine, since
sparse ED is exact (no bond-dimension truncation) and reaches N=20+ directly.
"""
import numpy as np
import quimb.tensor as qtn

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def xxz_mpo(N, delta, cyclic=False):
    sh = qtn.SpinHam1D(S=0.5, cyclic=cyclic)
    sh.add_term(1.0, _SX, _SX)
    sh.add_term(1.0, _SY, _SY)
    sh.add_term(delta, _SZ, _SZ)
    return sh.build_mpo(N)


def tfim_mpo(N, h, pin=1e-6, cyclic=False):
    sh = qtn.SpinHam1D(S=0.5, cyclic=cyclic)
    sh.add_term(-1.0, _SZ, _SZ)
    sh.add_term(-h, _SX)
    if pin != 0.0:
        # NOTE: SpinHam1D.__setitem__ *replaces* (does not add to) the
        # default one-site terms at an overridden site, so the uniform
        # field must be re-specified explicitly here alongside the pin,
        # or site 0 silently loses its -h*X field term.
        sh[0] += -h, _SX
        sh[0] += -pin, _SZ
    return sh.build_mpo(N)


def dmrg_ground_state(mpo, bond_dims=(50, 100, 200, 300), cutoff=1e-10, tol=1e-10):
    """Run DMRG2, return (energy, dense_statevector, max_bond_dim_used, converged)."""
    dmrg = qtn.DMRG2(mpo, bond_dims=list(bond_dims), cutoffs=cutoff)
    converged = dmrg.solve(tol=tol, verbosity=0)
    energy = float(np.real(dmrg.energy))
    psi_dense = np.asarray(dmrg.state.to_dense()).flatten()
    psi_dense = psi_dense / np.linalg.norm(psi_dense)
    max_chi = max(dmrg.state.bond_sizes()) if hasattr(dmrg.state, "bond_sizes") else None
    return energy, psi_dense, max_chi, converged
