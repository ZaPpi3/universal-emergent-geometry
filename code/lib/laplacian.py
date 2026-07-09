"""
Normalized mutual-information graph Laplacian and derived spectral
diagnostics: lambda1, lambda2, spectral gap, MI decay exponent, and
spectral embeddings.
"""
import numpy as np


def normalized_laplacian(I):
    """L = Id - D^{-1/2} I D^{-1/2}, D_ii = sum_j I_ij.

    Convention for isolated nodes (D_ii = 0, e.g. the spin-glass control
    where I is identically zero): the corresponding row/col of the
    normalized adjacency term is defined to be zero (not NaN), so that
    L reduces exactly to the identity matrix when I = 0 everywhere.
    """
    N = I.shape[0]
    D = I.sum(axis=1)
    Dinv_sqrt = np.zeros(N)
    nonzero = D > 1e-13
    Dinv_sqrt[nonzero] = 1.0 / np.sqrt(D[nonzero])
    norm_adj = (Dinv_sqrt[:, None] * I) * Dinv_sqrt[None, :]
    L = np.eye(N) - norm_adj
    return L


def spectral_quantities(L):
    """Return (eigenvalues ascending, eigenvectors), plus lambda1, lambda2, gap.
    lambda0 (~0, the trivial mode) is excluded from lambda1/lambda2.
    """
    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    lambda1 = float(evals[1])
    lambda2 = float(evals[2]) if len(evals) > 2 else float("nan")
    gap = lambda2 - lambda1
    v1 = evecs[:, 1]
    v2 = evecs[:, 2] if len(evals) > 2 else np.zeros_like(v1)
    return {
        "evals": evals,
        "evecs": evecs,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "gap": gap,
        "v1": v1,
        "v2": v2,
    }


def mi_decay_exponent(I, pbc=False, min_points=3):
    """Fit I(r) ~ r^{-alpha} via log-log linear regression of the
    distance-averaged mutual information. Returns (alpha, r_values, I_avg,
    r2) or (nan, ..., ...) if fewer than `min_points` valid (I>0) distance
    bins are available.
    """
    N = I.shape[0]
    idx_i, idx_j = np.triu_indices(N, k=1)
    r = np.abs(idx_i - idx_j)
    if pbc:
        r = np.minimum(r, N - r)
    vals = I[idx_i, idx_j]

    r_unique = np.unique(r)
    r_avg, I_avg = [], []
    for rv in r_unique:
        m = vals[r == rv]
        avg = m.mean()
        if avg > 1e-12:
            r_avg.append(rv)
            I_avg.append(avg)
    r_avg = np.array(r_avg, dtype=float)
    I_avg = np.array(I_avg, dtype=float)

    if len(r_avg) < min_points:
        return float("nan"), r_avg, I_avg, float("nan")

    x = np.log(r_avg)
    y = np.log(I_avg)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    alpha = -slope
    yhat = A @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), r_avg, I_avg, float(r2)
