import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import csv
import os

# ============================================================
# Spin operators
# ============================================================

sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_all(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def single_site_operator(op, i, N):
    ops = [I2] * N
    ops[i] = op
    return kron_all(ops)


def two_site_operator(op1, i, op2, j, N):
    ops = [I2] * N
    ops[i] = op1
    ops[j] = op2
    return kron_all(ops)


# ============================================================
# Hamiltonians
# ============================================================

def xxz_hamiltonian(N, delta):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H += two_site_operator(sx, i, sx, i + 1, N)
        H += two_site_operator(sy, i, sy, i + 1, N)
        H += delta * two_site_operator(sz, i, sz, i + 1, N)
    return H


def tfim_hamiltonian(N, h):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H += - two_site_operator(sz, i, sz, i + 1, N)
    for i in range(N):
        H += - h * single_site_operator(sx, i, N)
    return H


def spin_glass_hamiltonian(N, J_matrix, h=0.0):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            H += J_matrix[i, j] * two_site_operator(sz, i, sz, j, N)
    if h != 0.0:
        for i in range(N):
            H += - h * single_site_operator(sz, i, N)
    return H


def free_fermion_hamiltonian(N, t):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N - 1):
        H += - t * two_site_operator(sx, i, sx, i + 1, N)
        H += - t * two_site_operator(sy, i, sy, i + 1, N)
    return H


# ============================================================
# MI, Laplacian, entropy
# ============================================================

def partial_trace(rho, keep, N):
    dims = [2] * N
    reshaped = rho.reshape(dims + dims)
    trace_over = sorted(set(range(N)) - set(keep), reverse=True)
    current_N = N
    for site in trace_over:
        reshaped = np.trace(reshaped, axis1=site, axis2=site + current_N)
        current_N -= 1
    dim_keep = 2 ** len(keep)
    return reshaped.reshape(dim_keep, dim_keep)


def entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 0.0
    return float(-np.sum(eigvals * np.log(eigvals)))


def mutual_information_matrix(psi, N):
    rho = np.outer(psi, np.conjugate(psi))
    MI = np.zeros((N, N))
    single_entropies = []
    for i in range(N):
        rho_i = partial_trace(rho, [i], N)
        single_entropies.append(entropy(rho_i))
    for i in range(N):
        for j in range(i + 1, N):
            rho_ij = partial_trace(rho, [i, j], N)
            sij = entropy(rho_ij)
            mi = single_entropies[i] + single_entropies[j] - sij
            MI[i, j] = mi
            MI[j, i] = mi
    return MI


def normalized_laplacian(A):
    degree = np.sum(A, axis=1)
    degree[degree < 1e-12] = 1e-12
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    L = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def laplacian_spectrum_and_vectors(MI):
    L = normalized_laplacian(MI)
    eigvals, eigvecs = la.eigh(L)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


def mi_decay_exponent(MI):
    N = MI.shape[0]
    r_vals = np.arange(1, N // 2 + 1)
    mi_vals = []
    for r in r_vals:
        vals = []
        for i in range(N):
            j = (i + r) % N
            vals.append(MI[i, j])
        mi_vals.append(np.mean(vals))
    mi_vals = np.array(mi_vals)
    mask = mi_vals > 1e-6
    if np.sum(mask) < 2:
        return np.nan
    r_fit = r_vals[mask]
    mi_fit = mi_vals[mask]
    log_r = np.log(r_fit)
    log_mi = np.log(mi_fit)
    slope, _ = np.polyfit(log_r, log_mi, 1)
    alpha = -slope
    return float(alpha)


def spectral_embedding(eigvecs):
    v1 = eigvecs[:, 1]
    v2 = eigvecs[:, 2]
    return v1.real, v2.real


# ============================================================
# CSV helper
# ============================================================

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ============================================================
# Main suite
# ============================================================

def run_suite():
    Ns = [6, 8, 10]
    models = ["XXZ", "TFIM", "SpinGlass", "FreeFermion"]

    # Control parameter grids
    deltas_xxz = np.linspace(0.0, 2.0, 21)
    hs_tfim = np.linspace(0.0, 2.0, 21)
    J_strengths = np.linspace(0.0, 1.0, 21)
    ts_fermion = np.linspace(0.2, 2.0, 21)

    rng = np.random.default_rng(seed=123)
    base_J_dict = {}
    for N in Ns:
        base_J = rng.normal(0.0, 1.0, size=(N, N))
        base_J = (base_J + base_J.T) / 2.0
        np.fill_diagonal(base_J, 0.0)
        base_J_dict[N] = base_J

    # Storage for universality plots (N=8 only, for clarity)
    uni_lambda1 = {m: [] for m in models}
    uni_gap = {m: [] for m in models}
    uni_alpha = {m: [] for m in models}

    print("====================================================")
    print("   GOLD-STANDARD ENTANGLEMENT-GEOMETRY SUITE       ")
    print("   Models: XXZ, TFIM, SpinGlass, FreeFermion       ")
    print("   Sizes:  N = 6, 8, 10                            ")
    print("====================================================\n")

    for N in Ns:
        print(f"=== Chain size N = {N} ===")

        rows_xxz = []
        rows_tfim = []
        rows_sg = []
        rows_ff = []

        for idx, delta in enumerate(deltas_xxz):
            H = xxz_hamiltonian(N, delta)
            eigvals, eigvecs = la.eigh(H)
            ground_state = eigvecs[:, 0]
            MI = mutual_information_matrix(ground_state, N)
            spectrum, eigvecs_L = laplacian_spectrum_and_vectors(MI)
            lam1 = float(spectrum[1])
            lam2 = float(spectrum[2])
            gap = lam2 - lam1
            alpha = mi_decay_exponent(MI)
            rows_xxz.append([N, delta, lam1, lam2, gap, alpha])

            if N == 8:
                uni_lambda1["XXZ"].append(lam1)
                uni_gap["XXZ"].append(gap)
                uni_alpha["XXZ"].append(alpha)

        for idx, h in enumerate(hs_tfim):
            H = tfim_hamiltonian(N, h)
            eigvals, eigvecs = la.eigh(H)
            ground_state = eigvecs[:, 0]
            MI = mutual_information_matrix(ground_state, N)
            spectrum, eigvecs_L = laplacian_spectrum_and_vectors(MI)
            lam1 = float(spectrum[1])
            lam2 = float(spectrum[2])
            gap = lam2 - lam1
            alpha = mi_decay_exponent(MI)
            rows_tfim.append([N, h, lam1, lam2, gap, alpha])

            if N == 8:
                uni_lambda1["TFIM"].append(lam1)
                uni_gap["TFIM"].append(gap)
                uni_alpha["TFIM"].append(alpha)

        base_J = base_J_dict[N]
        for idx, s in enumerate(J_strengths):
            J_matrix = s * base_J
            H = spin_glass_hamiltonian(N, J_matrix, h=0.0)
            eigvals, eigvecs = la.eigh(H)
            ground_state = eigvecs[:, 0]
            MI = mutual_information_matrix(ground_state, N)
            spectrum, eigvecs_L = laplacian_spectrum_and_vectors(MI)
            lam1 = float(spectrum[1])
            lam2 = float(spectrum[2])
            gap = lam2 - lam1
            alpha = mi_decay_exponent(MI)
            rows_sg.append([N, s, lam1, lam2, gap, alpha])

            if N == 8:
                uni_lambda1["SpinGlass"].append(lam1)
                uni_gap["SpinGlass"].append(gap)
                uni_alpha["SpinGlass"].append(alpha)

        for idx, t in enumerate(ts_fermion):
            H = free_fermion_hamiltonian(N, t)
            eigvals, eigvecs = la.eigh(H)
            ground_state = eigvecs[:, 0]
            MI = mutual_information_matrix(ground_state, N)
            spectrum, eigvecs_L = laplacian_spectrum_and_vectors(MI)
            lam1 = float(spectrum[1])
            lam2 = float(spectrum[2])
            gap = lam2 - lam1
            alpha = mi_decay_exponent(MI)
            rows_ff.append([N, t, lam1, lam2, gap, alpha])

            if N == 8:
                uni_lambda1["FreeFermion"].append(lam1)
                uni_gap["FreeFermion"].append(gap)
                uni_alpha["FreeFermion"].append(alpha)

        # Write CSVs for this N
        write_csv(f"release/xxz_N{N}.csv",
                  ["N", "delta", "lambda1", "lambda2", "gap", "alpha"],
                  rows_xxz)
        write_csv(f"release/tfim_N{N}.csv",
                  ["N", "h", "lambda1", "lambda2", "gap", "alpha"],
                  rows_tfim)
        write_csv(f"release/spinglass_N{N}.csv",
                  ["N", "J_strength", "lambda1", "lambda2", "gap", "alpha"],
                  rows_sg)
        write_csv(f"release/freefermion_N{N}.csv",
                  ["N", "t", "lambda1", "lambda2", "gap", "alpha"],
                  rows_ff)

    # Convert universality arrays (N=8 only)
    deltas_xxz = np.array(deltas_xxz)
    hs_tfim = np.array(hs_tfim)
    J_strengths = np.array(J_strengths)
    ts_fermion = np.array(ts_fermion)
    for m in models:
        uni_lambda1[m] = np.array(uni_lambda1[m])
        uni_gap[m] = np.array(uni_gap[m])
        uni_alpha[m] = np.array(uni_alpha[m])

    # ============================================================
    # Universality plots (N=8)
    # ============================================================

    os.makedirs("release", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas_xxz, uni_lambda1["XXZ"], marker='o', label='XXZ (Δ)')
    plt.plot(hs_tfim, uni_lambda1["TFIM"], marker='s', label='TFIM (h)')
    plt.plot(J_strengths, uni_lambda1["SpinGlass"], marker='^', label='Spin Glass (J)')
    plt.plot(ts_fermion, uni_lambda1["FreeFermion"], marker='v', label='Free Fermions (t)')
    plt.xlabel("Control parameter")
    plt.ylabel("λ₁ (normalized Laplacian)")
    plt.title("Universal Behaviour of λ₁ Across Models (N=8)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("release/universality_lambda1.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas_xxz, uni_gap["XXZ"], marker='o', label='XXZ (Δ)')
    plt.plot(hs_tfim, uni_gap["TFIM"], marker='s', label='TFIM (h)')
    plt.plot(J_strengths, uni_gap["SpinGlass"], marker='^', label='Spin Glass (J)')
    plt.plot(ts_fermion, uni_gap["FreeFermion"], marker='v', label='Free Fermions (t)')
    plt.xlabel("Control parameter")
    plt.ylabel("Spectral Gap λ₂ - λ₁")
    plt.title("Universal Laplacian Spectral Gap Across Models (N=8)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("release/universality_gap.png", dpi=300)

    plt.figure(figsize=(8, 5))
    plt.plot(deltas_xxz, uni_alpha["XXZ"], marker='o', label='XXZ (Δ)')
    plt.plot(hs_tfim, uni_alpha["TFIM"], marker='s', label='TFIM (h)')
    plt.plot(J_strengths, uni_alpha["SpinGlass"], marker='^', label='Spin Glass (J)')
    plt.plot(ts_fermion, uni_alpha["FreeFermion"], marker='v', label='Free Fermions (t)')
    plt.xlabel("Control parameter")
    plt.ylabel("MI decay exponent α")
    plt.title("Universal Entanglement Decay Across Models (N=8)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("release/universality_mi_decay.png", dpi=300)

    # ============================================================
    # Spectral embeddings (N=8, representative parameters)
    # ============================================================

    N_embed = 8
    delta_crit = 1.0
    h_crit = 1.0
    J_strength_embed = 1.0
    t_embed = 1.0

    base_J = base_J_dict[N_embed]
    J_matrix_embed = J_strength_embed * base_J

    configs = [
        ("XXZ", xxz_hamiltonian(N_embed, delta_crit)),
        ("TFIM", tfim_hamiltonian(N_embed, h_crit)),
        ("SpinGlass", spin_glass_hamiltonian(N_embed, J_matrix_embed, h=0.0)),
        ("FreeFermion", free_fermion_hamiltonian(N_embed, t_embed)),
    ]

    plt.figure(figsize=(10, 10))
    for idx, (name, H) in enumerate(configs, start=1):
        eigvals, eigvecs = la.eigh(H)
        ground_state = eigvecs[:, 0]
        MI = mutual_information_matrix(ground_state, N_embed)
        spectrum, eigvecs_L = laplacian_spectrum_and_vectors(MI)
        x, y = spectral_embedding(eigvecs_L)

        ax = plt.subplot(2, 2, idx)
        ax.scatter(x, y, c=np.arange(N_embed), cmap='viridis', s=60)
        for i in range(N_embed):
            ax.text(x[i], y[i], str(i), fontsize=8, ha='center', va='center')
        ax.set_title(f"{name} Spectral Embedding (N={N_embed})")
        ax.set_xlabel("v₁(i)")
        ax.set_ylabel("v₂(i)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("release/universality_embeddings.png", dpi=300)

    print("\nRelease data written to 'release/' directory.")
    print("CSV files: xxz_N*.csv, tfim_N*.csv, spinglass_N*.csv, freefermion_N*.csv")
    print("PNG files: universality_*.png, universality_embeddings.png")


if __name__ == "__main__":
    run_suite()
