"""
Runs every experiment behind the paper, in dependency order, writing the
same data/*.csv|json|npz files the original per-experiment scripts wrote.

This is a straight merge of run_main_sweep.py, run_finite_size.py,
run_large_n_scaling.py, run_large_n_scaling_pbc.py, run_scaling_fits.py,
run_delta_sweep.py, run_central_charge_fit.py, run_scaling_collapse.py, and
run_dmrg_crosscheck.py into one file, done purely to cut the file count for
the public release - no computation changed (the two truly duplicated
lambda1(N, delta) helpers across the finite-size and delta-sweep stages are
now a single shared function; every other stage keeps its own namespaced
helpers to avoid any behavioural cross-talk).

run_pin_sensitivity.py is intentionally kept as its own standalone script:
main.tex Sec. II.B names it directly by filename as the source of a specific
quoted robustness number, so merging it would leave the paper citing a file
that no longer exists.

Run with: python run_all.py
Then:     python run_pin_sensitivity.py    (separate, see above)
          python make_figures.py           (builds figures/ from data/)

The DMRG cross-check stage (Table I) requires `quimb`; if it isn't
installed, that stage is skipped with a message instead of failing the rest
of the pipeline - every other stage, including all N=20 results, uses exact
sparse diagonalization and has no tensor-network dependency.
"""
import sys, os, csv, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import scipy.sparse.linalg as spla

from lib import hamiltonians as H
from lib import entanglement as E
from lib import laplacian as L
from fits import fit_power_law, fit_log_corrected, fit_saturating, windowed_fit_stability

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ============================================================ shared helpers
def _lambda1_xxz(N, delta):
    """Shared by stage_finite_size() and stage_delta_sweep() - identical in
    both original scripts, so defined once here instead of twice."""
    Hs = H.build_xxz(N, delta=delta)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    return sq["lambda1"]


# ============================================================ [1/9] main N=8 sweep
# originally run_main_sweep.py
MAIN_SWEEP_N = 8
MAIN_SWEEP_XS = np.round(np.linspace(0.02, 2.0, 41), 4)
MAIN_SWEEP_XS = np.unique(np.sort(np.append(MAIN_SWEEP_XS, 1.0)))  # critical pt on-grid


def _main_sweep_point_xxz(x):
    Hs = H.build_xxz(MAIN_SWEEP_N, delta=x)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, MAIN_SWEEP_N)
    return I


def _main_sweep_point_tfim(x):
    Hs = H.build_tfim(MAIN_SWEEP_N, h=x, pin=1e-6)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, MAIN_SWEEP_N)
    return I


def _main_sweep_point_spin_glass(seed):
    diag = H.build_spin_glass_diag(MAIN_SWEEP_N, seed=seed)
    e0, psi = E.ground_state_diag(diag)
    I, _ = E.mutual_information_matrix(psi, MAIN_SWEEP_N)
    return I


def _main_sweep_point_free_fermion(x):
    Hs = H.build_free_fermion(MAIN_SWEEP_N, t=x)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, MAIN_SWEEP_N)
    return I


def _main_sweep_analyze(I):
    lap = L.normalized_laplacian(I)
    sq = L.spectral_quantities(lap)
    alpha, r, Iavg, r2 = L.mi_decay_exponent(I)
    return sq, alpha, r2


def stage_main_sweep():
    print("\n=== [1/9] main N=8 parameter sweep (was run_main_sweep.py) ===")
    XS = MAIN_SWEEP_XS
    rows = []
    embeddings = {}

    t0 = time.time()
    for x in XS:
        sq, alpha, r2 = _main_sweep_analyze(_main_sweep_point_xxz(x))
        rows.append(("xxz", x, sq["lambda1"], sq["lambda2"], sq["gap"], alpha, r2))
        if abs(x - 1.0) < 1e-9:
            embeddings["xxz"] = (sq["v1"], sq["v2"])
    print(f"xxz sweep done in {time.time()-t0:.2f}s")

    t0 = time.time()
    for x in XS:
        sq, alpha, r2 = _main_sweep_analyze(_main_sweep_point_tfim(x))
        rows.append(("tfim", x, sq["lambda1"], sq["lambda2"], sq["gap"], alpha, r2))
        if abs(x - 1.0) < 1e-9:
            embeddings["tfim"] = (sq["v1"], sq["v2"])
    print(f"tfim sweep done in {time.time()-t0:.2f}s")

    t0 = time.time()
    for i, x in enumerate(XS):
        sq, alpha, r2 = _main_sweep_analyze(_main_sweep_point_spin_glass(seed=1000 + i))
        rows.append(("spin_glass", x, sq["lambda1"], sq["lambda2"], sq["gap"], alpha, r2))
        if i == len(XS) // 2:
            embeddings["spin_glass"] = (sq["v1"], sq["v2"])
    print(f"spin_glass sweep done in {time.time()-t0:.2f}s")

    t0 = time.time()
    for x in XS:
        sq, alpha, r2 = _main_sweep_analyze(_main_sweep_point_free_fermion(x))
        rows.append(("free_fermion", x, sq["lambda1"], sq["lambda2"], sq["gap"], alpha, r2))
        if abs(x - 1.0) < 1e-9:
            embeddings["free_fermion"] = (sq["v1"], sq["v2"])
    print(f"free_fermion sweep done in {time.time()-t0:.2f}s")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "main_sweep_N8.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "x", "lambda1", "lambda2", "gap", "alpha", "alpha_r2"])
        w.writerows(rows)

    np.savez(os.path.join(DATA_DIR, "embeddings_N8.npz"),
              **{f"{k}_v1": v[0] for k, v in embeddings.items()},
              **{f"{k}_v2": v[1] for k, v in embeddings.items()})

    ff_rows = [r for r in rows if r[0] == "free_fermion"]
    l1_ff = np.array([r[2] for r in ff_rows])
    print("free-fermion lambda1 spread across x (should be ~0):", l1_ff.max() - l1_ff.min())
    sg_rows = [r for r in rows if r[0] == "spin_glass"]
    l1_sg = np.array([r[2] for r in sg_rows])
    print("spin-glass lambda1 spread across x (should be exactly 0):", l1_sg.max() - l1_sg.min())
    print("wrote", os.path.join(DATA_DIR, "main_sweep_N8.csv"))


# ============================================================ [2/9] finite-size table
# originally run_finite_size.py
FINITE_SIZE_NS = [6, 8, 10, 12]


def _finite_size_lambda1_tfim(N, h):
    Hs = H.build_tfim(N, h=h, pin=1e-6)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    return sq["lambda1"]


def _finite_size_lambda1_ff(N, t=0.2):
    Hs = H.build_free_fermion(N, t=t)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    return sq["lambda1"]


def _finite_size_lambda1_sg(N, seed=1):
    diag = H.build_spin_glass_diag(N, seed=seed)
    e0, psi = E.ground_state_diag(diag)
    I, _ = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    return sq["lambda1"]


def stage_finite_size():
    print("\n=== [2/9] finite-size table, N=6,8,10,12 (was run_finite_size.py) ===")
    rows = {
        "XXZ (integrable), Delta=0": [],
        "XXZ (critical), Delta=1": [],
        "TFIM (critical), h=1": [],
        "Free fermions, t=0.2": [],
        "Spin glass, any J": [],
    }
    for N in FINITE_SIZE_NS:
        rows["XXZ (integrable), Delta=0"].append(_lambda1_xxz(N, 0.0))
        rows["XXZ (critical), Delta=1"].append(_lambda1_xxz(N, 1.0))
        rows["TFIM (critical), h=1"].append(_finite_size_lambda1_tfim(N, 1.0))
        rows["Free fermions, t=0.2"].append(_finite_size_lambda1_ff(N, 0.2))
        rows["Spin glass, any J"].append(_finite_size_lambda1_sg(N))
        print(f"N={N} done")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "finite_size_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series"] + [f"N={n}" for n in FINITE_SIZE_NS])
        for k, v in rows.items():
            w.writerow([k] + [f"{val:.6f}" for val in v])

    xxz0 = np.array(rows["XXZ (integrable), Delta=0"])
    ff = np.array(rows["Free fermions, t=0.2"])
    print("max |XXZ(Delta=0) - free fermion| across N:", np.max(np.abs(xxz0 - ff)))
    print("wrote", os.path.join(DATA_DIR, "finite_size_table.csv"))


# ============================================================ [3-4/9] large-N scaling
# originally run_large_n_scaling.py + run_large_n_scaling_pbc.py
LARGE_N_NS = [6, 8, 10, 12, 14, 16, 18, 20]


def _large_n_scaling_point(model, N, pbc=False):
    if model == "xxz":
        Hs = H.build_xxz(N, delta=1.0, pbc=pbc)
    elif model == "tfim":
        Hs = H.build_tfim(N, h=1.0, pin=1e-6, pbc=pbc)
    else:
        raise ValueError(model)
    e0, psi = E.ground_state_sparse(Hs)
    I, S1 = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    Shalf = E.half_chain_entropy(psi, N)
    return dict(model=model, N=N, pbc=pbc, E0=e0, lambda1=sq["lambda1"],
                lambda2=sq["lambda2"], gap=sq["gap"], S_half=Shalf)


def stage_large_n_scaling_obc():
    print("\n=== [3/9] extended critical-point scaling to N=20, OBC "
          "(was run_large_n_scaling.py) ===")
    rows = []
    for model in ["xxz", "tfim"]:
        for N in LARGE_N_NS:
            t0 = time.time()
            r = _large_n_scaling_point(model, N, pbc=False)
            rows.append(r)
            print(f"{model} N={N}: lambda1={r['lambda1']:.6f} "
                  f"S_half={r['S_half']:.6f}  ({time.time()-t0:.1f}s)")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "large_n_scaling_obc.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def stage_large_n_scaling_pbc():
    print("\n=== [4/9] extended critical-point scaling to N=20, PBC "
          "(was run_large_n_scaling_pbc.py) ===")
    rows = []
    for model in ["xxz", "tfim"]:
        for N in LARGE_N_NS:
            r = _large_n_scaling_point(model, N, pbc=True)
            rows.append(r)
            print(f"[PBC] {model} N={N}: lambda1={r['lambda1']:.6f} S_half={r['S_half']:.6f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "large_n_scaling_pbc.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


# ============================================================ [5/9] scaling-form fits
# originally run_scaling_fits.py
def _scaling_fits_load(model, bc="obc"):
    fname = f"large_n_scaling_{bc}.csv"
    Ns, l1s = [], []
    with open(os.path.join(DATA_DIR, fname)) as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                Ns.append(float(row["N"]))
                l1s.append(float(row["lambda1"]))
    return np.array(Ns), np.array(l1s)


def _scaling_fits_analyze(model, bc):
    N, y = _scaling_fits_load(model, bc)
    pw = fit_power_law(N, y)
    lg = fit_log_corrected(N, y)
    sat = fit_saturating(N, y)
    stab_pw = windowed_fit_stability(N, y, fit_power_law)
    stab_lg = windowed_fit_stability(N, y, fit_log_corrected)
    order = np.argsort(N)
    Ns, ys = N[order], y[order]
    stab_yinf = []
    for start in range(len(Ns) - 4 + 1):
        Nw, yw = Ns[start:], ys[start:]
        if len(Nw) < 4:
            break
        f = fit_saturating(Nw, yw)
        stab_yinf.append((float(Nw[0]), f["yinf"], f["yinf_err"], f["r2"]))

    print(f"\n=== {model} ({bc}) ===")
    print(f"power-law:      b={pw['b']:.4f}  R2={pw['r2']:.8f}")
    print(f"log-corrected:  c={lg['c']:.4f}  a={lg['a']:.4f}  R2={lg['r2']:.8f}")
    print(f"saturating:     yinf={sat['yinf']:.4f}+/-{sat['yinf_err']:.4f}  "
          f"a={sat['a']:.4f}  b={sat['b']:.4f}  R2={sat['r2']:.8f}")
    print("windowed b (power-law), (N_min, b, R2):")
    for row in stab_pw:
        print(f"    N_min={row[0]:.0f}  b={row[1]:.4f}  R2={row[2]:.7f}")
    print("windowed c (log-corrected), (N_min, c, R2):")
    for row in stab_lg:
        print(f"    N_min={row[0]:.0f}  c={row[1]:.4f}  R2={row[2]:.7f}")
    print("windowed yinf (saturating), (N_min, yinf, yinf_err, R2):")
    for row in stab_yinf:
        print(f"    N_min={row[0]:.0f}  yinf={row[1]:.4f}+/-{row[2]:.4f}  R2={row[3]:.7f}")

    return dict(model=model, bc=bc, N=N.tolist(), y=y.tolist(),
                power=pw, log=lg, saturating=sat,
                windowed_power=stab_pw, windowed_log=stab_lg, windowed_yinf=stab_yinf)


def stage_scaling_fits():
    print("\n=== [5/9] scaling-form fits at both critical points (was run_scaling_fits.py) ===")
    results = {}
    for bc in ["obc", "pbc"]:
        fname = os.path.join(DATA_DIR, f"large_n_scaling_{bc}.csv")
        if not os.path.exists(fname):
            print(f"skipping {bc}: {fname} not found yet")
            continue
        for model in ["xxz", "tfim"]:
            key = f"{model}_{bc}"
            r = _scaling_fits_analyze(model, bc)
            r_json = dict(r)
            r_json["power"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in r["power"].items()}
            r_json["log"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in r["log"].items()}
            r_json["saturating"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in r["saturating"].items()}
            results[key] = r_json

    with open(os.path.join(DATA_DIR, "scaling_fits.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", os.path.join(DATA_DIR, "scaling_fits.json"))


# ============================================================ [6/9] delta sweep
# originally run_delta_sweep.py
DELTA_SWEEP_NS = [8, 10, 12, 14, 16, 18, 20]
DELTA_SWEEP_DELTAS = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def stage_delta_sweep():
    print("\n=== [6/9] marginal-operator localization delta-sweep (was run_delta_sweep.py) ===")
    rows = []
    for delta in DELTA_SWEEP_DELTAS:
        l1s = []
        for N in DELTA_SWEEP_NS:
            l1 = _lambda1_xxz(N, delta)
            l1s.append(l1)
            rows.append(dict(delta=delta, N=N, lambda1=l1))
        print(f"delta={delta}: lambda1(N) = {['%.5f'%v for v in l1s]}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "delta_sweep_raw.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["delta", "N", "lambda1"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)

    fit_rows = []
    ns = np.array(DELTA_SWEEP_NS, dtype=float)
    for delta in DELTA_SWEEP_DELTAS:
        l1 = np.array([r["lambda1"] for r in rows if r["delta"] == delta])
        pw = fit_power_law(ns, l1)
        lg = fit_log_corrected(ns, l1)
        fit_rows.append(dict(delta=delta, b_power=pw["b"], r2_power=pw["r2"],
                              c_log=lg["c"], r2_log=lg["r2"]))
        print(f"delta={delta}: power-law b={pw['b']:.4f} R2={pw['r2']:.7f} | "
              f"log-corr c={lg['c']:.4f} R2={lg['r2']:.7f}")

    out2 = os.path.join(DATA_DIR, "delta_sweep_fits.csv")
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fit_rows[0].keys()))
        w.writeheader()
        w.writerows(fit_rows)
    print("wrote", out2)


# ============================================================ [7/9] central-charge fit
# originally run_central_charge_fit.py
def _central_charge_load(model):
    Ns, Ss = [], []
    with open(os.path.join(DATA_DIR, "large_n_scaling_pbc.csv")) as f:
        for row in csv.DictReader(f):
            if row["model"] == model:
                Ns.append(float(row["N"]))
                Ss.append(float(row["S_half"]))
    return np.array(Ns), np.array(Ss)


def _fit_central_charge(N, S):
    x = np.log(N)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    slope, intercept = coef
    c = 3 * slope
    yhat = A @ coef
    resid = S - yhat
    r2 = 1 - np.sum(resid ** 2) / np.sum((S - S.mean()) ** 2)
    return dict(c=float(c), const=float(intercept), r2=float(r2))


def stage_central_charge_fit():
    print("\n=== [7/9] Calabrese-Cardy central-charge fit (was run_central_charge_fit.py) ===")
    results = {}
    expected = {"tfim": 0.5, "xxz": 1.0}
    for model in ["tfim", "xxz"]:
        N, S = _central_charge_load(model)
        fit = _fit_central_charge(N, S)
        results[model] = dict(N=N.tolist(), S=S.tolist(), **fit, expected_c=expected[model])
        print(f"{model}: fitted c = {fit['c']:.4f}  (expected {expected[model]}, "
              f"CFT: {'Ising' if model=='tfim' else 'SU(2) Heisenberg'})  "
              f"R2={fit['r2']:.7f}")

    with open(os.path.join(DATA_DIR, "central_charge_fit.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(DATA_DIR, "central_charge_fit.json"))


# ============================================================ [8/9] TFIM gap collapse
# originally run_scaling_collapse.py
TFIM_COLLAPSE_NS = [8, 10, 12, 14, 16, 18, 20]
TFIM_COLLAPSE_HS = np.round(np.linspace(0.85, 1.15, 13), 4)


def _tfim_physical_gap(N, h):
    Hs = H.build_tfim(N, h=h, pin=0.0)  # pin=0: exact critical scaling, no symmetry-breaking needed
    evals = spla.eigsh(Hs, k=2, which="SA", return_eigenvectors=False)
    evals = np.sort(evals)
    return float(evals[1] - evals[0])


def stage_scaling_collapse():
    print("\n=== [8/9] TFIM energy-gap scaling collapse (was run_scaling_collapse.py) ===")
    rows = []
    for N in TFIM_COLLAPSE_NS:
        for h in TFIM_COLLAPSE_HS:
            g = _tfim_physical_gap(N, h)
            rows.append(dict(N=N, h=h, gap=g, N_gap=N * g, x_scaled=(h - 1.0) * N))
            print(f"N={N} h={h}: gap={g:.6f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "tfim_gap_collapse.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


# ============================================================ [9/9] DMRG cross-check
# originally run_dmrg_crosscheck.py - optional, requires `quimb`
def stage_dmrg_crosscheck():
    print("\n=== [9/9] DMRG cross-check, Table I (was run_dmrg_crosscheck.py) ===")
    try:
        from lib import dmrg_utils as D
    except ImportError:
        print("quimb is not installed - skipping the DMRG cross-check stage.")
        print("Install it with `pip install quimb` to reproduce Table I; every other")
        print("stage (including all N=20 results) has no dependency on it.")
        return

    def run_point(model, N, param):
        if model == "xxz":
            Hs = H.build_xxz(N, delta=param)
            mpo = D.xxz_mpo(N, delta=param)
        elif model == "tfim":
            Hs = H.build_tfim(N, h=param, pin=1e-6)
            mpo = D.tfim_mpo(N, h=param, pin=1e-6)
        else:
            raise ValueError(model)

        e_ed, psi_ed = E.ground_state_sparse(Hs)
        I_ed, _ = E.mutual_information_matrix(psi_ed, N)
        lap_ed = L.normalized_laplacian(I_ed)
        sq_ed = L.spectral_quantities(lap_ed)

        t0 = time.time()
        e_dmrg, psi_dmrg, max_chi, converged = D.dmrg_ground_state(mpo)
        t_dmrg = time.time() - t0

        phase = np.vdot(psi_ed, psi_dmrg)
        phase = phase / abs(phase) if abs(phase) > 0 else 1.0
        psi_dmrg_aligned = psi_dmrg / phase

        psi_err = float(np.max(np.abs(psi_ed - psi_dmrg_aligned)))

        I_dmrg, _ = E.mutual_information_matrix(psi_dmrg_aligned, N)
        lap_dmrg = L.normalized_laplacian(I_dmrg)
        sq_dmrg = L.spectral_quantities(lap_dmrg)

        return dict(
            model=model, N=N, param=param,
            E0_ed=e_ed, E0_dmrg=e_dmrg, E0_absdiff=abs(e_ed - e_dmrg),
            lambda1_ed=sq_ed["lambda1"], lambda1_dmrg=sq_dmrg["lambda1"],
            lambda1_absdiff=abs(sq_ed["lambda1"] - sq_dmrg["lambda1"]),
            psi_maxdiff=psi_err, I_maxdiff=float(np.max(np.abs(I_ed - I_dmrg))),
            max_bond_dim=max_chi, dmrg_converged=converged, dmrg_time_s=t_dmrg,
        )

    rows = []
    for N in [8, 10, 12, 14]:
        rows.append(run_point("xxz", N, 1.0))
        rows.append(run_point("tfim", N, 1.0))
        print(f"N={N} done: "
              f"xxz lambda1 diff={rows[-2]['lambda1_absdiff']:.2e}, "
              f"tfim lambda1 diff={rows[-1]['lambda1_absdiff']:.2e}")

    out = os.path.join(DATA_DIR, "dmrg_crosscheck.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)


# ============================================================ entry point
def main():
    t_start = time.time()
    stage_main_sweep()
    stage_finite_size()
    stage_large_n_scaling_obc()
    stage_large_n_scaling_pbc()
    stage_scaling_fits()
    stage_delta_sweep()
    stage_central_charge_fit()
    stage_scaling_collapse()
    stage_dmrg_crosscheck()
    print(f"\nAll stages complete in {time.time()-t_start:.1f}s.")
    print("Next: `python run_pin_sensitivity.py` (separate script, see module docstring),")
    print("then `python make_figures.py` to build figures/ from data/.")


if __name__ == "__main__":
    main()
