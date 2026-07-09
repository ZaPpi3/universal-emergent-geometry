"""
Generate all paper figures from the CSV/JSON/NPZ data products written by
the run_*.py scripts. Run this last, after all run_*.py scripts have
completed. Writes PNGs to ../figures.
"""
import sys, os, csv, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt

from lib.plotstyle import (apply_style, style_axes, MODEL_COLOR, MODEL_MARKER,
                            MODEL_LINESTYLE, MODEL_LABEL, SEQ_BLUE)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
apply_style()

MODELS = ["xxz", "tfim", "spin_glass", "free_fermion"]


def load_main_sweep():
    rows = {m: {"x": [], "lambda1": [], "lambda2": [], "gap": [], "alpha": [], "alpha_r2": []}
            for m in MODELS}
    with open(os.path.join(DATA_DIR, "main_sweep_N8.csv")) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            rows[m]["x"].append(float(row["x"]))
            rows[m]["lambda1"].append(float(row["lambda1"]))
            rows[m]["lambda2"].append(float(row["lambda2"]))
            rows[m]["gap"].append(float(row["gap"]))
            rows[m]["alpha"].append(float(row["alpha"]) if row["alpha"] != "nan" else np.nan)
            rows[m]["alpha_r2"].append(float(row["alpha_r2"]) if row["alpha_r2"] != "nan" else np.nan)
    for m in MODELS:
        for k in rows[m]:
            rows[m][k] = np.array(rows[m][k])
    return rows


# ------------------------------------------------------------------ Fig 1-3: overlay sweeps
def fig_overlay(rows, key, ylabel, fname, logy=False):
    fig, ax = plt.subplots(figsize=(3.6, 3.15))
    for m in MODELS:
        x, y = rows[m]["x"], rows[m][key]
        mask = np.isfinite(y)
        ax.plot(x[mask], y[mask], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                linestyle=MODEL_LINESTYLE[m], label=MODEL_LABEL[m],
                markevery=3, markerfacecolor="white", markeredgewidth=1.0)
    ax.set_xlabel(r"control parameter $x$" "\n" r"($\Delta$ for XXZ, $h$ for TFIM)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.axvline(1.0, color="#c3c2b7", linewidth=0.8, linestyle=":", zorder=0)
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2,
              columnspacing=1.0, handletextpad=0.4, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# ------------------------------------------------------------------ Fig 4: embeddings
def fig_embeddings():
    data = np.load(os.path.join(DATA_DIR, "embeddings_N8.npz"))
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5))
    for ax, m in zip(axes, MODELS):
        v1, v2 = data[f"{m}_v1"], data[f"{m}_v2"]
        ax.scatter(v1, v2, color=MODEL_COLOR[m], edgecolor="white", linewidth=0.6, s=40, zorder=3)
        ax.set_title(MODEL_LABEL[m], fontsize=9)
        ax.set_xlabel(r"$v_1$")
        if ax is axes[0]:
            ax.set_ylabel(r"$v_2$")
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "universality_embeddings.png"))
    plt.close(fig)
    print("wrote universality_embeddings.png")


# ------------------------------------------------------------------ Fig 5: finite-size table plot
def fig_finite_size():
    NS = [6, 8, 10, 12]
    series = {}
    with open(os.path.join(DATA_DIR, "finite_size_table.csv")) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            series[row[0]] = [float(v) for v in row[1:]]

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    style_map = {
        "XXZ (integrable), Delta=0": ("xxz", "-", r"XXZ (integrable), $\Delta=0$"),
        "XXZ (critical), Delta=1": ("xxz", "--", r"XXZ (critical), $\Delta=1$"),
        "TFIM (critical), h=1": ("tfim", "-", r"TFIM (critical), $h=1$"),
        "Free fermions, t=0.2": ("free_fermion", "-", r"Free fermions, $t=0.2$"),
        "Spin glass, any J": ("spin_glass", "-", r"Spin glass, any $J$"),
    }
    for label, vals in series.items():
        color_key, ls, disp_label = style_map[label]
        ax.plot(NS, vals, marker="o", linestyle=ls, color=MODEL_COLOR[color_key],
                 label=disp_label, markerfacecolor="white")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\lambda_1$")
    style_axes(ax)
    ax.legend(fontsize=6.2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "finite_size_scaling.png"))
    plt.close(fig)
    print("wrote finite_size_scaling.png")


# ------------------------------------------------------------------ Fig 6: critical scaling + residuals (3 forms)
def fig_critical_scaling(bc="obc"):
    with open(os.path.join(DATA_DIR, "scaling_fits.json")) as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), height_ratios=[2.2, 1])
    for col, model in enumerate(["tfim", "xxz"]):
        r = results[f"{model}_{bc}"]
        N = np.array(r["N"], dtype=float)
        y = np.array(r["y"], dtype=float)
        Nfine = np.linspace(N.min(), N.max(), 200)

        ax = axes[0, col]
        ax.plot(N, y, "o", color=MODEL_COLOR[model], markerfacecolor="white",
                 markeredgewidth=1.3, label="exact ED", zorder=5)

        pw, lg, sat = r["power"], r["log"], r["saturating"]
        ax.plot(Nfine, pw["a"] * Nfine ** (-pw["b"]), color="#898781",
                 linestyle="--", label=fr"power law ($b$={pw['b']:.2f})")
        ax.plot(Nfine, (lg["a"] / Nfine) * (1 + lg["c"] / np.log(Nfine)),
                 color=MODEL_COLOR[model], linestyle="-",
                 label=fr"log-corrected ($c$={lg['c']:.2f})")
        ax.plot(Nfine, sat["yinf"] + sat["a"] * Nfine ** (-sat["b"]),
                 color="#eda100", linestyle="-.",
                 label=fr"saturating ($y_\infty$={sat['yinf']:.2f})")
        ax.set_title(f"{MODEL_LABEL[model]} critical point ({bc.upper()})", fontsize=9)
        ax.set_ylabel(r"$\lambda_1(N)$")
        style_axes(ax)
        ax.legend(fontsize=6.5, loc="upper right")

        ax2 = axes[1, col]
        resid_pw = np.array(pw["resid"])
        best_key = "log" if model == "xxz" else "saturating"
        best = r[best_key]
        resid_best = np.array(best["resid"])
        ax2.axhline(0, color="#c3c2b7", linewidth=0.8)
        ax2.plot(N, resid_pw, "o--", color="#898781", markerfacecolor="white",
                  label="power law", markersize=4)
        ax2.plot(N, resid_best, "s-", color=MODEL_COLOR[model],
                  label=("log-corrected" if model == "xxz" else "saturating"), markersize=4)
        ax2.set_xlabel(r"$N$")
        ax2.set_ylabel("residual")
        style_axes(ax2)
        ax2.legend(fontsize=6.5)

    fig.tight_layout()
    fname = f"critical_scaling_extended_{bc}.png"
    fig.savefig(os.path.join(FIG_DIR, fname))
    plt.close(fig)
    print("wrote", fname)


# ------------------------------------------------------------------ Fig 7: central charge
def fig_central_charge():
    with open(os.path.join(DATA_DIR, "central_charge_fit.json")) as f:
        results = json.load(f)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for model, label_c in [("tfim", "1/2"), ("xxz", "1")]:
        r = results[model]
        N = np.array(r["N"])
        S = np.array(r["S"])
        x = np.log(N)
        ax.plot(x, S, "o", color=MODEL_COLOR[model], markerfacecolor="white",
                 markeredgewidth=1.3, label=fr"{MODEL_LABEL[model]}, fit $c$={r['c']:.3f} (exact {label_c})")
        xfine = np.linspace(x.min(), x.max(), 50)
        ax.plot(xfine, (r["c"] / 3) * xfine + r["const"], color=MODEL_COLOR[model], linewidth=1.0)
    ax.set_xlabel(r"$\ln N$")
    ax.set_ylabel(r"$S_{\rm half}$ (PBC)")
    style_axes(ax)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "central_charge_fit.png"))
    plt.close(fig)
    print("wrote central_charge_fit.png")


# ------------------------------------------------------------------ Fig 8: TFIM gap collapse
def fig_gap_collapse():
    Ns, hs, gaps = [], [], []
    with open(os.path.join(DATA_DIR, "tfim_gap_collapse.csv")) as f:
        for row in csv.DictReader(f):
            Ns.append(float(row["N"])); hs.append(float(row["h"])); gaps.append(float(row["gap"]))
    Ns, hs, gaps = np.array(Ns), np.array(hs), np.array(gaps)
    uniqueN = sorted(set(Ns))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    cmap_vals = np.linspace(0, 1, len(uniqueN))
    for i, N in enumerate(uniqueN):
        mask = Ns == N
        color = SEQ_BLUE[min(int(cmap_vals[i] * (len(SEQ_BLUE) - 1)), len(SEQ_BLUE) - 1)]
        order = np.argsort(hs[mask])
        h_, g_ = hs[mask][order], gaps[mask][order]
        axes[0].plot(h_, g_, "-o", color=color, markersize=3, label=f"N={int(N)}")
        axes[1].plot((h_ - 1.0) * N, N * g_, "-o", color=color, markersize=3, label=f"N={int(N)}")

    axes[0].set_xlabel(r"$h$")
    axes[0].set_ylabel(r"$\Delta E(h,N)$")
    axes[0].set_title("raw energy gap", fontsize=9)
    axes[1].set_xlabel(r"$(h-1)\,N$")
    axes[1].set_ylabel(r"$N \cdot \Delta E$")
    axes[1].set_title(r"collapse, $\nu=1$ (Ising)", fontsize=9)
    for ax in axes:
        style_axes(ax)
    axes[1].legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tfim_gap_collapse.png"))
    plt.close(fig)
    print("wrote tfim_gap_collapse.png")


# ------------------------------------------------------------------ Fig 9: delta sweep (marginal operator localization)
def fig_delta_sweep():
    deltas, c_log, r2_log, r2_pow = [], [], [], []
    with open(os.path.join(DATA_DIR, "delta_sweep_fits.csv")) as f:
        for row in csv.DictReader(f):
            deltas.append(float(row["delta"]))
            c_log.append(float(row["c_log"]))
            r2_log.append(float(row["r2_log"]))
            r2_pow.append(float(row["r2_power"]))
    deltas = np.array(deltas)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(deltas, np.abs(c_log), "o-", color=MODEL_COLOR["xxz"])
    axes[0].set_xlabel(r"$\Delta$")
    axes[0].set_ylabel(r"$|c|$ (log-correction coefficient)")
    axes[0].set_title("marginal-operator strength", fontsize=9)

    axes[1].plot(deltas, r2_pow, "o--", color="#898781", label="power law")
    axes[1].plot(deltas, r2_log, "s-", color=MODEL_COLOR["xxz"], label="log-corrected")
    axes[1].set_xlabel(r"$\Delta$")
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_title("fit quality vs. $\\Delta$", fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "delta_sweep_marginal_operator.png"))
    plt.close(fig)
    print("wrote delta_sweep_marginal_operator.png")


def main():
    rows = load_main_sweep()
    fig_overlay(rows, "lambda1", r"$\lambda_1$", "universality_lambda1.png")
    fig_overlay(rows, "gap", r"$\Delta_{\rm gap}$", "universality_gap.png")
    fig_overlay(rows, "alpha", r"$\alpha$ (MI decay exponent)", "universality_mi_decay.png")
    fig_embeddings()
    fig_finite_size()
    fig_critical_scaling("obc")
    fig_critical_scaling("pbc")
    fig_central_charge()
    fig_gap_collapse()
    fig_delta_sweep()


if __name__ == "__main__":
    main()
