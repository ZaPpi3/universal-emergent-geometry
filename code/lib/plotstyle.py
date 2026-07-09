"""
Shared matplotlib styling for all paper figures: a fixed, CVD-validated
categorical palette (one hue per model, never reassigned/cycled), paired
with distinct markers/linestyles per model so figures remain legible in
black-and-white print. Serif math/text to match the revtex4-2 typeset body.
"""
import matplotlib as mpl

# Fixed categorical assignment (validated CVD-safe palette; order preserved
# from the reference 8-hue sequence - slots 1, 6, 4, 5).
MODEL_COLOR = {
    "xxz": "#2a78d6",         # slot 1, blue
    "tfim": "#e34948",        # slot 6, red
    "spin_glass": "#008300",  # slot 4, green
    "free_fermion": "#4a3aa7",  # slot 5, violet
}
MODEL_MARKER = {
    "xxz": "o",
    "tfim": "s",
    "spin_glass": "^",
    "free_fermion": "D",
}
MODEL_LINESTYLE = {
    "xxz": "-",
    "tfim": "--",
    "spin_glass": ":",
    "free_fermion": "-.",
}
MODEL_LABEL = {
    "xxz": "XXZ",
    "tfim": "TFIM",
    "spin_glass": "Spin glass (control)",
    "free_fermion": "Free fermions",
}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "svg.fonttype": "none",
    })


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)
