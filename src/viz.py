"""Style plot konsisten (palet tervalidasi, mark tipis, grid tipis)."""
import matplotlib.pyplot as plt
import matplotlib as mpl

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE_LIGHT = "#b7d3f6"
MUTED = "#a9a8a2"
STATUS = {"Sehat": "#1baf7a", "Waspada": "#eda100", "Rentan": "#e34948"}


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "lines.linewidth": 2, "legend.frameon": False,
        "legend.fontsize": 9, "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 160,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def money_fmt(ax, axis="y"):
    def _f(x, _):
        if abs(x) >= 1e6:
            return f"${x/1e6:.1f}M"
        if abs(x) >= 1e4:
            return f"${x/1e3:.0f}K"
        if abs(x) >= 1e3:
            return f"${x/1e3:.1f}K".replace(".0K", "K")
        return f"${x:.0f}"
    f = mpl.ticker.FuncFormatter(_f)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)
