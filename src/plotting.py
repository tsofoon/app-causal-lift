"""Shared chart styling: thin marks, recessive axes, one accent color for the story."""

import matplotlib.pyplot as plt

TREATED = "#2a78d6"   # blue: treated users / the estimate being discussed
CONTROL = "#8c8b86"   # neutral gray: control users / reference
ALT = "#eb6834"       # orange: second series when a comparison needs one
TRUTH = "#0b0b0b"     # the known ground truth
INK_2 = "#52514e"     # secondary text


def set_style():
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.2),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c9c8c2",
            "axes.labelcolor": INK_2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.color": "#ecebe7",
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def truth_line(ax, value, orientation="h", label="True ATT"):
    if orientation == "h":
        ax.axhline(value, color=TRUTH, lw=1.2, ls="--", zorder=1)
        ax.annotate(f"{label} = {value:.3f}", xy=(1, value), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=TRUTH, fontsize=9)
    else:
        ax.axvline(value, color=TRUTH, lw=1.2, ls="--", zorder=1)
        ax.annotate(f"{label} = {value:.3f}", xy=(value, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", color=TRUTH, fontsize=9)


def forest(ax, ests, highlight, xlim):
    """Point estimates with 95% CIs, one row per method; off-scale rows are labeled."""
    import numpy as np

    yy = np.arange(len(ests))[::-1]
    for yi, e in zip(yy, ests):
        color = TREATED if e.method == highlight else CONTROL
        if xlim[0] <= e.ci[0] and e.ci[1] <= xlim[1]:
            ax.plot(e.ci, [yi, yi], color=color, lw=2)
            ax.plot(e.att, yi, "o", color=color, ms=7)
            ax.text(e.ci[1] + 0.1, yi, f"${e.att:.2f}", va="center", color=INK_2, fontsize=9)
        else:
            sign = "−" if e.att < 0 else ""
            ax.text(xlim[0] + 0.1, yi, f"{sign}\\${abs(e.att):.0f}, 95% CI ± \\${1.96 * e.se:.0f} "
                    "(off scale)", va="center", color=INK_2, fontsize=9)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(yy, [e.method for e in ests])
    ax.set_xlim(*xlim)
    ax.set_xlabel("Estimated ATT: change in 28-day 3P spend per subscriber ($), 95% CI")
    ax.grid(axis="y", visible=False)
