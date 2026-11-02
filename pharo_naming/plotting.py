"""Figure style shared by all plots.

Figures are drawn at the size at which they are printed in a two-column IEEE
paper, so the text in them is the size the reader sees, and saved as PGF
code (a .tex file) that the paper inputs: LaTeX draws them and sets their text
in the font of the paper.  Colors come from a palette checked for colour-vision deficiencies; line
styles repeat the distinction for greyscale printing.
"""

from __future__ import annotations


COLUMN_WIDTH = 3.5  # inches, one column of the IEEE conference format
TEXT_WIDTH = 7.16  # inches, both columns

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# One hue, light to dark, for ordered quantities.
SEQUENTIAL_RAMP = ("#e4eefb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

def pyplot():
    """Return ``matplotlib.pyplot`` configured for the paper's figures."""
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        # Text is typeset by LaTeX in the paper's font; measure it with Times.
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{mathptmx}",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
    })
    return plt


def version_axis(ax, versions: list[str], rotation: int = 0) -> None:
    """Label the x axis with the release sequence."""
    ax.set_xticks(range(len(versions)), versions, fontsize=6, rotation=rotation)
    ax.set_xlim(-0.5, len(versions) - 0.5)
    ax.set_xlabel("Pharo version")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def save_figure(fig, path) -> None:
    """Write a figure as PGF code, to be included with ``\\input`` (needs the
    ``pgf`` package).  Every element is vector: no image file is written."""
    fig.savefig(path, format="pgf")
