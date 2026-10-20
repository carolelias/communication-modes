"""Estilo comum das figuras (paleta categórica fixa, eixos discretos)."""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
# permite rodar cada script isoladamente sem instalar o pacote
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# paleta categórica em ordem fixa (nunca reciclada)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = "RdBu_r"  # campo real: dois polos + meio neutro
SEQUENTIAL = "Blues"  # intensidade: um só matiz, claro → escuro

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    }
)


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"salvo: {path.relative_to(FIG_DIR.parent)}")
