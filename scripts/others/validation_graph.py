import os
import sys
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
sys.path.append(os.path.join(os.getcwd()))
from scripts.config import default_output_folder


def destination_folder():
    """Configurar a pasta de destino dos gráficos."""

    folder = os.path.join(default_output_folder, "graphs", "validation")

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def matploblib_config():
    """Configurar matplotlib."""

    plt.style.use("seaborn-paper")

    font_dir = [r"font\computer-modern"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "CMU Serif"

    axes = {
        "labelsize": 22,
        "titlesize": 18,
        "titleweight": "bold",
        "labelweight": "bold",
    }
    matplotlib.rc("axes", **axes)

    lines = {"linewidth": 2}
    matplotlib.rc("lines", **lines)

    legends = {"fontsize": 14}
    matplotlib.rc("legend", **legends)

    savefig = {"dpi": 300}
    matplotlib.rc("savefig", **savefig)

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        color=["004c6d", "7aaac6"],
        linestyle=["-", "--"]
    )
    matplotlib.rcParams["ytick.labelsize"] = 15
    matplotlib.rcParams["xtick.labelsize"] = 15


def generate_plot():
    x = []
    y = []
    yexp = []

    with open("scripts/others/data_bahkitiari.csv", encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i != 0:
                x.append(float(row[0]))
                y.append(float(row[1]))
                yexp.append(float(row[2]))

    fig, ax = plt.subplots(num="Bahktiari COP vs T6 (Experimental)", figsize=(9.2, 7))
    ax.set_xlabel(r"$ T_{6} $ (°C)")
    ax.set_ylabel("COP")
    ax.plot(x, y, label=r"$COP_{calc}$ Este Trabalho")
    ax.plot(x, yexp, label=r"$COP_{exp}$ Bakhtiari et al. (2011)")

    ax.legend()
    ax.set_ylim(
        0.95 * min([min(y), min(yexp)]),
        1.05 * max([max(y), max(yexp)]),
    )
    ax.grid()
    fig.savefig(os.path.join(destination_folder(), "COP-VS-T6_bahkitiari.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    matploblib_config()
    generate_plot()
