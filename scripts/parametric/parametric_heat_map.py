import os
import sys
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd()))
from scripts.config import default_output_folder


def destination_folder():
    """Configurar a pasta de destino dos gráficos."""

    folder = os.path.join(default_output_folder, "graphs")

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def set_matplotlib_globalconfig():
    """Configurar matplotlib."""

    plt.style.use("seaborn-paper")

    font_dir = [r"font\computer-modern"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "CMU Serif"

    axes = {
        "labelsize": 24,
        "titlesize": 16,
        "titleweight": "bold",
        "labelweight": "bold",
    }
    matplotlib.rc("axes", **axes)

    lines = {"linewidth": 2}
    matplotlib.rc("lines", **lines)

    legends = {"fontsize": 20}
    matplotlib.rc("legend", **legends)

    savefig = {"dpi": 300}
    matplotlib.rc("savefig", **savefig)

    font = {"size": 20, "weight": "bold"}
    matplotlib.rc("font", **font)

    matplotlib.rcParams["ytick.labelsize"] = 20
    matplotlib.rcParams["xtick.labelsize"] = 20
    matplotlib.rcParams["axes.grid"] = False


def generate_heatmap():
    """Gerar heatmap de resumo da análise paramétrica."""

    set_matplotlib_globalconfig()
    variables = [r"$\dot{m}_{9}$", r"$X_{CH_4}$", r"$T_{10}$", r"$T_{13}$",
                 r"$T_{19}$", r"$T_{22}$", "MR", r"$T_{34}$",
                 r"$\varepsilon_{u}$", r"$\varepsilon_{d}$"]
    parameters = ["EUF", r"$\dot{m}_{38}$", r"$\dot{Ex}_d$", r"$\psi$"]

    percents = np.array([[11.31821341, 891.5355839, 601.6265135, -1.555488967],
                        [-41.01964856, 205.6259085, 503.4539378, -40.12668897],
                        [-8.589027932, -12.00428289, 1.010252793, -7.619984046],
                        [13.33517626, 19.67599834, -1.366421406, 11.6561157],
                        [-3.605375751, -5.035828791, 0.426176311, -3.19916955],
                        [0.611138481, 0.855973569, -0.071267381, 0.541858838],
                        [55.13550184, 97.39713312, 4.120957777, 45.56824307],
                        [-29.39122489, -39.88965197, -15.89802051, 19.33715521],
                        [81.53255717, 159.9601031, 5.613445439, 99.09909395],
                        [134.1516606, 270.7009112, 8.155320459, 60.66865264]])

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(percents, cmap="coolwarm", vmin=-40, vmax=100, aspect=0.35)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(parameters)), labels=parameters)
    ax.set_yticks(np.arange(len(variables)), labels=variables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(variables)):
        for j in range(len(parameters)):
            text = ax.text(j, i, f"{percents[i, j]:.2f} %",
                           ha="center", va="center", color="w")

    fig.tight_layout()
    plt.tight_layout(w_pad=0.5)
    plt.savefig(os.path.join(destination_folder(), f"parametric-heat-map-libr.pdf"), bbox_inches='tight')


if __name__ == "__main__":
    generate_heatmap()
