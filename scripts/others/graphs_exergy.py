import os
import sys
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.append(os.path.join(os.getcwd()))
from scripts.config import default_output_folder


def destination_folder():
    """Configurar a pasta de destino dos gráficos."""

    folder = os.path.join(default_output_folder, "graphs", "exergy")

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
        "labelsize": 32,
        "titlesize": 28,
        "titleweight": "bold",
        "labelweight": "bold",
    }
    matplotlib.rc("axes", **axes)

    lines = {"linewidth": 2}
    matplotlib.rc("lines", **lines)

    legends = {"fontsize": 22}
    matplotlib.rc("legend", **legends)

    savefig = {"dpi": 300}
    matplotlib.rc("savefig", **savefig)

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        color=["8fbed9", "004c6d"]
    )
    matplotlib.rcParams["ytick.labelsize"] = 28
    matplotlib.rcParams["xtick.labelsize"] = 28

    matplotlib.rcParams["axes.grid"] = True


def exd_graph():
    """Gráfico das destruições de exergia dos volumes de controle do sistema."""

    exd_k_libr = {
        "compressor": 15.8719014,
        "regenerador": 35.9994965,
        "cc": 163.205706,
        "turbina": 16.4252122,
        "absorvedor": 0.620634485,
        "gerador": 4.52427394,
        "condensador": 0.520269154,
        "evaporador": 0.364249391,
        "vs": 0.002141506,
        "vr": 0.04672878,
        "hx": 0.154546269,
        "bomba": 1.30749E-05,
        "retificador": 0,
        "rhx": 0,
        "umidificador": 8.41433196,
        "desumidificador": 10.2478122,
        "aquecedor": 39.5269246
    }

    exd_k_nh3 = {
        "compressor": 15.8719014,
        "regenerador": 35.9994965,
        "cc": 163.205706,
        "turbina": 16.4252122,
        "absorvedor": 2.36279661,
        "gerador": 4.16686019,
        "condensador": 0.494578691,
        "evaporador": 0.322206188,
        "vs": 0.108074775,
        "vr": 0.033672699,
        "hx": 0.545410987,
        "bomba": 0.006807562,
        "retificador": 0.109811981,
        "rhx": 0.050646134,
        "umidificador": 8.23502758,
        "desumidificador": 10.029415,
        "aquecedor": 38.2098032
    }

    labels = ("Compressor", "Regenerador", "CC", "Turbina", "Absorvedor", "Gerador", "Condensador",
              "Evaporador", "VS", "VR", "SHX", "Bomba", "Retificador", "RHX", "Umidificador", "Desumidificador",
              "Aquecedor")
    exd_total = {}
    for libr_exd, nh3_exd, label in zip(exd_k_libr.values(), exd_k_nh3.values(), labels):
        exd_total.update({label: (libr_exd, nh3_exd)})

    exd_sorted = dict(sorted(exd_total.items(), key=lambda x: x[1][0]))

    width = 0.20
    dist = 0.25
    labels_sorted = list(exd_sorted.keys())
    libr_sorted = [exd[0] for exd in exd_sorted.values()]
    nh3_sorted = [exd[1] for exd in exd_sorted.values()]

    y_pos = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(19, 14))
    rects1 = ax.barh(y_pos - dist, libr_sorted, width, label=r'$LiBr/H_2O$')
    rects2 = ax.barh(y_pos + dist, nh3_sorted, width, label=r'$NH_3/H_2O$')
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=28)

    ax.set_xlim(None, 180)

    ax.set_yticks(y_pos, labels=labels_sorted)
    ax.set_xlabel(r'$\dot{Ex}_{d,k}$ (kW)')
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.8))
    axins = inset_axes(ax, width=8.8, height=6, loc=4, borderpad=2)
    axins.set_xlim(None, 3)
    axins.set_title("Zoom")

    zoomed_data = dict(itertools.islice(exd_sorted.items(), 9))
    z_labels_sorted = list(zoomed_data.keys())
    z_libr_sorted = [exd[0] for exd in zoomed_data.values()]
    z_nh3_sorted = [exd[1] for exd in zoomed_data.values()]

    z_y_pos = np.arange(len(z_labels_sorted))
    z_rects1 = axins.barh(z_y_pos - dist, z_libr_sorted, width, label=r'$LiBr/H_2O$')
    z_rects2 = axins.barh(z_y_pos + dist, z_nh3_sorted, width, label=r'$NH_3/H_2O$')
    for bars in axins.containers:
        axins.bar_label(bars, fmt="%.2f", padding=3, fontsize=28)
    axins.set_yticks(z_y_pos, labels=z_labels_sorted)
    ax.grid(False)
    axins.grid(False)
    filepath = os.path.join(destination_folder(), "exd_equipamentos_ptbr.pdf")
    plt.savefig(filepath, bbox_inches="tight")


def psi_graph():
    """Gráfico das eficiências exergéticas dos volumes de controle do sistema."""

    psi_k_libr = {
        "compressor": 89.4834085,
        "regenerador": 81.9937253,
        "cc": 77.2006026,
        "turbina": 93.9574472,
        "absorvedor": 16.2562929,
        "gerador": 34.946967,
        "condensador": 16.5331278,
        "evaporador": 57.7866817,
        "vs": 99.9936133,
        "vr": 82.2486342,
        "hx": 71.0114559,
        "bomba": 95.1620773,
        "retificador": 0,
        "rhx": 0,
        "umidificador": 76.9895279,
        "desumidificador": 63.5996644,
        "aquecedor": 39.3516872
    }

    psi_k_nh3 = {
        "compressor": 89.4834085,
        "regenerador": 81.9937253,
        "cc": 77.2006026,
        "turbina": 93.9574472,
        "absorvedor": 6.10080987,
        "gerador": 52.7482211,
        "condensador": 16.0695365,
        "evaporador": 60.7465191,
        "vs": 99.9874831,
        "vr": 99.9838625,
        "hx": 72.4284912,
        "bomba": 95.1422866,
        "retificador": 8.997,
        "rhx": 45.93,
        "umidificador": 76.989493,
        "desumidificador": 63.599674,
        "aquecedor": 39.6467978
    }

    labels = ("Compressor", "Regenerador", "CC", "Turbina", "Absorvedor", "Gerador", "Condensador",
              "Evaporador", "VS", "VR", "SHX", "Bomba", "Retificador", "RHX", "Umidificador", "Desumidificador",
              "Aquecedor")
    psi_total = {}
    for libr_psi, nh3_psi, label in zip(psi_k_libr.values(), psi_k_nh3.values(), labels):
        psi_total.update({label: (libr_psi, nh3_psi)})

    psi_sorted = dict(sorted(psi_total.items(), key=lambda x: x[1][1]))

    width = 0.20
    dist = 0.25
    labels_sorted = list(psi_sorted.keys())
    libr_sorted = [psi[0] for psi in psi_sorted.values()]
    nh3_sorted = [psi[1] for psi in psi_sorted.values()]

    y_pos = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(19, 14))
    rects1 = ax.barh(y_pos - dist, libr_sorted, width, label=r'$LiBr/H_2O$')
    rects2 = ax.barh(y_pos + dist, nh3_sorted, width, label=r'$NH_3/H_2O$')
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=28)
    ax.set_xlim(None, 110)

    ax.set_yticks(y_pos, labels=labels_sorted)
    ax.set_xlabel(r'$\psi_{k}$ (%)')
    ax.legend()

    ax.grid(False)
    filepath = os.path.join(destination_folder(), "psi_equipamentos_ptbr.pdf")
    plt.savefig(filepath, bbox_inches="tight")


def exd_graph_paper():
    """Gráfico das destruições de exergia dos volumes de controle do sistema em inglês e só para o sistema LiBr."""

    exd_k_libr = {
        "compressor": 15.8719014,
        "regenerador": 35.9994965,
        "cc": 163.205706,
        "turbina": 16.4252122,
        "absorvedor": 0.620634485,
        "gerador": 4.52427394,
        "condensador": 0.520269154,
        "evaporador": 0.364249391,
        "vs": 0.002141506,
        "vr": 0.04672878,
        "hx": 0.154546269,
        "bomba": 1.30749E-05,
        "umidificador": 8.41433196,
        "desumidificador": 10.2478122,
        "aquecedor": 39.5269246
    }

    labels = ("Compressor", "Regenerator", "CC", "Turbine", "Absorber", "Generator", "Condenser",
              "Evaporator", "SV", "RV", "SHX", "Pump", "Humidifier", "Dehumidifier", "Heater")

    exd_total = {}
    for exd, label in zip(exd_k_libr.values(), labels):
        exd_total.update({label: exd})

    exd_sorted = dict(sorted(exd_total.items(), key=lambda x: x[1]))

    width = 0.45
    labels_sorted = list(exd_sorted.keys())
    libr_sorted = [exd for exd in exd_sorted.values()]

    y_pos = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(16, 11))
    bars = ax.barh(y_pos, libr_sorted, width, color="#306bac")
    # for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=19)
    ax.set_xlim(None, 180)

    ax.set_yticks(y_pos, labels=labels_sorted)
    ax.set_xlabel(r'$\dot{Ex}_{d,k}$ (kW)')
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.8))
    axins = inset_axes(ax, width=6.8, height=4, loc=4, borderpad=2)
    axins.set_xlim(None, 10)
    axins.set_title("Zoom in")

    zoomed_data = dict(itertools.islice(exd_sorted.items(), 9))
    z_labels_sorted = list(zoomed_data.keys())
    z_libr_sorted = [exd for exd in zoomed_data.values()]

    z_y_pos = np.arange(len(z_labels_sorted))
    z_bars = axins.barh(z_y_pos, z_libr_sorted, width, color="#306bac")

    for bars in axins.containers:
        axins.bar_label(bars, fmt="%.2f", padding=3, fontsize=19)
    axins.set_yticks(z_y_pos, labels=z_labels_sorted)
    ax.grid(False)
    axins.grid(False)

    filepath = os.path.join(destination_folder(), "exd_equipamentos_enus.jpg")
    filepath_pdf = os.path.join(destination_folder(), "exd_equipamentos_enus.pdf")
    plt.savefig(filepath, bbox_inches="tight")
    plt.savefig(filepath_pdf, bbox_inches="tight")


if __name__ == "__main__":
    matploblib_config()
    psi_graph()
    exd_graph()
    exd_graph_paper()
