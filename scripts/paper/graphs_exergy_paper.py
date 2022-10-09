import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import itertools


def matploblib_config():
    plt.style.use("ggplot")

    font_dir = [r"C:\Root\Download\computer-modern"]
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

    # matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    #     color=["r", "b", "g", "m", "k"]
    # )
    matplotlib.rcParams["ytick.labelsize"] = 15
    matplotlib.rcParams["xtick.labelsize"] = 15


def exd_graph():
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

    # exd_k_nh3 = {
    #     "compressor": 15.8719014,
    #     "regenerador": 35.9994965,
    #     "cc": 163.205706,
    #     "turbina": 16.4252122,
    #     "absorvedor": 2.36279661,
    #     "gerador": 4.16686019,
    #     "condensador": 0.494578691,
    #     "evaporador": 0.322206188,
    #     "vs": 0.108074775,
    #     "vr": 0.033672699,
    #     "hx": 0.545410987,
    #     "bomba": 0.006807562,
    #     "retificador": 0.109811981,
    #     "rhx": 0.050646134,
    #     "umidificador": 8.23502758,
    #     "desumidificador": 10.029415,
    #     "aquecedor": 38.2098032
    # }

    labels = ("Compressor", "Regenerator", "Comb. Chamber", "Turbine", "Absorber", "Generator", "Condenser",
              "Evaporator", "SV", "RV", "SHX", "Pump", "Humidifier", "Dehumidifier", "Heater")
    exd_total = {}
    for exd, label in zip(exd_k_libr.values(), labels):
        exd_total.update({label: exd})

    exd_sorted = dict(sorted(exd_total.items(), key=lambda x: x[1]))

    width = 0.35
    labels_sorted = list(exd_sorted.keys())
    libr_sorted = [exd for exd in exd_sorted.values()]

    y_pos = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(y_pos, libr_sorted, width, color="#306bac")
    # bars = ax.barh(y_pos, libr_sorted, color="#306bac")
    # rects1 = ax.barh(y_pos - width / 2, libr_sorted, width, label=r'$LiBr/H_2O$')
    # rects2 = ax.barh(y_pos + width / 2, nh3_sorted, width, label=r'$NH_3/H_2O$')
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", padding=3)

    # values = [exd for exd in exd_k_libr.values()]
    # y_pos = np.arange(len(label))

    # ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos, labels=labels_sorted)
    ax.set_xlabel(r'$\dot{Ex}_{d,k}$ (kW)')
    ax.set_title('Exergy destruction at kth equipment')
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.8))
    axins = inset_axes(ax, width=6.4, height=4, loc=4, borderpad=2)
    # axins.tick_params(labelleft=False, labelbottom=False)

    zoomed_data = dict(itertools.islice(exd_sorted.items(), 9))
    z_labels_sorted = list(zoomed_data.keys())
    z_libr_sorted = [exd for exd in zoomed_data.values()]
    # z_nh3_sorted = [exd[1] for exd in zoomed_data.values()]

    z_y_pos = np.arange(len(z_labels_sorted))
    z_bars = axins.barh(z_y_pos, z_libr_sorted, width, color="#306bac")
    # z_bars = axins.barh(z_y_pos, z_libr_sorted, color="#306bac")
    # z_rects1 = axins.barh(z_y_pos - width / 2, z_libr_sorted, width, label=r'$LiBr/H_2O$')
    # z_rects2 = axins.barh(z_y_pos + width / 2, z_nh3_sorted, width, label=r'$NH_3/H_2O$')
    for bars in axins.containers:
        axins.bar_label(bars, fmt="%.3f", padding=3)
    axins.set_yticks(z_y_pos, labels=z_labels_sorted)
    ax.grid(False)
    axins.grid(False)
    path = r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\05 - Imagens e diagramas\resultados"
    # filename = "exd_equipamentos_enus.pdf"
    filename = "exd_equipamentos_enus.jpg"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    # plt.show()


def psi_graph():
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
        "umidificador": 76.9895279,
        "desumidificador": 63.5996644,
        "aquecedor": 39.3516872
    }

    # psi_k_nh3 = {
    #     "compressor": 89.4834085,
    #     "regenerador": 81.9937253,
    #     "cc": 77.2006026,
    #     "turbina": 93.9574472,
    #     "absorvedor": 6.10080987,
    #     "gerador": 52.7482211,
    #     "condensador": 16.0695365,
    #     "evaporador": 60.7465191,
    #     "vs": 99.9874831,
    #     "vr": 99.9838625,
    #     "hx": 72.4284912,
    #     "bomba": 95.1422866,
    #     "retificador": 8.997,
    #     "rhx": 45.93,
    #     "umidificador": 76.989493,
    #     "desumidificador": 63.599674,
    #     "aquecedor": 39.6467978
    # }

    labels = ("Compressor", "Regenerator", "Comb. Chamber", "Turbine", "Absorber", "Generator", "Condenser",
              "Evaporator", "SV", "RV", "SHX", "Pump", "Humidifier", "Dehumidifier", "Heater")

    psi_total = {}
    for psi, label in zip(psi_k_libr.values(), labels):
        psi_total.update({label: psi})

    psi_sorted = dict(sorted(psi_total.items(), key=lambda x: x[1]))

    width = 0.35
    labels_sorted = list(psi_sorted.keys())
    libr_sorted = [psi for psi in psi_sorted.values()]
    # nh3_sorted = [psi[1] for psi in psi_sorted.values()]

    y_pos = np.arange(len(labels_sorted))

    fig, ax = plt.subplots(figsize=(14, 8))
    # rects1 = ax.barh(y_pos - width / 2, libr_sorted, width, label=r'$LiBr/H_2O$')
    # rects2 = ax.barh(y_pos + width / 2, nh3_sorted, width, label=r'$NH_3/H_2O$')
    bars = ax.barh(y_pos, libr_sorted, width, color="#306bac")
    # bars = ax.barh(y_pos, libr_sorted, color="#306bac")

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=3)

    # values = [psi for psi in psi_k_libr.values()]
    # y_pos = np.arange(len(label))

    # ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos, labels=labels_sorted)
    ax.set_xlabel(r'$\psi_{k}$ (%)')
    ax.set_title('Exergetic efficiency of kth equipment')
    ax.legend()

    ax.grid(False)
    path = r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\05 - Imagens e diagramas\resultados"
    # filename = "psi_equipamentos_enus.pdf"
    filename = "psi_equipamentos_enus.jpg"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath)


if __name__ == "__main__":
    matploblib_config()
    exd_graph()
    psi_graph()
