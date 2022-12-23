import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scripts.config import default_output_folder


def destination_folder():
    """Configurar a pasta de destino dos gráficos."""

    folder = os.path.join(default_output_folder, "graphs")

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def matplotlib_config():
    """Configurações de estilo do matplotlib. Executar esse método antes da geração dos gráficos."""

    plt.style.use("ggplot")

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
        color=["r", "b", "g", "m", "k"]
    )
    matplotlib.rcParams["ytick.labelsize"] = 15
    matplotlib.rcParams["xtick.labelsize"] = 15


def generate_piechart():
    """Gera o gráfico referente aos custos dos subsistemas do sistema de trigeração."""

    data = {
        "cost_2019_turbine": 328508.08,
        "cost_2019_gerador": 297.45,
        "cost_2019_absorvedor": 2245.51,
        "cost_2019_condensador": 456.37,
        "cost_2019_evaporador": 1603.98,
        "cost_2019_hx": 637.36,
        "cost_2019_bomba": 48.21,
        "cost_2019_vs": 7.00,
        "cost_2019_vr": 0.61,
        "cost_2019_sra": 5296.50,
        "cost_2019_u": 1944.49,
        "cost_2019_d": 1023.42,
        "cost_2019_aquecedor": 1936.68,
        "cost_2019_fan": 377.74,
        "cost_2019_hdh": 5282.33,
        "cost_CAPEX_2019_trigen": 339086.91,
    }

    labels = ["Microturbinas", "SRA", "HDH"]

    sizes = [
        data["cost_2019_turbine"] / data["cost_CAPEX_2019_trigen"] * 100,
        data["cost_2019_sra"] / data["cost_CAPEX_2019_trigen"] * 100,
        data["cost_2019_hdh"] / data["cost_CAPEX_2019_trigen"] * 100
    ]

    for i, (label, size) in enumerate(zip(labels, sizes)):
        labels[i] = f"{label} ({size:.3f}%)"

    explode = (0.1, 0, 0)
    fig1, ax1 = plt.subplots()
    colors = ["#004c6d", "#638fb0", "#b3d9f8"]
    patches, _ = ax1.pie(sizes, colors=colors, startangle=120, explode=explode)
    plt.legend(patches, labels, loc="best")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()

    fig1.savefig(os.path.join(destination_folder(), "piechart_trigeracao.pdf"))


if __name__ == "__main__":
    matplotlib_config()
    generate_piechart()
