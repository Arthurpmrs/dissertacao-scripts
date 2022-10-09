import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from config import font_location


def matplotlib_config():
    plt.style.use("seaborn-paper")

    font_dir = [font_location]
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

    matplotlib.rcParams["ytick.labelsize"] = 15
    matplotlib.rcParams["xtick.labelsize"] = 15


def consumo_de_agua():
    data = pd.read_csv(r"scripts\introduction\global-freshwater-use-over-the-long-run.csv")
    filt = data["Entity"] == "World"
    pd.set_option('display.max_rows', 500)
    world_data = data.loc[filt]
    world_data["Freshwater use"] = world_data["Freshwater use"].apply(lambda x: int(x) / 1e9)

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.fill_between(world_data.loc[:, "Year"], world_data.loc[:, "Freshwater use"], alpha=0.6)
    ticks = np.linspace(world_data.loc[330, "Year"], world_data.loc[440, "Year"], 8)
    ticks = [round(year) for year in ticks]
    ax.set_xticks(ticks)
    ax.set_ylabel(r"Consumo de água $\mathrm{km}^3 \cdot \mathrm{ano}^{-1}$")
    ax.set_xlabel("Anos")
    ax.grid()

    fig.savefig(r"models\graphs\consumo_de_agua.pdf")
    plt.show()


def perdas_energeticas():
    plt.rcParams['figure.figsize'] = (13.0, 8)
    data = {
        "centrais": ("Centrais elétricas", 55.7),
        "transmissao": ("Perdas de transmissão e distribuição de eletricidade", 27.6),
        "carvoarias": ("Carvoarias", 10.7),
        "refinarias": ("Refinarias e plantas de gás natural", 2.5),
        "coquerias": ("Coquerias", 0.5),
        "nuclear": ("Ciclo combustível nuclear", 0.2),
        "destilarias": ("Destilarias", 0.1),
        "outras": ("Outras transformações", 0.8),
        "outras2": ("Outras perdas de distribuição e armazenagem", 1.8)
    }
    df = pd.DataFrame({"names": [label for key, (label, value) in data.items()],
                       "values": [value for key, (label, value) in data.items()]})

    colors = ["#004c6d", "#125e7f", "#217192", "#3085a5", "#3e99b7", "#4dadc9", "#5dc2dc", "#6dd7ed", "#7eedff"]

    p1 = df.plot.bar(x="names", y="values", color=colors, width=1.0)
    plt.xlabel("2020")
    cmap = dict(zip(df.names, colors))

    # create the rectangles for the legend
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]

    # add the legend
    plt.legend(handles=patches, loc='upper right', borderaxespad=1)
    plt.tick_params(
        # axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)  # labels along the bottom edge are off

    x = [label for key, (label, value) in data.items()]
    y_num = [value for key, (label, value) in data.items()]
    y = [f"{value}%" for key, (label, value) in data.items()]
    for i in range(len(x)):
        plt.text(i, y_num[i] + 0.5, y[i], ha='center', fontsize="16")

    plt.savefig("models/graphs/perdas.pdf", bbox_inches="tight", transparent=True)
    plt.show()


def consumo_energetico():
    sizes = [26.8, 30.9, 23.2, 5.0, 2.5, 9.4, 2.2]
    labels = ["Carvão", "Petróleo", "Gás natural", "Nuclear", "Hidrelétrica", "Biocombustíveis e resíduos", "Outros"]
    colors = ["#004c6d", "#176486", "#2b7e9e", "#3e99b7", "#52b4cf", "#67d0e7", "#7eedff"]
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    labels_values = [f"{label} ({value}%)" for label, value in zip(labels, sizes)]
    wedges, texts = ax1.pie(sizes, colors=colors, labels=labels_values, wedgeprops=dict(width=0.5),
                            textprops={'fontsize': 20})
    # wedges, texts = ax1.pie(sizes, colors=colors, wedgeprops=dict(width=0.5))

    # plt.legend(patches, labels_values, loc="best")
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # kw = dict(arrowprops=dict(arrowstyle="-", color="k"), va="center")
    # for p, label in zip(wedges, labels_values):
    #     ang = np.deg2rad((p.theta1 + p.theta2) / 2)
    #     y = np.sin(ang)
    #     x = np.cos(ang)
    #     horizontalalignment = "center" if abs(x) < abs(y) else "right" if x < 0 else "left"
    #     ax1.annotate(label, xy=(1.05 * x, 1.05 * y), xytext=(1.2 * x, 1.2 * y),
    #                  horizontalalignment=horizontalalignment, **kw, fontsize=15)
    plt.tight_layout()
    plt.savefig("models/graphs/consumo_energetico.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    matplotlib_config()
    consumo_de_agua()
    perdas_energeticas()
    consumo_energetico()
