import sys
import os
import csv
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
from config import default_output_folder


def destination_folder():
    """Configurar a pasta de destino dos gráficos."""

    folder = os.path.join(default_output_folder, "graphs", "introduction")

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def matplotlib_config():
    """Configurações de estilo do matplotlib. Executar esse método antes da geração dos gráficos."""

    plt.style.use("seaborn-paper")

    font_dir = [r"font/computer-modern"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.family"] = "CMU Serif"

    axes = {
        "labelsize": 26,
        "titlesize": 18,
        "titleweight": "bold",
        "labelweight": "bold",
    }
    matplotlib.rc("axes", **axes)

    lines = {"linewidth": 2}
    matplotlib.rc("lines", **lines)

    legends = {"fontsize": 18}
    matplotlib.rc("legend", **legends)

    savefig = {"dpi": 300}
    matplotlib.rc("savefig", **savefig)

    matplotlib.rcParams["ytick.labelsize"] = 20
    matplotlib.rcParams["xtick.labelsize"] = 20


def consumo_de_agua_apos_2014():
    """Complemento dos dados do consumo de água. Banco de dados FAO - AQUASTAT."""

    organized_data = {}
    with open(r'scripts\introduction\region_sheet_data.csv', newline='') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=';')
        for row in spamreader:
            year = int(row["Year"])
            if year not in organized_data:
                organized_data.update({
                    year: 0
                })
            if row["Variables"] == "Total water withdrawal":
                organized_data[year] += float(row["Value"].replace(",", "."))

    pd.options.mode.chained_assignment = None
    df = pd.DataFrame.from_dict(organized_data, orient="index", columns=["Freshwater use"])
    filtered_df = df.loc[2019:2015]
    filtered_df["Year"] = filtered_df.index
    return filtered_df


def consumo_de_agua():
    """Gráfico do consumo de água. Banco de dados FAO - AQUASTAT."""

    data = pd.read_csv(r"scripts\introduction\global-freshwater-use-over-the-long-run.csv")
    filt = data["Entity"] == "World"
    pd.set_option('display.max_rows', 500)
    world_data = data.loc[filt]
    filtered_data = world_data.loc[:, ["Year", "Freshwater use"]]
    filtered_data["Freshwater use"] = filtered_data["Freshwater use"].apply(lambda x: int(x) / 1e9)

    new_data = consumo_de_agua_apos_2014()
    filtered_data = filtered_data.append(new_data.sort_values("Year"))

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.fill_between(filtered_data["Year"], filtered_data["Freshwater use"], alpha=0.7)
    ticks = np.linspace(1901, 2019, 10)
    ticks = [round(year) for year in ticks]
    ax.set_xticks(ticks)
    ax.set_ylabel(r"Consumo de água $\mathrm{km}^3 \cdot \mathrm{ano}^{-1}$")
    ax.set_xlabel("Anos")
    ax.grid()

    fig.savefig(os.path.join(destination_folder(), "consumo_de_agua.pdf"))


def perdas_energeticas():
    """Gráfico das perdas de energia no Brasil. Banco de dados BEN."""

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
    df = pd.DataFrame({"names": [label for _, (label, _) in data.items()],
                       "values": [value for _, (_, value) in data.items()]})

    colors = ["#004c6d", "#125e7f", "#217192", "#3085a5", "#3e99b7", "#4dadc9", "#5dc2dc", "#6dd7ed", "#7eedff"]

    _ = df.plot.bar(x="names", y="values", color=colors, width=1.0)
    plt.xlabel("2020")
    cmap = dict(zip(df.names, colors))

    # create the rectangles for the legend
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]

    # add the legend
    plt.legend(handles=patches, loc='upper right', borderaxespad=1)
    plt.tick_params(
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False
    )

    x = [label for _, (label, _) in data.items()]
    y_num = [value for _, (_, value) in data.items()]
    y = [f"{value}%" for _, (_, value) in data.items()]

    for i in range(len(x)):
        plt.text(i, y_num[i] + 0.5, y[i], ha='center', fontsize="24")

    plt.savefig(
        os.path.join(destination_folder(), "perdas.pdf"),
        bbox_inches="tight", transparent=True
    )


def consumo_energetico():
    """Gráfico do consumo energético mundial. Banco de dados da IEA."""

    sizes = [26.8, 30.9, 23.2, 5.0, 2.5, 9.4, 2.2]
    labels = ["Carvão", "Petróleo", "Gás natural", "Nuclear", "Hidrelétrica", "Biocombustíveis e resíduos", "Outros"]
    colors = ["#004c6d", "#176486", "#2b7e9e", "#3e99b7", "#52b4cf", "#67d0e7", "#7eedff"]
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    labels_values = [f"{label} ({value}%)" for label, value in zip(labels, sizes)]
    wedges, texts = ax1.pie(sizes, colors=colors, labels=labels_values, wedgeprops=dict(width=0.5),
                            textprops={'fontsize': 28})

    plt.savefig(
        os.path.join(destination_folder(), "consumo_energetico.pdf"),
        bbox_inches="tight"
    )


def main():
    """Executar as funções de criação de gráficos."""

    matplotlib_config()
    consumo_de_agua()
    perdas_energeticas()
    consumo_energetico()


if __name__ == "__main__":
    main()
