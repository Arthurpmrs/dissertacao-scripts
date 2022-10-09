import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

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

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    color=["r", "b", "g", "m", "k"]
)
matplotlib.rcParams["ytick.labelsize"] = 15
matplotlib.rcParams["xtick.labelsize"] = 15

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

labels_1 = ["Microturbinas", "Gerador", "Absorvedor", "Condensador", "Evaporador", "SHX", "Bomba", "VS", "VR",
            "Umidificador", "Desumidificador", "Aquecedor", "Ventilador"]

labels_2 = ["Microturbinas", "SRA", "HDH"]

sizes_1 = [
    data["cost_2019_turbine"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_gerador"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_absorvedor"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_condensador"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_evaporador"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_hx"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_bomba"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_vs"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_vr"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_u"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_d"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_aquecedor"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_fan"] / data["cost_CAPEX_2019_trigen"] * 100
]

sizes_2 = [
    data["cost_2019_turbine"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_sra"] / data["cost_CAPEX_2019_trigen"] * 100,
    data["cost_2019_hdh"] / data["cost_CAPEX_2019_trigen"] * 100
]

for i, (label, size) in enumerate(zip(labels_2, sizes_2)):
    labels_2[i] = f"{label} ({size:.3f}%)"

explode_1 = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
explode_2 = (0.1, 0, 0)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.4, 7))
fig1, ax1 = plt.subplots()
# ax1.pie(sizes_1, explode=explode_1, labels=labels_1, autopct='%1.1f%%',
#         shadow=False, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# ax1.pie(sizes_2, explode=explode_2, labels=labels_2,
#         shadow=False, startangle=90)
# colors = ['royalblue', 'red', 'lime']
# colors = ["tab:blue", "tab:red", "tab:green"]
colors = ["#004c6d", "#638fb0", "#b3d9f8"]
# colors = ["#003f5c", "#bc5090", "#ffa600"]
patches, texts = ax1.pie(sizes_2, colors=colors, startangle=120, explode=explode_2)
plt.legend(patches, labels_2, loc="best")
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()

fig1.savefig(
    r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\03 - Draft e Revisões\Defesa\dissertacao-tex\Cap_8_analise-economica\Figuras\piechart_trigeracao.pdf"
)
