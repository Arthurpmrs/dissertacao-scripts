import os
import sys
import math
import plotly.io as pio
from numpy import i0
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utilities import get_base_folder
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import plotly.graph_objects as go
import random

pio.kaleido.scope.mathjax = None


class ModelSankey:

    def __init__(self, EES_model: str, run_id: str):
        self.base_path = get_base_folder(EES_model)
        self.run_id = run_id
        self.paths = self.set_paths()

    def set_paths(self):
        paths = {
            'data': os.path.join(self.base_path, '.solver', self.run_id)
        }

        for path in paths.values():
            if not os.path.exists(path):
                os.makedirs(path)

        return paths

    def load_data(self) -> list:
        filename_arrays = os.path.join(self.paths.get("data"), "clean_arrays.csv")
        df_arrays = pd.read_csv(filename_arrays, delimiter=";", decimal=",")
        exergy = df_arrays["ex"]
        exergy.index = list(range(1, 39))

        filename_outputs = os.path.join(self.paths.get("data"), "outputs.xlsx")
        outputs = pd.read_excel(filename_outputs)
        outputs = outputs.set_index("Unnamed: 0")

        data = [
            ["Combustível", "Brayton", exergy[9]],
            ["Brayton", "Produtos", outputs.loc["W_net", 0]],
            ["Brayton", "Destruição", outputs.loc["Exd_brayton", 0]],
            ["Brayton", "Exaustão Brayton", exergy[6]],
            ["Exaustão Brayton", "SRA", exergy[6]],
            ["SRA", "Exaustão SRA", exergy[7]],
            ["SRA", "Produtos", exergy[31] - exergy[30]],
            ["SRA", "Destruição", outputs.loc["Exd_sra", 0]],
            ["Exaustão SRA", "HDH", exergy[7]],
            ["HDH", "Exaustão HDH", exergy[8]],
            ["HDH", "Produtos", exergy[38]],
            ["HDH", "Destruição", outputs.loc["Exd_hdh", 0]]
        ]
        labels = ["Combustível", "Brayton", "SRA", "HDH", "Produtos", "Destruição", "Exaustão Brayton",
                  "Exaustão SRA", "Exaustão HDH"]
        for i, label in enumerate(labels):
            for j, t in enumerate(data):
                for k, item in enumerate(t):
                    if item == label:
                        data[j][k] = i
        return data

    def prepare_componentes(self, lang: str = "pt") -> dict:
        data = self.load_data()

        source = [item[0] for item in data]
        target = [item[1] for item in data]
        value = [item[2] for item in data]

        if lang.lower() == "pt":
            labels = ["Combustível", "Brayton", "SRA", "HDH", "Produtos", "Destruição", "Exaustão Brayton",
                      "Exaustão SRA", "Exaustão"]
            pos = [(0.10, 0.7), (0.65, 0.05), (0.65, 0.7), (6, 6), (0.5, 0.35), (6, 6), (0.85, 0.02),
                   (0.8, 0.43), (0.81, 0.20), (1.0, 0.186), (0.94, 0.10), (0.95, 0.33)]
            annotations = [f"<b>{v:.2f} kW</b>".replace(".", ",") for v in value]
        elif lang.lower() == "en":
            labels = ["Fuel", "Brayton", "ARS", "HDH", "Products", "Losses", "Brayton Outlet",
                      "ARS Outlet", "Outlet"]
            pos = [(0.12, 0.7), (0.65, 0.05), (0.65, 0.7), (6, 6), (0.5, 0.35), (6, 6), (0.85, 0.02),
                   (0.8, 0.43), (0.81, 0.20), (1.0, 0.186), (0.94, 0.10), (0.95, 0.33)]
            annotations = [f"<b>{v:.2f} kW</b>" for v in value]
        else:
            raise ValueError("Lang should be pt or en.")

        node_colors = ["#141B41" for i in range(9)]
        link_colors = ["rgba(48, 107, 172, 0.5)", "rgba(111, 156, 235, 0.5)", "rgba(152, 185, 242, 0.5)", "rgba(145, 142, 244, 0.5)",
                       "rgba(255, 215, 0, 1)", "rgba(255,0,0, 1)", "rgba(111, 156, 235, 0.5)", "rgba(152, 185, 242, 0.5)", "rgba(145, 142, 244, 0.5)"]
        return (labels, pos, source, target, value, node_colors, link_colors, annotations)

    def generate(self, lang: str = "pt"):
        labels, pos, source, target, value, node_colors, link_colors, annotations = self.prepare_componentes(lang)

        labels = [f"<b>{label}</b>" for label in labels]

        fig = go.Figure(data=[go.Sankey(
            valueformat=".2f",
            valuesuffix=" kW",
            arrangement="snap",
            node=dict(
                pad=5,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors,
                x=[0.05, 0.25, 0.58, 0.82, 1, 1, 0.42, 0.68, 1],
                y=[0.5, 0.5, 0.7, 0.7, 1, 0.3, 0.6, 0.7, 0.1],
            ),
            link=dict(
                source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=target,
                value=value,
                color=[link_colors[i] for i in source],
                label=[f"{v:.2f} kW" for v in value]
            ))])

        for p, a in zip(pos, annotations):
            fig.add_annotation(x=p[0], y=p[1],
                               text=a,
                               showarrow=False,
                               font=dict(
                size=28,
            )
            )

        fig.update_layout(font_size=24, width=1550, height=650)
        final_path = os.path.join(self.paths["data"], f"sankey_diagram_{lang}.pdf")
        fig.write_image(final_path)
        final_path = os.path.join(self.paths["data"], f"sankey_diagram_{lang}.jpg")
        fig.write_image(final_path)
        fig.show()


def main(path):
    run_id = "CASO BASE"

    sankey = ModelSankey(path, run_id)
    sankey.generate(lang="en")
    sankey.generate(lang="pt")


if __name__ == "__main__":
    EES_model_libr = r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_LiBrH2O.EES"
    main(EES_model_libr)
    EES_model_nh3 = r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_NH3H2O.EES"
    main(EES_model_nh3)
