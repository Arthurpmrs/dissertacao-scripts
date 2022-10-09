import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ees.parametric_graphs import Graphs


class GraphsHDHExd(Graphs):

    def __init__(self, base_paths, variable):
        super().__init__(base_paths, variable)

    def get_titles(self, lang):
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "exd": {
                    "trigeracao_LiBrH2O": r"$ \dot{Ex}_{d} $ HDH - Trigeração (SRA: $ LiBr/H_2O $)",
                    "trigeracao_NH3H2O": r"$ \dot{Ex}_{d} $ HDH - Trigeração (SRA: $ NH_3/H_2O $)",
                },
                "labels": {
                    "sys": "HDH",
                    "aq": "Aquecedor",
                    "desu": "Desumidificador",
                    "umid": "Umidificador",
                },
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "exd": {
                    "trigeracao_LiBrH2O": r"$ \dot{Ex}_{d} $ HDH - Trigeneration (ARS: $ LiBr/H_2O)$",
                    "trigeracao_NH3H2O": r"$ \dot{Ex}_{d} $ HDH - Trigeneration (ARS: $ NH_3/H_2O)$",
                },
                "labels": {
                    "sys": "HDH",
                    "aq": "Heater",
                    "desu": "Dehumidifier",
                    "umid": "Humidifier",
                },
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def generate(self, var_display_str, lang="pt-BR"):
        titles = self.get_titles(lang)

        axes = {
            "labelsize": 20,
            "titlesize": 18 if lang in ["pt-BR", "pt_BR", "ptbr"] else 16,
            "titleweight": "bold",
            "labelweight": "bold",
        }
        matplotlib.rc("axes", **axes)

        graph_letters = ["a)", "b)"]

        fig, axs = plt.subplots(1, 2, figsize=(18.4, 7), num="DecompEUFSys")
        for ax, (model, df), letter in zip(axs, self.dfs.items(), graph_letters):
            ax.set_title(titles["exd"][model])
            ax.set_title(
                letter,
                fontfamily="serif",
                loc="left",
                style="italic",
                fontweight="normal",
            )
            ax.set_xlabel(var_display_str)
            ax.set_ylabel(r"$\dot{Ex}_{d,i}$")
            ax.plot(
                df[self.variable],
                df["Exd_aquecedor"],
                color="#004c6d",
                linestyle="-",
                label=titles["labels"]["aq"],
            )
            ax.plot(
                df[self.variable],
                df["Exd_umidificador"],
                color="#00608a",
                linestyle="--",
                label=titles["labels"]["umid"],
            )
            ax.plot(
                df[self.variable],
                df["Exd_desumidificador"],
                color="#0075a7",
                linestyle=":",
                label=titles["labels"]["desu"],
            )
            ax.plot(
                df[self.variable],
                df["Exd_hdh"],
                color="#008ac6",
                linestyle="-.",
                label=titles["labels"]["sys"],
            )
            ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(self.plots_folder, f"plot_{lang}_EXD-HDH_{self.variable}.pdf"),
        )
        fig.clf()
