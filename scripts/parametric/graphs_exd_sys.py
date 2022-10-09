import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ees.parametric_graphs import Graphs


class GraphsSysExd(Graphs):

    def __init__(self, base_paths, variable):
        super().__init__(base_paths, variable)

    def get_titles(self, lang):
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "exds": {
                    "trigeracao_LiBrH2O": r"$ \dot{Ex}_{d,i} $ - Trigeração (SRA: $ LiBr/H_2O)$",
                    "trigeracao_NH3H2O": r"$ \dot{Ex}_{d,i} $ - Trigeração (SRA: $ NH_3/H_2O)$",
                },
                "labels": {
                    "comp": "Compressor",
                    "cc": "C. Combustão",
                    "regen": "Regenerador",
                    "turb": "Turbina",
                    "gen": "Gerador",
                    "cond": "Condensador",
                    "hx": "SHX",
                    "bomba": "Bomba",
                    "vs": "Válvula da Sol.",
                    "abs": "Absorvedor",
                    "evap": "Evaporador",
                    "vr": "Válvula do Refri.",
                    "aq": "Aquecedor",
                    "umid": "Umidificador",
                    "desu": "Desumidificador",
                    "reti": "Retificador",
                    "rhx": "RHX",
                },
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "exds": {
                    "trigeracao_LiBrH2O": r"$ \dot{Ex}_{d,i} $ - Trigeneration (ARS: $ LiBr/H_2O)$",
                    "trigeracao_NH3H2O": r"$ \dot{Ex}_{d,i} $ - Trigeneration (ARS: $ NH_3/H_2O)$",
                },
                "labels": {
                    "comp": "Compressor",
                    "cc": "Comb. Chamber",
                    "regen": "Regenerator",
                    "turb": "Turbine",
                    "gen": "Generator",
                    "cond": "Condenser",
                    "hx": "SHX",
                    "bomba": "Pump",
                    "vs": "Solution Valve",
                    "abs": "Absorber",
                    "evap": "Evaporator",
                    "vr": "Refrig. Valve",
                    "aq": "Heater",
                    "umid": "Humidifier",
                    "desu": "Dehumidifier",
                    "reti": "Rectifier",
                    "rhx": "RHX",
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
        legends = {"fontsize": 12}
        matplotlib.rc("legend", **legends)
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
            color=["004c6d", "005b82", "006a98", "007aaf", "008ac6"]
        )

        graph_letters = ["a)", "b)"]
        fig, axs = plt.subplots(1, 2, figsize=(18.4, 7), num="DecompostionExd")
        lines = ["-", "--", "-.", ":"]
        for ax, (model, df), letter in zip(axs, self.dfs.items(), graph_letters):
            linecycler = cycle(lines)
            ax.set_title(titles["exds"][model])
            ax.set_title(
                letter,
                fontfamily="serif",
                loc="left",
                style="italic",
                fontweight="normal",
            )
            ax.set_xlabel(var_display_str)
            ax.set_ylabel(r"$\dot{Ex}_{d,k}$")
            ax.plot(
                df[self.variable],
                df["Exd_compressor"],
                label=titles["labels"]["comp"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_cc"],
                label=titles["labels"]["cc"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_regenerador"],
                label=titles["labels"]["regen"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_turbina"],
                label=titles["labels"]["turb"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_gerador"],
                label=titles["labels"]["gen"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_condensador"],
                label=titles["labels"]["cond"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_hx"],
                label=titles["labels"]["hx"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_bomba"],
                label=titles["labels"]["bomba"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_vs"],
                label=titles["labels"]["vs"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_absorvedor"],
                label=titles["labels"]["abs"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_evaporador"],
                label=titles["labels"]["evap"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_vr"],
                label=titles["labels"]["vr"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_aquecedor"],
                label=titles["labels"]["aq"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_umidificador"],
                label=titles["labels"]["umid"],
                linestyle=next(linecycler),
            )
            ax.plot(
                df[self.variable],
                df["Exd_desumidificador"],
                label=titles["labels"]["desu"],
                linestyle=next(linecycler),
            )
            if "Exd_rhx" in df.columns:
                ax.plot(
                    df[self.variable],
                    df["Exd_rhx"],
                    label=titles["labels"]["rhx"],
                    linestyle=next(linecycler),
                )
            if "Exd_retificador" in df.columns:
                ax.plot(
                    df[self.variable],
                    df["Exd_retificador"],
                    label=titles["labels"]["reti"],
                    linestyle=next(linecycler),
                )

            ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self.plots_folder, f"plot_{lang}_Decomp-EXD-sys_{self.variable}.pdf"
            )
        )
        fig.clf()
