import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ees.parametric_graphs import Graphs


class GraphsSysEffDecomp(Graphs):

    def __init__(self, base_paths, variable):
        super().__init__(base_paths, variable)

    def get_titles(self, lang):
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "EUFdecomp": {
                    "trigeracao_LiBrH2O": r"Decomposição EUF - Trigeração (SRA: $ LiBr/H_2O $)",
                    "trigeracao_NH3H2O": r"Decomposição EUF - Trigeração (SRA: $ NH_3/H_2O $)",
                },
                "psidecomp": {
                    "trigeracao_LiBrH2O": r"Decomposição $ \psi_{sys} $ - Trigeração (SRA: $ LiBr/H_2O $)",
                    "trigeracao_NH3H2O": r"Decomposição $ \psi_{sys} $ - Trigeração (SRA: $ NH_3/H_2O $)",
                },
                "labels": {
                    "sys": "Sistema",
                    "hdh": "HDH",
                    "sra": "SRA",
                    "turbina": "Microturbinas",
                },
                "EUFeixo": "Componentes EUF",
                "PSIeixosys": "Sistema, Microturbinas",
                "PSIeixosra": "SRA, HDH",
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "EUFdecomp": {
                    "trigeracao_LiBrH2O": r"EUF Decomposition - Trigeneration (ARS: $ LiBr/H_2O)$",
                    "trigeracao_NH3H2O": r"EUF Decomposition - Trigeneration (ARS: $ NH_3/H_2O)$",
                },
                "psidecomp": {
                    "trigeracao_LiBrH2O": r"$ \psi_{sys} $ Decomposition - Trigeneration (ARS: $LiBr/H_2O$)",
                    "trigeracao_NH3H2O": r"$ \psi_{sys} $ Decomposition - Trigeneration (ARS: $NH_3/H_2O$)",
                },
                "labels": {
                    "sys": "System",
                    "hdh": "HDH",
                    "sra": "ARS",
                    "turbina": "Microturbines",
                },
                "EUFeixo": "EUF Components",
                "PSIeixosys": "System, Microturbines",
                "PSIeixosra": "ARS, HDH",
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

        # matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
        #     color=["004c6d", "4c7c9b", "86b0cc", "c1e7ff"],
        #     linestyle=["-", "--", ":", "-."]
        # )

        graph_letters = ["a)", "b)"]

        # EUF
        fig1, axs1 = plt.subplots(1, 2, figsize=(18.4, 7), num="DecompEUFSys")
        for ax, (model, df), letter in zip(axs1, self.dfs.items(), graph_letters):
            ax.set_title(titles["EUFdecomp"][model])
            ax.set_title(
                letter,
                fontfamily="serif",
                loc="left",
                style="italic",
                fontweight="normal",
            )
            ax.set_xlabel(var_display_str)
            ax.set_ylabel(titles["EUFeixo"])
            ax.plot(
                df[self.variable],
                df["EUF_sys_turbina"],
                label=titles["labels"]["turbina"],
            )
            ax.plot(df[self.variable], df["EUF_sys_sra"], label=titles["labels"]["sra"])
            ax.plot(df[self.variable], df["EUF_sys_hdh"], label=titles["labels"]["hdh"])
            ax.plot(df[self.variable], df["EUF_sys"], label=titles["labels"]["sys"])
            ax.legend(loc="center right")

        fig1.tight_layout()
        fig1.savefig(
            os.path.join(
                self.plots_folder, f"plot_{lang}_EUF-decomp_{self.variable}.pdf"
            )
        )
        fig1.clf()

        # Psi_sys
        fig2, axs2 = plt.subplots(1, 2, figsize=(18.4, 7), num="DecompPSISys")
        for ax, (model, df), letter in zip(axs2, self.dfs.items(), graph_letters):
            ax.set_title(titles["psidecomp"][model])
            ax.set_title(
                letter,
                fontfamily="serif",
                loc="left",
                style="italic",
                fontweight="normal",
            )
            ax.set_xlabel(var_display_str)
            ax.set_ylabel(titles["PSIeixosys"])
            (l1,) = ax.plot(
                df[self.variable],
                df["psi_sys_turbina"],
                color="#004c6d",
                linestyle="-",
                label=titles["labels"]["turbina"],
            )
            (l2,) = ax.plot(
                df[self.variable],
                df["psi_sys_1"],
                color="#00608a",
                linestyle="--",
                label=titles["labels"]["sys"],
            )
            ax2 = ax.twinx()
            ax2.set_ylabel(titles["PSIeixosra"])
            (l3,) = ax2.plot(
                df[self.variable],
                df["psi_sys_sra"],
                color="#0075a7",
                linestyle=":",
                label=titles["labels"]["sra"],
            )
            (l4,) = ax2.plot(
                df[self.variable],
                df["psi_sys_hdh"],
                color="#008ac6",
                linestyle="-.",
                label=titles["labels"]["hdh"],
            )
            ax2.grid(False)
            ax.legend(handles=[l1, l2, l3, l4], loc="upper right")

        fig2.tight_layout()
        fig2.savefig(
            os.path.join(
                self.plots_folder, f"plot_{lang}_psi-decomp_vs_{self.variable}.pdf"
            ),
            # bbox_inches='tight'
        )

        fig2.clf()
