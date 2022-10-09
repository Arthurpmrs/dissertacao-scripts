import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from ees.parametric_graphs import Graphs
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class GraphsDefault(Graphs):

    def __init__(self, base_paths, variable):
        super().__init__(base_paths, variable)

    def get_titles(self, lang):
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "m_dot[38]": r"Vazão mássica de água dessalinizada ($ \dot{m}_{38} $)",
                "GOR": "Gained Output Ratio (GOR)",
                "eta_brayton": r"Eficiência Energética do ciclo de Brayton ($ \eta_{brayton} $)",
                "W_net": r"Trabalho Líquido da Turbina ($ \dot{W}_{net} $)",
                "psi_sys_1": r"Eficiência Exergética do Sistema de Trigeração ($ \psi_{sys} $)",
                "Exd_sys": r"Exergia Destruída do Sistema de Trigeração ($ \dot{Ex}_{d, sys} $)",
                "EUF_sys": r"Fator de Utilização de Energia ($EUF$)",
                "COP_1": "Coeficiente de Desempenho (COP)",
                "models": {
                    "trigeracao_LiBrH2O": r"Trigeração (SRA: $LiBr/H_2O$)",
                    "trigeracao_NH3H2O": r"Trigeração (SRA: $NH_3/H_2O$)",
                },
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "m_dot[38]": r"Freshwater mass flow rate ($ \dot{m}_{38} $)",
                "GOR": "Gained Output Ratio (GOR)",
                "eta_brayton": r"Brayton's cycle Energetic Efficiency ($ \eta_{Brayton} $)",
                "W_net": r"Net Turbine Work ($ \dot{W}_{net} $)",
                "psi_sys_1": r"Trigeneration's Exergetic Efficiency ($ \psi_{sys} $)",
                "Exd_sys": r"Trigenerations's Exergy Destruction ($ \dot{Ex}_{d, sys} $)",
                "EUF_sys": "Energy Utilization Factor (EUF)",
                "COP_1": "Coefficient of Performance (COP)",
                "models": {
                    "trigeracao_LiBrH2O": r"Trigeneration (ARS: $LiBr/H_2O$)",
                    "trigeracao_NH3H2O": r"Trigeneration (ARS: $NH_3/H_2O$)",
                },
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def generate(self, var_display_str, lang="pt-BR"):

        titles = self.get_titles(lang)
        # Dessalinização
        fig1, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(18.4, 7), num="Dessalinização")
        ax1.set_title(titles["m_dot[38]"])
        ax1.set_xlabel(var_display_str)
        ax1.set_ylabel(r"$\dot{m}_{38} \: (kg \: s^{-1})$")
        ax2.set_title(titles["GOR"])
        ax2.set_xlabel(var_display_str)
        ax2.set_ylabel("GOR")
        ax1.set_title("a)", fontfamily="serif", loc="left",
                      style="italic", fontweight="normal")
        ax2.set_title("b)", fontfamily="serif", loc="left",
                      style="italic", fontweight="normal")

        for model, df in self.dfs.items():
            ax1.plot(df[self.variable], df["m_dot[38]"],
                     label=titles["models"][model])
            ax2.plot(df[self.variable], df["GOR"],
                     label=titles["models"][model])

        # Axis Limits
        ax1.set_ylim(
            0.95 * min([df["m_dot[38]"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["m_dot[38]"].max() for _, df in self.dfs.items()]),
        )
        ax2.set_ylim(
            0.95 * min([df["GOR"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["GOR"].max() for _, df in self.dfs.items()]),
        )
        ax1.legend()
        ax2.legend()

        fig1.tight_layout()
        fig1.savefig(
            os.path.join(self.plots_folder,
                         f"plot_{lang}_GOR_m_dot[38]_vs_{self.variable}.pdf")
        )
        # Turbina a gás
        fig3, (ax3, ax4) = plt.subplots(
            1, 2, figsize=(18.4, 7), num="Turbina a gás")
        ax3.set_title(titles["W_net"])
        ax3.set_xlabel(var_display_str)
        ax3.set_ylabel(r"$ \dot{W}_{net} $ (kW)")

        ax4.set_title(titles["eta_brayton"])
        ax4.set_xlabel(var_display_str)
        ax4.set_ylabel(r" $ \eta_{brayton} $ (%)")
        ax3.set_title(
            "a)", fontfamily="serif", loc="left", style="italic", fontweight="normal"
        )
        ax4.set_title(
            "b)", fontfamily="serif", loc="left", style="italic", fontweight="normal"
        )

        for model, df in self.dfs.items():
            ax3.plot(df[self.variable], df["W_net"],
                     label=titles["models"][model])
            ax4.plot(
                df[self.variable], df["eta_brayton"], label=titles["models"][model]
            )

        # Axis Limits
        ax3.set_ylim(
            0.95 * min([df["W_net"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["W_net"].max() for _, df in self.dfs.items()]),
        )
        ax4.set_ylim(
            0.95 * min([df["eta_brayton"].min()
                        for _, df in self.dfs.items()]),
            1.05 * max([df["eta_brayton"].max()
                        for _, df in self.dfs.items()]),
        )
        ax3.legend()
        ax4.legend()

        fig3.tight_layout()
        fig3.savefig(
            os.path.join(
                self.plots_folder,
                f"plot_{lang}_eta_brayton_W_net_vs_{self.variable}.pdf",
            ),
        )

        # Exergia
        fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(18.4, 7), num="Exergia")
        ax5.set_title(titles["psi_sys_1"])
        ax5.set_xlabel(var_display_str)
        ax5.set_ylabel(r"$ \psi_{sys} $ (%)")

        ax6.set_title(titles["Exd_sys"])
        ax6.set_xlabel(var_display_str)
        ax6.set_ylabel(r" $ \dot{Ex}_{d,sys} $ (kW)")
        ax5.set_title(
            "a)", fontfamily="serif", loc="left", style="italic", fontweight="normal"
        )
        ax6.set_title(
            "b)", fontfamily="serif", loc="left", style="italic", fontweight="normal"
        )

        for model, df in self.dfs.items():
            ax5.plot(df[self.variable], df["psi_sys_1"],
                     label=titles["models"][model])
            ax6.plot(df[self.variable], df["Exd_sys"],
                     label=titles["models"][model])

        # Axis Limits
        ax5.set_ylim(
            0.95 * min([df["psi_sys_1"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["psi_sys_1"].max() for _, df in self.dfs.items()]),
        )
        ax6.set_ylim(
            0.95 * min([df["Exd_sys"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["Exd_sys"].max() for _, df in self.dfs.items()]),
        )
        ax5.legend()
        ax6.legend()

        fig5.tight_layout()
        fig5.savefig(
            os.path.join(
                self.plots_folder, f"plot_{lang}_exergy_vs_{self.variable}.pdf"
            ),
        )

        # Eficiência energética (EUF)
        fig7, ax7 = plt.subplots(num="Eficiência Energética", figsize=(9.2, 7))
        ax7.set_title(titles["EUF_sys"])
        ax7.set_xlabel(var_display_str)
        ax7.set_ylabel(r"$ EUF $")
        for model, df in self.dfs.items():
            ax7.plot(df[self.variable], df["EUF_sys"],
                     label=titles["models"][model])
        ax7.legend()
        ax7.set_ylim(
            0.95 * min([df["EUF_sys"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["EUF_sys"].max() for _, df in self.dfs.items()]),
        )
        fig7.tight_layout()
        fig7.savefig(
            os.path.join(self.plots_folder,
                         f"plot_{lang}_EUF_vs_{self.variable}.pdf"),
        )

        # COP
        fig8, ax8 = plt.subplots(num="COP", figsize=(9.2, 7))
        ax8.set_title(titles["COP_1"])
        ax8.set_xlabel(var_display_str)
        ax8.set_ylabel(r"COP")
        for model, df in self.dfs.items():
            ax8.plot(df[self.variable], df["COP_1"],
                     label=titles["models"][model])
        ax8.legend()
        ax8.set_ylim(
            0.95 * min([df["COP_1"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["COP_1"].max() for _, df in self.dfs.items()]),
        )
        plt.legend()
        fig8.tight_layout()
        fig8.savefig(
            os.path.join(self.plots_folder,
                         f"plot_{lang}_COP_vs_{self.variable}.pdf"),
        )

        figs = [fig1, fig3, fig5, fig7, fig8]
        for fig in figs:
            fig.clf()
