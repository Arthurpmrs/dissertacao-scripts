import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class GraphsEffectivenessHDH:

    def __init__(self, base_paths, lang="pt-BR"):
        self.base_paths = base_paths
        self.lang = lang
        self.dfs = self.get_df()
        self.plots_folder = self.set_plots_folder()
        self.set_matplotlib_globalconfig()

    def get_df(self):
        dfs = {}
        for base_path in self.base_paths:
            results_path = os.path.join(base_path, ".results", "epsilon_d")
            results = {}
            for dirname in os.listdir(results_path):
                if os.path.isdir(os.path.join(results_path, dirname)):
                    filepath = os.path.join(results_path, dirname, "parametric_result.csv")
                    epsilon_u = dirname.split(" ")[-1]
                    if self.lang in ["pt-BR", "pt_BR", "ptbr"]:
                        epsilon_u = epsilon_u.replace(".", ",")
                    df = (
                        pd.read_csv(filepath, sep=";")
                        .drop(index=2)
                        .drop(index=25)
                        .drop(index=26)
                        .drop(index=27)
                        .drop(index=35)
                        .drop(index=36)
                    )
                    results.update({epsilon_u: df})

            dfs.update({os.path.basename(base_path): results})
        return dfs

    def set_plots_folder(self):
        models_folder = os.path.dirname(self.base_paths[0])
        plots_folder = os.path.join(models_folder, "both", ".plots", "new_color", "epsilon_d")

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        return plots_folder

    def set_matplotlib_globalconfig(self):
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

        legends = {"fontsize": 12}
        matplotlib.rc("legend", **legends)

        savefig = {"dpi": 300}
        matplotlib.rc("savefig", **savefig)

        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
            color=["004c6d", "005b82", "006a98", "007aaf", "008ac6"]
        )
        matplotlib.rcParams["ytick.labelsize"] = 15
        matplotlib.rcParams["xtick.labelsize"] = 15

    def get_titles(self):
        if self.lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "m_dot[38]": r"Vazão mássica de água pura ($ \dot{m}_{38} $)",
                "GOR": "Gained Output Ratio (GOR)",
                "eta_brayton": r"Eficiência Energética do ciclo de Brayton ($ \eta_{Brayton} $)",
                "W_net": r"Trabalho Líquido da Turbina ($ \dot{W}_{net} $)",
                "psi_sys_1": r"Eficiência Exergética do Sistema de Trigeração ($ \psi_{sys} $)",
                "Exd_sys": r"Exergia Destruída do Sistema de Trigeração ($ \dot{Ex}_{d, sys} $)",
                "EUF_sys": "Fator de Utilização de Energia (EUF)",
                "COP_1": "Coeficiente de Desempenho (COP)",
                "labelvar": r"$ \varepsilon_{u} = $",
                "models": {
                    "trigeracao_LiBrH2O": r"Tri($LiBr/H_2O$)",
                    "trigeracao_NH3H2O": r"Tri($NH_3/H_2O$)",
                },
            }
        elif self.lang in ["en-US", "en_US", "enus"]:
            titles = {
                "m_dot[38]": r"Freshwater mass flow rate ($ \dot{m}_{38} $)",
                "GOR": "Gained Output Ratio (GOR)",
                "eta_brayton": r"Brayton's cycle Energetic Efficiency ($ \eta_{Brayton} $)",
                "W_net": r"Net Turbine Work ($ \dot{W}_{net} $)",
                "psi_sys_1": r"Trigeneration's Exergetic Efficiency ($ \psi_{sys} $)",
                "Exd_sys": r"Trigenerations's Exergy Destruction ($ \dot{Ex}_{d, sys} $)",
                "EUF_sys": "Energy Utilization Factor (EUF)",
                "COP_1": "Coefficient of Performance (COP)",
                "labelvar": r"$ \varepsilon_{h} = $",
                "models": {
                    "trigeracao_LiBrH2O": r"Tri($LiBr/H_2O$)",
                    "trigeracao_NH3H2O": r"Tri($NH_3/H_2O$)",
                },
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def generate(self):
        titles = self.get_titles()
        var_display_str = r"$ \varepsilon_{d} $"
        lines = ["-", "--"]

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.4, 7), num="Dessalinização")
        ax1.set_title(titles["m_dot[38]"])
        ax1.set_xlabel(var_display_str)
        ax1.set_ylabel(r"$\dot{m}_{38} \: (kg \: s^{-1}$")
        ax2.set_title(titles["GOR"])
        ax2.set_xlabel(var_display_str)
        ax2.set_ylabel("GOR")
        ax1.set_title("a)", fontfamily="serif", loc="left", style="italic", fontweight="normal")
        ax2.set_title("b)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        for (model, results), line in zip(self.dfs.items(), lines):
            for epsilon_h, df in results.items():
                ax1.plot(
                    df["epsilon_d"],
                    df["m_dot[38]"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)
                ax2.plot(
                    df["epsilon_d"],
                    df["GOR"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)

        ax1.legend()
        ax2.legend()
        fig1.tight_layout()
        fig1.savefig(
            os.path.join(self.plots_folder, f"plot_{self.lang}_GOR_m_dot[38]_vs_epsilon_d.pdf"),
        )

        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(18.4, 7), num="Turbina a gás")
        ax3.set_title(titles["W_net"])
        ax3.set_xlabel(var_display_str)
        ax3.set_ylabel(r"$ \dot{W}_{net} $ (kW)")
        ax4.set_title(titles["eta_brayton"])
        ax4.set_xlabel(var_display_str)
        ax4.set_ylabel(r" $ \eta_{brayton} $ (%)")
        ax3.set_title("a)", fontfamily="serif", loc="left", style="italic", fontweight="normal")
        ax4.set_title("b)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        for (model, results), line in zip(self.dfs.items(), lines):
            for epsilon_h, df in results.items():
                ax3.plot(
                    df["epsilon_d"],
                    df["W_net"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)
                ax4.plot(
                    df["epsilon_d"],
                    df["eta_brayton"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)

        ax3.legend()
        ax4.legend()
        fig3.tight_layout()
        fig3.savefig(
            os.path.join(self.plots_folder, f"plot_{self.lang}_eta_brayton_W_net_vs_epsilon_d.pdf"),
        )

        fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(18.4, 7), num="Exergia")
        ax5.set_title(titles["psi_sys_1"])
        ax5.set_xlabel(var_display_str)
        ax5.set_ylabel(r"$ \psi_{sys} $ (%)")
        ax6.set_title(titles["Exd_sys"])
        ax6.set_xlabel(var_display_str)
        ax6.set_ylabel(r" $ \dot{Ex}_{d,sys} $ (kW)")
        ax5.set_title("a)", fontfamily="serif", loc="left", style="italic", fontweight="normal")
        ax6.set_title("b)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        for (model, results), line in zip(self.dfs.items(), lines):
            for epsilon_h, df in results.items():
                ax5.plot(
                    df["epsilon_d"],
                    df["psi_sys_1"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)
                ax6.plot(
                    df["epsilon_d"],
                    df["Exd_sys"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)
        ax5.legend()
        ax6.legend()
        fig5.tight_layout()
        fig5.savefig(
            os.path.join(self.plots_folder, f"plot_{self.lang}_exergy_vs_epsilon_d.pdf"),
        )

        fig7, ax7 = plt.subplots(num="Eficiência Energética", figsize=(9.2, 7))
        ax7.set_title(titles["EUF_sys"])
        ax7.set_xlabel(var_display_str)
        ax7.set_ylabel(r"$ EUF $")
        for (model, results), line in zip(self.dfs.items(), lines):
            for epsilon_h, df in results.items():
                ax7.plot(
                    df["epsilon_d"],
                    df["EUF_sys"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)

        ax7.legend()
        fig7.tight_layout()
        fig7.savefig(
            os.path.join(self.plots_folder, f"plot_{self.lang}_EUF_vs_epsilon_d.pdf"),
        )

        fig8, ax8 = plt.subplots(num="COP", figsize=(9.2, 7))
        ax8.set_title(titles["COP_1"])
        ax8.set_xlabel(var_display_str)
        ax8.set_ylabel(r"COP")
        for (model, results), line in zip(self.dfs.items(), lines):
            for epsilon_h, df in results.items():
                ax8.plot(
                    df["epsilon_d"],
                    df["COP_1"],
                    linestyle=line,
                    label=f'{titles["models"][model]}: ' + titles["labelvar"] + epsilon_h)

        ax8.legend()
        fig8.tight_layout()
        fig8.savefig(
            os.path.join(self.plots_folder, f"plot_{self.lang}_COP_vs_epsilon_d.pdf"),
        )

        figs = [fig1, fig3, fig5, fig7, fig8]
        for fig in figs:
            fig.clf()
