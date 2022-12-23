import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


class GraphsPaper:

    def __init__(self, model_path, variable, run_id):
        self.variable = variable
        self.run_id = run_id
        self.base_path = os.path.dirname(model_path)
        self.model_name = ".".join(os.path.basename(model_path).split(".")[:-1])
        self.df = self.get_df()
        self.plots_folder = self.set_plots_folder()
        self.set_matplotlib_globalconfig()

    def get_df(self):
        filepath = os.path.join(
            self.base_path, "results", self.model_name, ".ParamAnalysis",
            self.run_id, ".results", self.variable, "parametric_result.csv"
        )
        return pd.read_csv(filepath, sep=";")

    def set_plots_folder(self):
        plots_folder = os.path.join(self.base_path, "results", self.model_name,
                                    ".ParamAnalysis", self.run_id, ".plots", "paper")

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        return plots_folder

    def set_matplotlib_globalconfig(self):
        plt.style.use("seaborn-paper")

        font_dir = [r"font\computer-modern"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)

        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["font.family"] = "CMU Serif"

        axes = {
            "labelsize": 24,
            "titlesize": 16,
            "titleweight": "bold",
            "labelweight": "bold",
        }
        matplotlib.rc("axes", **axes)

        lines = {"linewidth": 2}
        matplotlib.rc("lines", **lines)

        legends = {"fontsize": 20}
        matplotlib.rc("legend", **legends)

        savefig = {"dpi": 300}
        matplotlib.rc("savefig", **savefig)

        matplotlib.rcParams["ytick.labelsize"] = 20
        matplotlib.rcParams["xtick.labelsize"] = 20
        matplotlib.rcParams["axes.grid"] = False

    def generate_plot(self, var_display_str):
        """Gerar plots da análise paramétrica para o artigo."""

        fig, ax = plt.subplots(figsize=(13, 7))
        fig.subplots_adjust(right=0.75, left=0.25)

        twin1 = ax.twinx()
        twin2 = ax.twinx()
        twin3 = ax.twinx()

        twin2.spines["right"].set_color("grey")
        twin3.spines["left"].set_color("grey")
        # Necessário para colocar o ylabel e os ticks do lado esquerdo.
        twin3.yaxis.set_label_position("left")
        twin3.yaxis.tick_left()

        # Offset the right spine of twin2.  The ticks and label have already been
        # placed on the right by twinx above.
        twin2.spines.right.set_position(("axes", 1.2))
        twin3.spines.left.set_position(("axes", -0.2))

        colors = ["#0085cc", "#008702", "#d45800", "#8d00b0"]
        lss = ["solid", "dotted", "dashed", "dashdot"]
        self.df["psi_partial"] = self.df["psi_partial"].apply(lambda x: 100 * x)
        p1, = ax.plot(self.df[self.variable], self.df["EUF_sys"], colors[0], label=r"$EUF$", ls=lss[0])
        p2, = twin1.plot(self.df[self.variable], self.df["Exd_sys"], colors[2], label=r"$\dot{Ex}_{d,sys}$", ls=lss[1])
        p3, = twin2.plot(self.df[self.variable], self.df["psi_sys_1"], colors[3], label=r"$\psi_{sys}$", ls=lss[2])
        p4, = twin3.plot(self.df[self.variable], self.df["m_dot[38]"], colors[1], label=r"$\dot{m}_{38}$", ls=lss[3])

        offset = 0.01
        ax.set_ylim(
            (1 - offset) * min(list(self.df["EUF_sys"])),
            (1 + offset) * max(list(self.df["EUF_sys"])),
        )
        twin1.set_ylim(
            (1 - offset - 0.02) * min(list(self.df["Exd_sys"])),
            (1 + offset + 0.02) * max(list(self.df["Exd_sys"])),
        )
        twin2.set_ylim(
            (1 - offset - 0.02) * min(list(self.df["psi_sys_1"])),
            (1 + offset + 0.02) * max(list(self.df["psi_sys_1"])),
        )
        twin3.set_ylim(
            (1 - offset) * min(list(self.df["m_dot[38]"])),
            (1 + offset) * max(list(self.df["m_dot[38]"])),
        )

        ax.set_xlabel(var_display_str)
        ax.set_ylabel(r"$EUF$")
        twin1.set_ylabel(r"$\dot{Ex}_{d,sys}$ (kW)")
        twin2.set_ylabel(r"$\psi_{sys}$ (%)")
        twin3.set_ylabel(r"$\dot{m}_{38} (kg \cdot s^{-1}$)")

        ax.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())
        twin3.yaxis.label.set_color(p4.get_color())

        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        ax.tick_params(axis='x', **tkw)

        ax.legend(handles=[p1, p2, p3, p4], loc="upper center")

        ax.grid()

        plt.savefig(os.path.join(self.plots_folder, f"plot_paper_{self.variable}.pdf"))
        plt.savefig(os.path.join(self.plots_folder, f"plot_paper_{self.variable}.jpg"))
        del fig


def main():
    model_path = r"C:\Root\Universidade\Mestrado\dissertacao-scripts\models\trigeracao_LiBrH2O.EES"
    run_id = "param_analysis_v1"

    graph = GraphsPaper(model_path, "X_biogas_ch4", run_id)
    graph.generate_plot(r'$ x_{CH_4} $')

    graph = GraphsPaper(model_path, 'T[22]', run_id)
    graph.generate_plot(r'$ T_{22} $ ($^{\circ}$C)')

    graph = GraphsPaper(model_path, 'T[19]', run_id)
    graph.generate_plot(r'$ T_{19} $ ($^{\circ}$C)')

    graph = GraphsPaper(model_path, 'T[10]', run_id)
    graph.generate_plot(r'$ T_{10} $ ($^{\circ}$C)')

    graph = GraphsPaper(model_path, 'T[13]', run_id)
    graph.generate_plot(r'$ T_{13} $ ($^{\circ}$C)')

    graph = GraphsPaper(model_path, "MR", run_id)
    graph.generate_plot("MR")

    graph = GraphsPaper(model_path, 'T[34]', run_id)
    graph.df.drop(index=6, inplace=True)
    graph.generate_plot(r'$ T_{34} $ ($^{\circ}$C)')


if __name__ == "__main__":
    main()
