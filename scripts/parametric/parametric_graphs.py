import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class DefaultGraphs:

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
                                    ".ParamAnalysis", self.run_id, ".plots", self.variable)

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

    def get_plotable_data(self, partial):
        euf = {"data": self.df["EUF_sys"], "legend": r"$EUF$", "label": r"$EUF$"}
        m38 = {"data": self.df["m_dot[38]"], "legend": r"$\dot{m}_{38}$", "label": r"$\dot{m}_{38} (\mathrm{kg} \cdot \mathrm{s}^{-1}$)"}
        if partial:
            exd = {"data": self.df["Exd_partial"], "legend": r"$\dot{Ex}_{d,p}$", "label": r"$\dot{Ex}_{d,p}$ (kW)"}
            psi = {"data": self.df["psi_partial"].apply(lambda x: 100 * x), "legend": r"$\psi_{p}$", "label": r"$\psi_{p}$ (%)"}
        else:
            exd = {"data": self.df["Exd_sys"], "legend": r"$\dot{Ex}_{d,sys}$", "label": r"$\dot{Ex}_{d,sys}$ (kW)"}
            psi = {"data": self.df["psi_sys_1"], "legend": r"$\psi_{sys}$", "label": r"$\psi_{sys}$ (%)"}

        return euf, m38, exd, psi

    def base_plot(self, var_display_str, partial=True, loc="best"):
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

        # colors = ["#004c6d", "#0057a3", "#005bd7", "#0051ff"]
        # colors = ["#003f5c", "#7a5195", "#ef5675", "#ffa600"]
        # colors = ["#004c6d", "#34546a", "#4e5b67", "#636363"]
        # colors = ["#004c6d", "#4c3f81", "#93065b", "#a10000"]
        # colors = ["#004c6d", "#718799", "#c17360", "#a10000"]
        colors = ["#0085cc", "#008702", "#d45800", "#8d00b0"]
        lss = ["solid", "dotted", "dashed", "dashdot"]
        # colors = ["b", "r", "g", "m"]

        euf, m38, exd, psi = self.get_plotable_data(partial=partial)

        p1, = ax.plot(self.df[self.variable], euf["data"], colors[0], label=euf["legend"], ls=lss[0])
        p2, = twin1.plot(self.df[self.variable], exd["data"], colors[2], label=exd["legend"], ls=lss[1])
        p3, = twin2.plot(self.df[self.variable], psi["data"], colors[3], label=psi["legend"], ls=lss[2])
        p4, = twin3.plot(self.df[self.variable], m38["data"], colors[1], label=m38["legend"], ls=lss[3])

        offset = 0.01
        ax.set_ylim(
            (1 - offset) * min(list(euf["data"])),
            (1 + offset) * max(list(euf["data"])),
        )
        twin1.set_ylim(
            (1 - offset - 0.02) * min(list(exd["data"])),
            (1 + offset + 0.02) * max(list(exd["data"])),
        )
        twin2.set_ylim(
            (1 - offset - 0.02) * min(list(psi["data"])),
            (1 + offset + 0.02) * max(list(psi["data"])),
        )
        twin3.set_ylim(
            (1 - offset) * min(list(m38["data"])),
            (1 + offset) * max(list(m38["data"])),
        )

        ax.set_xlabel(var_display_str)
        ax.set_ylabel(euf["label"])
        twin1.set_ylabel(exd["label"])
        twin2.set_ylabel(psi["label"])
        twin3.set_ylabel(m38["label"])

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

        ax.legend(handles=[p1, p2, p3, p4], loc=loc)

        ax.grid()

        plt.tight_layout(w_pad=0.5)
        plt.savefig(os.path.join(self.plots_folder, f"plot_{self.variable}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(self.plots_folder, f"plot_{self.variable}.jpg"), bbox_inches='tight')
        del fig


class HDHEffGraphs:

    def __init__(self, model_paths, run_id):
        self.run_id = run_id
        self.model_paths = model_paths
        self.dfs = self.get_df()
        self.plots_folder = self.set_plots_folder()
        self.set_matplotlib_globalconfig()

    def get_df(self):
        dfs = {}
        e_u_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        folder_names = [f"hdh_e_u_{e_u}" for e_u in e_u_values]
        for model_path in self.model_paths:
            base_path = os.path.dirname(model_path)
            model_name = ".".join(os.path.basename(model_path).split(".")[:-1])
            # results_path = os.path.join(base_path, ".results", "epsilon_d")
            results = {}

            for i, folder_name in enumerate(folder_names):
                data_filepath = os.path.join(base_path, "results", model_name,
                                             ".ParamAnalysis", folder_name, ".results",
                                             "epsilon_d", "parametric_result.csv")
                if os.path.isfile(data_filepath):
                    results.update({e_u_values[i]: pd.read_csv(data_filepath, sep=";")})

            dfs.update({model_name: results})
        return dfs

    def set_plots_folder(self):
        base_path = os.path.dirname(self.model_paths[0])
        plots_folder = os.path.join(base_path, "results", "combined_graph")
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        return plots_folder

    def set_matplotlib_globalconfig(self):
        plt.style.use("seaborn-paper")

        font_dir = [r"font/computer-modern"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)

        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["font.family"] = "CMU Serif"

        axes = {
            "labelsize": 30,
            "titlesize": 26,
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

        matplotlib.rcParams["ytick.labelsize"] = 26
        matplotlib.rcParams["xtick.labelsize"] = 26
        matplotlib.rcParams["axes.grid"] = True

    def get_plotable_data(self, partial):
        euf = {"data": self.df["EUF_sys"], "legend": r"$EUF$", "label": r"$EUF$"}
        m38 = {"data": self.df["m_dot[38]"], "legend": r"$\dot{m}_{38}$", "label": r"$\dot{m}_{38} (\mathrm{kg} \cdot \mathrm{s}^{-1}$)"}
        if partial:
            exd = {"data": self.df["Exd_partial"], "legend": r"$\dot{Ex}_{d,p}$", "label": r"$\dot{Ex}_{d,p}$ (kW)"}
            psi = {"data": self.df["psi_partial"].apply(lambda x: 100 * x), "legend": r"$\psi_{p}$", "label": r"$\psi_{p}$ (%)"}
        else:
            exd = {"data": self.df["Exd_sys"], "legend": r"$\dot{Ex}_{d,sys}$", "label": r"$\dot{Ex}_{d,sys}$ (kW)"}
            psi = {"data": self.df["psi_sys_1"], "legend": r"$\psi_{sys}$", "label": r"$\psi_{sys}$ (%)"}

        return euf, m38, exd, psi

    def base_plot(self):
        var_display_str = r"$ \varepsilon_{d} $"

        fig, ((ax_euf, ax_m38), (ax_exd, ax_psi)) = plt.subplots(2, 2, figsize=(18, 12))
        # ax_euf.set_title("Fator de utilização de Energia (EUF)")
        ax_euf.set_xlabel(var_display_str)
        ax_euf.set_ylabel(r"EUF")
        ax_euf.set_title("a)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        # ax_m38.set_title(r"Vazão de água dessalinizada ($\dot{m}_{38}$)")
        ax_m38.set_xlabel(var_display_str)
        ax_m38.set_ylabel(r"$\dot{m}_{38}$ ($\mathrm{kg} \cdot \mathrm{s}^{-1}$)")
        ax_m38.set_title("b)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        # ax_exd.set_title(r"Taxa de destruição de exergia parcial ($\dot{Ex}_{d,p}$)")
        ax_exd.set_xlabel(var_display_str)
        ax_exd.set_ylabel(r"$\dot{Ex}_{d,p}$ (kW)")
        ax_exd.set_title("c)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        # ax_psi.set_title(r"Eficiência exergética parcial ($\psi_{p}$)")
        ax_psi.set_xlabel(var_display_str)
        ax_psi.set_ylabel(r"$\psi_{p}$ (%)")
        ax_psi.set_title("d)", fontfamily="serif", loc="left", style="italic", fontweight="normal")

        lines = ["-", "--"]
        legend_labels = {
            "trigeracao_LiBrH2O": r"Tri($LiBr/H_2O$): $ \varepsilon_{u} $ = ",
            "trigeracao_NH3H2O": r"Tri($NH_3/H_2O$): $ \varepsilon_{u} $ = "
        }
        colors = ["#0085cc", "#008702", "#d45800", "#8d00b0", "#eb7cb7"]
        for (model, results), line in zip(self.dfs.items(), lines):
            for (epsilon_u, df), color in zip(results.items(), colors):
                ax_euf.plot(
                    df["epsilon_d"],
                    df["EUF_sys"],
                    linestyle=line,
                    color=color,
                    label=f"{legend_labels[model]}{epsilon_u}")

                ax_m38.plot(
                    df["epsilon_d"],
                    df["m_dot[38]"],
                    linestyle=line,
                    color=color,
                    label=f"{legend_labels[model]}{epsilon_u}")

                ax_exd.plot(
                    df["epsilon_d"],
                    df["Exd_partial"],
                    linestyle=line,
                    color=color,
                    label=f"{legend_labels[model]}{epsilon_u}")

                ax_psi.plot(
                    df["epsilon_d"],
                    df["psi_partial"],
                    linestyle=line,
                    color=color,
                    label=f"{legend_labels[model]}{epsilon_u}")

        lines, labels = ax_euf.get_legend_handles_labels()
        fig.legend(lines, labels, bbox_to_anchor=(0, 0, 1, 0), loc='lower left', ncol=4, mode="expand")

        fig.tight_layout()

        fig.subplots_adjust(bottom=0.198)
        fig.savefig(
            os.path.join(self.plots_folder, "epsilon_d.pdf"), bbox_inches='tight'
        )


def main():
    model_paths = [
        r"C:\Root\Universidade\Mestrado\dissertacao-scripts\models\trigeracao_LiBrH2O.EES",
        r"C:\Root\Universidade\Mestrado\dissertacao-scripts\models\trigeracao_NH3H2O.EES"
    ]
    variable_legend_locs = [
        ["upper center", "best", "best", "best", "best", "best", "best", "lower right", "lower center"],
        ["upper center", "best", "center left", "center right", "best", "best", "best", "lower right", "lower center"]
    ]
    run_id = "param_analysis_v1"

    for model_path, locs in zip(model_paths, variable_legend_locs):
        graph = DefaultGraphs(model_path, "X_biogas_ch4", run_id)
        graph.base_plot(r'$ x_{CH_4} $', partial=False, loc=locs[0])
        del graph

        graph = DefaultGraphs(model_path, "m_dot[9]", run_id)
        graph.base_plot(r'$ \dot{m}_{9} $ ($\mathrm{kg} \cdot \mathrm{s}^{-1}$)', partial=False, loc=locs[1])
        del graph

        graph = DefaultGraphs(model_path, 'T[10]', run_id)
        graph.base_plot(r'$ T_{10} $ ($^{\circ}$C)', loc=locs[2])
        del graph

        graph = DefaultGraphs(model_path, 'T[13]', run_id)
        graph.base_plot(r'$ T_{13} $ ($^{\circ}$C)', loc=locs[3])
        del graph

        graph = DefaultGraphs(model_path, 'T[19]', run_id)
        graph.base_plot(r'$ T_{19} $ ($^{\circ}$C)', loc=locs[4])
        del graph

        graph = DefaultGraphs(model_path, 'T[22]', run_id)
        graph.base_plot(r'$ T_{22} $ ($^{\circ}$C)', loc=locs[5])
        del graph

        graph = DefaultGraphs(model_path, "MR", run_id)
        graph.base_plot("MR", loc=locs[7])
        del graph

        graph = DefaultGraphs(model_path, 'T[34]', run_id)
        graph.df.drop(index=6, inplace=True)
        graph.base_plot(r'$ T_{34} $ ($^{\circ}$C)', loc=locs[8])
        del graph

    graph = HDHEffGraphs(model_paths, run_id)
    graph.base_plot()
    del graph


if __name__ == "__main__":
    main()
