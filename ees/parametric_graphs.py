import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class Graphs:
    def __init__(self, base_paths, variable):
        self.variable = variable
        self.base_paths = base_paths
        self.dfs = self.get_df()
        self.plots_folder = self.set_plots_folder()
        self.set_matplotlib_globalconfig()

    def get_df(self):
        dfs = {}
        for base_path in self.base_paths:
            filepath = os.path.join(
                base_path, ".results", self.variable, "parametric_result.csv"
            )

            dfs.update({os.path.basename(base_path): pd.read_csv(filepath, sep=";")})
        return dfs

    def set_plots_folder(self):
        models_folder = os.path.dirname(self.base_paths[0])
        # plots_folder = os.path.join(models_folder, "both", ".plots", self.variable)
        plots_folder = os.path.join(models_folder, "both", ".plots", "new_color", self.variable)

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

        legends = {"fontsize": 14}
        matplotlib.rc("legend", **legends)

        savefig = {"dpi": 300}
        matplotlib.rc("savefig", **savefig)

        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
            color=["004c6d", "7aaac6"],
            linestyle=["-", "--"]
        )
        matplotlib.rcParams["ytick.labelsize"] = 15
        matplotlib.rcParams["xtick.labelsize"] = 15
