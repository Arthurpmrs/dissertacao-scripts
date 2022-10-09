import os
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class OptGraph:

    def __init__(self, base_path: str, idx: str = None):
        self.base_path = base_path
        if idx:
            self.idx = idx
        else:
            self.idx = self.last_generated_idx()
        self.plots_folder = self.set_plots_folder()
        self.results = self.load_results(self.idx)
        self.set_matplotlib_globalconfig()

    def set_plots_folder(self) -> str:
        plots_folder = os.path.join(self.base_path, ".opt", self.idx, ".plots")

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        return plots_folder

    def last_generated_idx(self) -> int:
        ids_folder = os.path.join(self.base_path, ".opt")
        idxs = []
        for dir in os.listdir(ids_folder):
            filepath = os.path.join(ids_folder, dir)
            if os.path.isdir(filepath):
                idxs.append(dir)
        last_generated_idx = sorted(idxs, reverse=True)[0]
        return last_generated_idx

    def load_results(self, idx: int) -> dict:
        filename = os.path.join(self.base_path, ".opt", idx, ".results", f"results.json")
        with open(filename, "r") as jsonfile:
            results = json.load(jsonfile)
        return results

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
            color=["r", "b", "g", "m", "k"]
        )
        matplotlib.rcParams["ytick.labelsize"] = 15
        matplotlib.rcParams["xtick.labelsize"] = 15

    def get_titles(self, lang: str) -> dict:
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "fitness": "Histórico de Aptidão",
                "error": "Histórico de Erros",
                "error-label": "Erro",
                "xlabel": "Gerações"
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "fitness": "Fitness History",
                "error": "Error History",
                "error-label": "Error",
                "xlabel": "Generations"
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def generate(self, target_display: str, lang: str = "pt-BR"):
        target_name = list(self.results["best_target"].keys())[0]
        target_history = pd.DataFrame(self.results["gen_history"])
        titles = self.get_titles(lang)

        fig, ax = plt.subplots(num="fitness", figsize=(9.2, 7))
        ax.set_title(titles["fitness"])
        ax.set_xlabel(titles["xlabel"])
        ax.set_ylabel(target_display)
        ax.plot(target_history.loc[:, "best_target"], marker="o")
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.plots_folder, f"plot_{lang}_fitness-history_{target_name}.svg"),
        )
        fig.clf()
        fig2, ax2 = plt.subplots(num="error", figsize=(9.2, 7))
        ax2.set_title(titles["error"])
        ax2.set_xlabel(titles["xlabel"])
        ax2.set_ylabel(titles["error-label"])
        ax2.plot(target_history.loc[:, "error"], marker="o")
        fig2.tight_layout()
        fig2.savefig(
            os.path.join(self.plots_folder, f"plot_{lang}_error-history_{target_name}.svg"),
        )
        fig2.clf()
