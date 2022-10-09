import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from ees.parametric_graphs import Graphs
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


class DollarAnalisys():

    def __init__(self, EES_models):
        self.paths = self.set_paths(EES_models)
        self.dfs = self.get_df()
        self.set_matplotlib_globalconfig()

    def set_paths(self, EES_models):
        paths = {}
        for model_name, EES_model in EES_models.items():
            model_folder = os.path.dirname(EES_model)
            model_filename = '.'.join(os.path.basename(EES_model).split(".")[:-1])
            base_path = os.path.join(
                model_folder, model_filename, ".parametric"
            )
            path = {
                "base_path": base_path,
                "csv": os.path.join(base_path, ".results", "dolar", "parametric_result.csv"),
                "plot": os.path.join(base_path, ".plots", "analiseDolar")
            }
            if not os.path.exists(path.get("plot")):
                os.makedirs(path.get("plot"))
            paths.update({model_name: path})
        return paths

    def get_df(self):
        dfs = {}
        for model_name, path in self.paths.items():
            filepath = path.get("csv")
            df = pd.read_csv(filepath, sep=";")
            if 'tpb2' in df.columns:
                df = df.rename(columns={'tpb2': 'payback_simples_2'})
            dfs.update({model_name: df})
        return dfs

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

    def get_titles(self, lang):
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                "num": "Tempo de Payback Simples vs Preço do Dólar",
                "title": "Tempo de Payback Simples vs Preço do Dólar",
                "dolar": "Preço do Dólar (R$)",
                "tpb": "Tempo de Payback Simples (anos)",
                "models": {
                    "libr": r"Trigeração (SRA: $LiBr/H_2O$)",
                    "nh3": r"Trigeração (SRA: $NH_3/H_2O$)",
                },
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                "num": "Payback Time vs Dollar prices",
                "title": "Payback Time vs Dollar prices",
                "dolar": "Dollar Price (R$)",
                "tpb": "Payback Time (years)",
                "models": {
                    "libr": r"Trigeneration (ARS: $LiBr/H_2O$)",
                    "nh3": r"Trigeneration (ARS: $NH_3/H_2O$)",
                },
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def generate(self, lang):
        titles = self.get_titles(lang)

        # Dessalinização
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7), num=titles["num"])
        ax1.set_title(titles["title"])
        ax1.set_xlabel(titles["dolar"])
        ax1.set_ylabel(titles["tpb"])

        for model_name, df in self.dfs.items():
            ax1.plot(df["dolar"], df["payback_simples_2"], label=titles['models'][model_name])

        # Axis Limits
        ax1.set_ylim(
            0.95 * min([df["payback_simples_2"].min() for _, df in self.dfs.items()]),
            1.05 * max([df["payback_simples_2"].max() for _, df in self.dfs.items()]),
        )
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(
            os.path.join(self.paths['libr'].get("plot"),
                         f"plot_{lang}_dolar_vs_payback.pdf")
        )

    def generate_artigo(self, model_name):
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
        ax1.set_title("Payback time vs Dollar price")
        ax1.set_xlabel("Dollar price (R$)")
        ax1.set_ylabel("Simple Payback Time (years)")

        df = self.dfs[model_name]
        ax1.plot(df["dolar"], df["payback_simples_2"], color="#306bac")

        # Axis Limits
        ax1.set_ylim(
            0.95 * df["payback_simples_2"].min(),
            1.05 * df["payback_simples_2"].max(),
        )
        # ax1.legend()
        fig1.tight_layout()
        fig1.savefig(
            os.path.join(r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_LiBrH2O\paper",
                         "plot_dolar_vs_payback_artigo.pdf")
        )
        fig1.savefig(
            os.path.join(r"C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_LiBrH2O\paper",
                         "plot_dolar_vs_payback_artigo.jpg")
        )
