import os
import sys
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from .utilities import check_model_path, get_base_folder, add_folder, ParamAnalysisMissingError


class OptParamGraphs:

    def __init__(self, EES_model: str, run_ID: str, results: dict):
        self.base_path = get_base_folder(EES_model)
        self.run_ID = run_ID
        self.results = results
        self.plots_folder = self.set_plots_folder()
        self.set_matplotlib_globalconfig()

    def set_plots_folder(self) -> str:
        plots_folder = add_folder(self.base_path, ".optParamAnalysis", self.run_ID, ".plots")
        return plots_folder

    def set_target_variable(self, target_variable, target_variable_display="", problem="max"):
        self.target_variable = target_variable
        self.target_variable_display = target_variable_display
        self.optimization_problem = problem

    def set_matplotlib_globalconfig(self):
        plt.style.use("ggplot")

        font_dir = [r"C:\Root\Download\computer-modern"]
        for font in font_manager.findSystemFonts(font_dir):
            font_manager.fontManager.addfont(font)

        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["font.family"] = "CMU Serif"

        axes = {
            "labelsize": 20,
            "titlesize": 16,
            "titleweight": "bold",
            "labelweight": "bold",
        }
        matplotlib.rc("axes", **axes)

        lines = {"linewidth": 2}
        matplotlib.rc("lines", **lines)

        legends = {"fontsize": 11}
        matplotlib.rc("legend", **legends)

        savefig = {"dpi": 300}
        matplotlib.rc("savefig", **savefig)

        matplotlib.rcParams["ytick.labelsize"] = 15
        matplotlib.rcParams["xtick.labelsize"] = 15
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler('color', ['004c6d', '175e7f', '297191', '3984a3',
                                                                             '4998b6', '5aadc8', '6bc1da', '7dd6ed', '8fecff'])
