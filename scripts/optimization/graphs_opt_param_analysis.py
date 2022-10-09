import os
import sys
import math
sys.path.append(os.path.join(os.getcwd(), 'src'))
from ees.optimization_param_graphs import OptParamGraphs
from ees.utilities import d_difference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import ticker


class DefaultParamAnalysisGraph(OptParamGraphs):

    def __init__(self, EES_model: str, run_ID: str, results: dict):
        super().__init__(EES_model, run_ID, results)

    def get_titles(self, lang: str) -> dict:
        if lang in ["pt-BR", "pt_BR", "ptbr"]:
            titles = {
                'cxTwoPoint': 'PontoDuplo',
                'cxSimulatedBinaryBounded': 'SBB',
                'cxBlend': 'Mistura',
                'mutGaussian': "Gaussiana",
                'mutPolynomialBounded': 'PB',
                'mutUniformInt': 'UI',
                'selTournament': 'Tourn.',
                'selBest': 'Melhor',
                'selRoulette': 'Roleta',
                'selStochasticUniversalSampling': 'SUS',
                'generations': 'Gerações',
                'time-title': 'Tempo de Execução',
                'time-axis': 'Tempo (s)',
                'dv-title': 'Variáveis de Decisão',
                'dv-axis': 'Variáveis'
            }
        elif lang in ["en-US", "en_US", "enus"]:
            titles = {
                'cxTwoPoint': 'TwoPoint',
                'cxSimulatedBinaryBounded': 'SBB',
                'cxBlend': 'Blend',
                'mutGaussian': "Gaussian",
                'mutPolynomialBounded': 'PB',
                'mutUniformInt': 'UniformInt',
                'selTournament': 'Tourn.',
                'selBest': 'Best',
                'selRoulette': 'Roulette',
                'selStochasticUniversalSampling': 'SUS',
                'generations': 'Generations',
                'time-title': 'Execution Time',
                'time-axis': 'Time (s)',
                'dv-title': 'Decision Variables',
                'dv-axis': 'Variables'
            }
        else:
            raise ValueError("Linguagem não suportada!")

        return titles

    def round_target(self, target_values):
        if self.target_variable == "EUF_sys":
            case = 4
        elif self.target_variable == "psi_sys_1":
            case = 3
        elif self.target_variable == "m_dot[38]":
            case = 4
        else:
            case = 4

        return [round(tv, case) for tv in target_values]

    def generate(self, lang: str = "pt-BR"):
        titles = self.get_titles(lang)
        yticks = []
        for param, values in self.results.items():
            labels = []
            generations = []
            times = []
            target_values = []
            decision_variables = []
            old_dict = values[list(values.keys())[-1]]["param_value"][next(iter(values[list(values.keys())[-1]]["param_value"]))]
            for idx, value in values.items():
                target_variable = list(value["best_target"].keys())[0]
                for p, v in value["param_value"].items():
                    if isinstance(v, dict):
                        v_unclean = v
                        v = d_difference(old_dict, v_unclean)
                        old_dict = v_unclean
                labels.append(str(v))
                generations.append(value["generations"])
                times.append(round(value["evolution_time"]))
                target_values.append(value["best_target"][next(iter(value["best_target"]))])
                decision_variables.append(value["best_individual"])

            for i, label in enumerate(labels):
                if label in titles.keys():
                    labels[i] = titles[label]

            dv_df = pd.DataFrame(decision_variables, index=labels)

            width = 0.35
            target_values = self.round_target(target_values)

            fig = plt.figure(figsize=(10, 10))
            sub1 = fig.add_subplot(2, 2, (1, 2))
            rec1 = sub1.bar(labels, target_values, width=width)
            sub1.set_title(self.target_variable_display)
            sub1.set_ylabel(self.target_variable_display)
            sub1.bar_label(rec1, padding=3, fontsize=12, fontfamily="CMU Serif")
            yticks = sub1.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub1.set_yticks(yticks)
            sub1.set_title("a)", loc="left",
                           style="italic", fontweight="normal")

            sub2 = fig.add_subplot(2, 2, 3)
            rec2 = sub2.bar(labels, generations, width=width)
            sub2.bar_label(rec2, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub2.set_title(titles["generations"])
            sub2.set_ylabel(titles["generations"])
            yticks = sub2.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub2.set_yticks(yticks)
            sub2.set_title("b)", loc="left",
                           style="italic", fontweight="normal")

            sub3 = fig.add_subplot(2, 2, 4)
            rec3 = sub3.bar(labels, times, width=width)
            rec3 = sub3.bar_label(rec3, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub3.set_title(titles["time-title"])
            sub3.set_ylabel(titles["time-axis"])
            yticks = sub3.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub3.set_yticks(yticks)
            sub3.set_title("c)", loc="left",
                           style="italic", fontweight="normal")

            fig.tight_layout()

            folder = os.path.join(self.plots_folder, lang)

            if not os.path.exists(folder):
                os.makedirs(folder)

            filename = os.path.join(folder, f"{param}.pdf")
            plt.savefig(filename)
            fig.clf()

    def generate_log(self, lang: str = "pt-BR"):
        titles = self.get_titles(lang)
        yticks = []
        for param, values in self.results.items():
            labels = []
            generations = []
            times = []
            target_values = []
            decision_variables = []
            old_dict = values[list(values.keys())[-1]]["param_value"][next(iter(values[list(values.keys())[-1]]["param_value"]))]
            for idx, value in values.items():
                target_variable = list(value["best_target"].keys())[0]
                for p, v in value["param_value"].items():
                    if isinstance(v, dict):
                        v_unclean = v
                        v = d_difference(old_dict, v_unclean)
                        old_dict = v_unclean
                labels.append(str(v))
                generations.append(value["generations"])
                times.append(round(value["evolution_time"]))
                target_values.append(value["best_target"][next(iter(value["best_target"]))])
                decision_variables.append(value["best_individual"])

            for i, label in enumerate(labels):
                if label in titles.keys():
                    labels[i] = titles[label]

            dv_df = pd.DataFrame(decision_variables, index=labels)

            width = 0.35
            fig = plt.figure(figsize=(10, 10))

            sub1 = fig.add_subplot(2, 2, (1, 2))

            # Arredondando para 4 casos para evitar problemas de visualização no gráfico log
            target_values = [round(v, 4) for v in target_values]

            rec1 = sub1.bar(labels, target_values, width=width)
            sub1.set_title(self.target_variable_display)
            sub1.set_ylabel(self.target_variable_display)
            sub1.bar_label(rec1, padding=3, fontsize=12, fontfamily="CMU Serif")

            # Transforma o gráfico de barras na escala logarítimica.
            sub1.set_yscale("log")
            sub1.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%2.4f'))

            ticks = sub1.get_yticks(minor=True)
            sub1.set_ylim(sub1.get_ylim()[0], ticks[-1])
            sub1.set_title("a)", loc="left",
                           style="italic", fontweight="normal")

            sub2 = fig.add_subplot(2, 2, 3)
            rec2 = sub2.bar(labels, generations, width=width)
            sub2.bar_label(rec2, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub2.set_title(titles["generations"])
            sub2.set_ylabel(titles["generations"])
            yticks = sub2.get_yticks()
            sub2.set_ylim(sub2.get_ylim()[0], yticks[-1])
            sub2.set_title("b)", loc="left",
                           style="italic", fontweight="normal")

            sub3 = fig.add_subplot(2, 2, 4)
            rec3 = sub3.bar(labels, times, width=width)
            rec3 = sub3.bar_label(rec3, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub3.set_title(titles["time-title"])
            sub3.set_ylabel(titles["time-axis"])
            yticks = sub3.get_yticks()
            sub3.set_ylim(sub3.get_ylim()[0], yticks[-1])
            sub3.set_title("c)", loc="left",
                           style="italic", fontweight="normal")

            fig.tight_layout()
            folder = os.path.join(self.plots_folder, lang, "logplot")

            if not os.path.exists(folder):
                os.makedirs(folder)

            filename = os.path.join(folder, f"{param}.pdf")
            plt.savefig(filename)
            fig.clf()

    def generate_old(self, lang: str = "pt-BR"):
        titles = self.get_titles(lang)
        yticks = []
        for param, values in self.results.items():
            labels = []
            generations = []
            times = []
            target_values = []
            decision_variables = []
            old_dict = values[list(values.keys())[-1]]["param_value"][next(iter(values[list(values.keys())[-1]]["param_value"]))]
            for idx, value in values.items():
                target_variable = list(value["best_target"].keys())[0]
                for p, v in value["param_value"].items():
                    if isinstance(v, dict):
                        v_unclean = v
                        v = d_difference(old_dict, v_unclean)
                        old_dict = v_unclean
                labels.append(str(v))
                generations.append(value["generations"])
                times.append(round(value["evolution_time"]))
                target_values.append(value["best_target"][next(iter(value["best_target"]))])
                decision_variables.append(value["best_individual"])

            for i, label in enumerate(labels):
                if label in titles.keys():
                    labels[i] = titles[label]

            dv_df = pd.DataFrame(decision_variables, index=labels)

            width = 0.35
            fig = plt.figure(figsize=(10, 14))

            sub1 = fig.add_subplot(3, 2, (1, 2))
            rec1 = sub1.bar(labels, target_values, width=width)
            sub1.set_title(self.target_variable_display)
            sub1.set_ylabel(self.target_variable_display)
            sub1.bar_label(rec1, padding=3, fontsize=12, fontfamily="CMU Serif")
            yticks = sub1.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub1.set_yticks(yticks)
            # sub1.set_ylim([0, math.ceil(max(target_values))])

            sub0 = fig.add_subplot(3, 2, (3, 4))
            new_columns = []
            for column in dv_df:
                if 'T' in column:
                    column = column.replace("[", "_{").replace("]", "}")
                    column = f"$ {column} $ (ºC)"
                new_columns.append(column)

            dv_df.columns = new_columns
            dv_df.plot(ylabel=titles["dv-axis"], kind='bar', stacked=False, ax=sub0, title=titles["dv-title"], rot=360)

            sub0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
                        fancybox=False, shadow=False, ncol=7)

            sub2 = fig.add_subplot(3, 2, 5)
            rec2 = sub2.bar(labels, generations, width=width)
            sub2.bar_label(rec2, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub2.set_title(titles["generations"])
            sub2.set_ylabel(titles["generations"])
            yticks = sub2.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub2.set_yticks(yticks)

            sub3 = fig.add_subplot(3, 2, 6)
            rec3 = sub3.bar(labels, times, width=width)
            rec3 = sub3.bar_label(rec3, padding=3, fontsize=12, fontfamily="CMU Serif", wrap=True)
            sub3.set_title(titles["time-title"])
            sub3.set_ylabel(titles["time-axis"])
            yticks = sub3.get_yticks()
            np.append(yticks, (yticks[-1] - yticks[-2]))
            sub3.set_yticks(yticks)

            fig.tight_layout()
            pos = sub0.get_position()
            sub1.set_position([pos.x0, pos.y0 + pos.height + 0.07 * 0.55,
                               pos.width,
                               pos.height + 0.07 * 0.45])

            filename = os.path.join(self.plots_folder, f"{lang}_analysis-of-{param}.jpg")
            plt.savefig(filename)
            fig.clf()
