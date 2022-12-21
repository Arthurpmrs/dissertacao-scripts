import os
import time
import json
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import ticker
from ees.optimization import OptimizationStudy
from .utilities import check_model_path, d_difference, add_folder, ParamAnalysisMissingError


class OptParamAnalysis:

    def __init__(
        self, EES_exe: str, EES_model: str, inputs: dict, outputs: list,
        decision_variables: dict, base_config: dict, params: dict, run_ID: str = None
    ):
        self.EES_exe = EES_exe
        self.EES_model = check_model_path(EES_model)
        self.run_ID = run_ID if run_ID else str(round(time.time()))
        self.paths = self.set_paths()
        self.logger = self.setup_logging()
        self.inputs = inputs
        self.outputs = outputs
        self.decision_variables = decision_variables
        self.base_config = base_config
        self.params = params

    def set_paths(self) -> str:
        """Configurar Paths das pastas para os resultados."""

        model_folder = os.path.dirname(self.EES_model)
        model_name = '.'.join(os.path.basename(self.EES_model).split('.')[:-1])
        base_folder = add_folder(model_folder, "results", model_name, ".optParamAnalysis", self.run_ID)
        paths = {
            "base_folder": base_folder,
            "plots": add_folder(base_folder, ".plots"),
            "logs": add_folder(base_folder, ".log"),
            "results": add_folder(base_folder, ".results")
        }
        return paths

    def set_optimizer(self, optimizer: OptimizationStudy):
        """Especificar otimizador."""

        self.optimizer = optimizer

    def set_target_variable(self, target_variable, target_variable_display="", problem="max"):
        """Especificar variável alvo."""

        self.target_variable = target_variable
        self.target_variable_display = target_variable_display
        self.optimization_problem = problem.lower()

    def setup_logging(self) -> logging.Logger:
        """Configurar logger."""

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s:%(filename)s:%(message)s')

        file_handler = logging.FileHandler(
            os.path.join(
                self.paths['logs'],
                f'{self.run_ID}_opt_parameter_analysis.log'
            ))
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def param_analysis(self) -> dict:
        """Executa a análise de sensibilidade dos parâmetros internos."""

        results = {}
        for param, values in self.params.items():
            param_results = {}
            for i, value in enumerate(values):

                if value == None:
                    continue

                config = {**self.base_config}
                config.update(value)

                print(" ")
                print(f"Iniciando nova análise de >{param}< com os seguintes valores:")
                print(value)

                filtered_result = {}
                eesopt = self.optimizer(self.EES_exe, self.EES_model, self.inputs, self.outputs)
                eesopt.set_decision_variables(self.decision_variables)
                eesopt.set_target_variable(self.target_variable, self.target_variable_display, self.optimization_problem)
                result = eesopt.execute(config)

                if result == {}:
                    param_results.update({eesopt.runID: None})
                    continue

                filtered_result = {
                    result["run_ID"]: {
                        "best_target": result["best_target"],
                        "best_individual": result["best_individual"],
                        "generations": result["generations"],
                        "evolution_time": result["evolution_time"],
                        "param_studied": param,
                        "param_value": value,
                        "config": result["config"],
                        "best_output": result["best_output"],
                    }
                }
                param_results.update(filtered_result)

                # Save run result to file
                folderpath = add_folder(self.paths["results"], self.target_variable, param)

                filename = f"result_run_{i + 1}.json"
                filepath = os.path.join(folderpath, filename)

                with open(filepath, 'w') as jsonfile:
                    json.dump(filtered_result, jsonfile)

                filename_readable = f"result-readable_run_{i + 1}.json"
                filepath_readable = os.path.join(folderpath, filename_readable)

                with open(filepath_readable, 'w') as jsonfile:
                    json.dump(filtered_result, jsonfile, indent=4)

                del eesopt

            results.update({param: param_results})

        self.results = results
        return results

    def compute_best_results(self):
        """Determinar configuração de parâmetros com melhor resultado."""

        if not self.results:
            raise ParamAnalysisMissingError("Não foi realizada uma análise de paâmetros")

        for param, values in self.results.items():
            if None in values.keys():
                raise ParamAnalysisMissingError("Uma das análises falhou!")

            self.log(f"Análise de: {param}")
            sorted_results = sorted(
                [(idx, result["best_target"][self.target_variable]) for idx, result in values.items()],
                key=lambda x: x[1],
                reverse=True
            )
            for result in sorted_results:
                self.log(f"ID: {result[0]} | {self.target_variable}: {result[1]} | {param}: {values[result[0]]['param_value']}")

            best_result = values[sorted_results[0][0]]

            self.log("Informações do melhor resultado:")
            self.log(f"Run ID: {sorted_results[0][0]}")
            self.log(f"Tempo de Execução: {datetime.timedelta(seconds=best_result['evolution_time'])}")
            self.log(f"Gerações para a convergência: {best_result['generations']}")
            self.log("Melhor valor da função objetivo:")
            self.log(best_result["best_target"])
            self.log("Melhor Indivíduo (Conjunto de variáveis de decisão):")
            self.log({k: round(v, 4) for (k, v) in best_result["best_individual"].items()})
            self.log("Parâmetros do Algoritmo Genético:")
            self.log(best_result["config"])
            self.log("Output referente ao melhor indivíduo: ")
            self.log({k: round(v, 4) for (k, v) in best_result["best_output"].items()})
            self.log(" ")

    def get_result_from_file(self) -> dict:
        """Lê resultados da análise a partir de arquivos salvos em disco."""

        results = {}
        for param, values in self.params.items():
            folderpath = os.path.join(self.paths["results"], self.target_variable, param)
            param_results = {}
            for i, _ in enumerate(values):
                filename = f"result_run_{i + 1}.json"
                filepath = os.path.join(folderpath, filename)

                with open(filepath, 'r') as jsonfile:
                    param_results.update(json.load(jsonfile))

            results.update({param: param_results})
        self.results = results
        return results

    def log(self, text: str, verbose=True):
        """Registrar mensagens de log."""

        self.logger.info(text)
        if verbose:
            print(text)


class OptParamAnalysisGraphs:

    def __init__(self, EES_model: str, run_ID: str, results: dict):
        self.run_ID = run_ID
        self.results = results
        self.plots_folder = self.set_plots_folder(EES_model)
        self.set_matplotlib_globalconfig()

    def set_plots_folder(self, EES_model) -> str:
        model_folder = os.path.dirname(EES_model)
        model_filename = '.'.join(os.path.basename(EES_model).split('.')[:-1])

        plots_folder = add_folder(model_folder, "results", model_filename,
                                  ".optParamAnalysis", self.run_ID, ".plots")
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
