import os
import sys
import time
import math
import json
import logging
import datetime
from icecream import ic
from ees.optimization import OptimizationStudy
from .utilities import check_model_path, get_base_folder, add_folder, ParamAnalysisMissingError


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
        """Basic paths configuration."""
        model_folder = get_base_folder(self.EES_model)
        base_folder = add_folder(model_folder, ".optParamAnalysis", self.run_ID)
        paths = {
            "base_folder": base_folder,
            "plots": add_folder(base_folder, ".plots"),
            "logs": add_folder(base_folder, ".log"),
            "results": add_folder(base_folder, ".results")
        }
        return paths

    def set_optimizer(self, optimizer: OptimizationStudy):
        self.optimizer = optimizer

    def set_target_variable(self, target_variable, target_variable_display="", problem="max"):
        self.target_variable = target_variable
        self.target_variable_display = target_variable_display
        self.optimization_problem = problem.lower()

    def setup_logging(self) -> logging.Logger:
        """Logging configuration."""
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
        """Execute metaheuristic optimization parameter sensitivity analysis. Returns and sets results dict."""
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
            self.log(f"Melhor valor da função objetivo:")
            self.log(best_result["best_target"])
            self.log(f"Melhor Indivíduo (Conjunto de variáveis de decisão):")
            self.log({k: round(v, 4) for (k, v) in best_result["best_individual"].items()})
            self.log(f"Parâmetros do Algoritmo Genético:")
            self.log(best_result["config"])
            self.log("Output referente ao melhor indivíduo: ")
            self.log({k: round(v, 4) for (k, v) in best_result["best_output"].items()})
            self.log(" ")

    def get_result_from_file(self) -> dict:
        """Read parameter analisys results from folder. Returns results dict."""
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
        self.logger.info(text)
        if verbose:
            print(text)
