import os
import math
import time
import json
import logging
import datetime
import random
import traceback
import win32ui
import dde
import pyperclip
import subprocess
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from ctypes import ArgumentError
from rich import print
from ctypes import ArgumentError
from deap import base
from deap import creator
from deap import tools
from .utilities import check_model_path


class OptimizationStudy:

    def __init__(self, EES_exe, EES_model, base_case_inputs, outputs, runID=None):
        self.EES_exe = EES_exe
        self.EES_model = check_model_path(EES_model)
        self.base_case_inputs = base_case_inputs
        self.outputs = outputs
        self.runID = runID if runID else str(round(time.time()))
        self.paths = self.set_paths()
        self.logger = self.setup_logging()
        self.consecutive_error_count = 0
        self.is_ready = {
            'target_variable': False,
            'decision_variables': False,
            'DDE': False,
            'optimizer': False
        }

    def check_is_ready(self):
        if not all(value == True for value in self.is_ready.values()):
            raise Exception("Algo de errado ocorreu!")

    def set_paths(self):
        model_folder = os.path.dirname(self.EES_model)
        model_filename = os.path.basename(self.EES_model)
        base_folder = os.path.join(
            model_folder,
            "results",
            '.'.join(model_filename.split('.')[:-1])
        )
        id_folder = os.path.join(base_folder, ".opt", self.runID)
        paths = {
            "base_folder": base_folder,
            "id_folder": id_folder,
            "plots": os.path.join(id_folder, ".plots"),
            "logs": os.path.join(id_folder, ".logs"),
            "results": os.path.join(id_folder, ".results")
        }
        # Check if folders exists and if not, creates them.
        for _, path in paths.items():
            if not os.path.exists(path):
                os.makedirs(path)

        return paths

    def set_target_variable(self, target_variable, target_variable_display="", problem="max"):
        self.target_variable = target_variable
        self.target_variable_display = target_variable_display
        self.optimization_problem = problem.lower()
        if problem.lower() == "min":
            self.invalid_target_value = math.inf
        elif problem.lower() == "max":
            self.invalid_target_value = -math.inf
        else:
            raise ArgumentError("Wrong problem value. Must be max or min.")
        self.is_ready['target_variable'] = True

    def set_decision_variables(self, decision_variables):
        """Adds the decision variables dict as a attribute of the class."""
        self.decision_variables = decision_variables
        self.is_ready['decision_variables'] = True

    def setup_DDE(self):
        # Closes any instance of EES that are already running.
        if "EES.exe" in str(subprocess.check_output('tasklist')):
            self.log(">> Uma instância do EES foi encontrada aberta. Ela será fechada.")
            os.system("taskkill /f /im  EES.exe")

        self.log(f">> Abrindo o EES em {self.EES_exe}")
        subprocess.Popen([self.EES_exe, '/hide'], shell=True, close_fds=True)
        time.sleep(15)
        self.server = dde.CreateServer()
        self.server.Create("PyhtonDDExyUiosdjU")

        self.connector = dde.CreateConversation(self.server)
        self.connector.ConnectTo("EES", "DDE")

        self.log(f">> Abrindo modelo {self.EES_model}")
        self.connector.Exec(f"[Open {self.EES_model}]")
        self.connector.Exec(f"[HideWindow ErrorMessages]")
        self.connector.Exec(f"[HideWindow WarningMessages]")

        self.is_ready['DDE'] = True

    def close(self):
        self.log(">> Fechando o EES.")
        try:
            self.connector.Exec("[QUIT]")
        except dde.error as e:
            self.logger.exception(e)
            os.system("taskkill /f /im  EES.exe")
        self.server.Shutdown()
        time.sleep(10)

    def cleanup_dde(self):
        """Closes DDE Server and shutsdown EES if opened, so it can be restarted."""
        try:
            self.server.Shutdown()
            os.system("taskkill /f /im  EES.exe")
            del self.connector
            del self.server
        except Exception as deletion_exception:
            # A Exception could happen if the server and EES are already closed.
            self.logger.exception(deletion_exception)

    def dde_error_handler(self, error):
        """Handles the restart of EES if DDE exec error persists."""
        self.consecutive_error_count += 1
        self.log(f">> Erro: Conexão DDE falhou. A variável target para esta rodada será considerado 0.")
        self.log(traceback.format_exc(), verbose=False)
        if self.consecutive_error_count > 2:
            self.log(">> O erro persiste. Reiniciando o EES.")
            self.cleanup_dde()
            self.setup_DDE()
            self.consecutive_error_count = 0

    def eval_EES_model(self, individual):
        # Remove 0 and negative values from decision variables
        for variable, limits in zip(individual, self.decision_variables.values()):
            if variable <= 0 or (variable < limits[0] or variable > limits[1]):
                return (self.invalid_target_value, )

        try:
            self.prepare_inputs(individual)
            self.connector.Exec('[SOLVE]')
            target_variable = self.get_output()
            self.consecutive_error_count = 0
        except dde.error as e:
            self.dde_error_handler(e)
            target_variable = self.invalid_target_value

        return (target_variable, )

    def prepare_inputs(self, individual):
        new_inputs = {}
        new_inputs.update(self.base_case_inputs)
        for (variable, _), ind_variable_value in zip(self.decision_variables.items(), individual):
            new_inputs.update({variable: ind_variable_value})

        input_chunks = OptimizationStudy.variable_dict_splitter(new_inputs, (254 - 35))
        for chunk in input_chunks:
            input_variables = " ".join([str(v) for v in chunk.keys()])
            input_values = " ".join([str(v) for v in chunk.values()])
            pyperclip.copy(input_values)
            self.connector.Exec(f"[Import \'Clipboard\' {input_variables}]")
            pyperclip.copy('')

    def get_output(self):
        output_chunks = OptimizationStudy.variable_list_splitter(self.outputs, (254 - 35))
        results = []
        error_has_ocorred = False
        for chunk in output_chunks:
            output_variables = " ".join([str(var) for var in chunk])
            self.connector.Exec(f"[Export \'Clipboard\' {output_variables}]")
            result = pyperclip.paste()
            pyperclip.copy('')

            result = result.replace("\t", " ").replace("\r\n", " ")
            results.extend(result.split(" "))

        self.output_dict = {}
        error_count = 0
        for output, result in zip(self.outputs, results):
            try:
                value = float(result)
                # Prevents EES from returning values of a run thar has not converged. I.E. some variables have their guess value.
                if result == "1.00000000E+00":
                    error_count += 1
            except ValueError:
                value = 0
                error_has_ocorred = True
            self.output_dict.update({output: value})

        if error_has_ocorred or error_count > 3:
            self.log(">> Erro: O EES não exportou valores corretos. O indivíduo é inválido.")
            self.output_dict.update({self.target_variable: self.invalid_target_value})

        return self.output_dict[self.target_variable]

    def setup_logging(self):
        logfolder = self.paths['logs']

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s:%(filename)s:%(message)s')

        file_handler = logging.FileHandler(
            os.path.join(
                logfolder,
                f'{self.__class__.__name__}.log'
            ))
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def log(self, text, verbose=True):
        self.logger.info(text)
        if verbose:
            print(text)

    @staticmethod
    def variable_dict_splitter(d, max_len):
        l = d.keys()
        len_of_l = len(" ".join(l))
        num_of_chunks = math.ceil(len_of_l / max_len)
        chunk_size = math.ceil(len_of_l / num_of_chunks)

        chunks = []
        chunk = []
        for variable in l:
            chunk.append(variable)
            if len(" ".join(chunk)) >= chunk_size:
                chunks.append({var: d[var] for var in chunk})
                chunk = []
        else:
            chunks.append({var: d[var] for var in chunk})

        return chunks

    @staticmethod
    def variable_list_splitter(l, max_len):
        len_of_l = len(" ".join(l))
        num_of_chunks = math.ceil(len_of_l / max_len)
        chunk_size = math.ceil(len_of_l / num_of_chunks)

        chunks = []
        chunk = []
        for variable in l:
            chunk.append(variable)
            if len(" ".join(chunk)) >= chunk_size:
                chunks.append(chunk)
                chunk = []
        else:
            chunks.append(chunk)
        return chunks


class GAOptimizationStudy(OptimizationStudy):

    def __init__(self, EES_exe, EES_model, base_case_inputs, outputs, runID=None):
        super().__init__(EES_exe, EES_model, base_case_inputs, outputs, runID)

    def feasible(self, individual):
        self.eval_EES_model(individual)
        check = []
        for _, value in self.output_dict.items():
            if value >= 0:
                check.append(True)
            else:
                check.append(False)
        if all(value == True for value in check):
            return True
        return False

    def setup_optimizer(self, config):
        if self.optimization_problem == "min":
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        elif self.optimization_problem == "max":
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        else:
            raise ArgumentError("Not valid optimization problem.")
        self.toolbox = base.Toolbox()

        attrs = []
        # Attribute generator
        for variable, limits in self.decision_variables.items():
            attr_name = f"attr_{variable}"
            self.toolbox.register(attr_name, random.uniform, limits[0], limits[1])
            attrs.append(getattr(self.toolbox, attr_name))

        # Structure initializers
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              tuple(attrs), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_EES_model)
        self.toolbox.register("mate", getattr(tools, config["crossover"]["method"]), **config["crossover"]["params"])
        self.toolbox.register("mutate", getattr(tools, config["mutation"]["method"]), **config["mutation"]["params"])
        self.toolbox.register("select", getattr(tools, config["selection"]["method"]), **config["selection"]["params"])
        self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.feasible, self.invalid_target_value))

        self.is_ready['optimizer'] = True

    def execute(self, config):
        result = {}
        try:
            self.setup_DDE()
            self.setup_optimizer(config)
            self.check_is_ready()
            result = self.optimize(config)
            del creator.Individual
            # Necessário saber se é maximização ou minimização para deletar o objeto correto.
            if self.optimization_problem == "min":
                del creator.FitnessMin
            elif self.optimization_problem == "max":
                del creator.FitnessMax
        except Exception as e:
            self.logger.exception(e)
            self.log(">> Erro: Algo de errado ocorreu. Esta execução está comprometida.")
            self.log(traceback.format_exc())
        finally:
            self.close()
        return result

    def optimize(self, config):
        """Genetic Algorithm optimization algorithm."""
        # Tempo inicial
        start_time = time.time()

        # Configuração da Seed
        random.seed(config["seed"])

        # Population
        pop_num = config["population"]
        pop = self.toolbox.population(pop_num)

        # Crossover and mutation rates
        CXPB = config["crossover"]["rate"]
        MUTPB = config["mutation"]["rate"]

        # Starting Evolution
        self.log("---- Início da evolução ----")

        # Evaluate the entire population
        fitnesses = []
        for i, ind in enumerate(pop):
            result = self.toolbox.evaluate(ind)
            self.log(f"Nº: {i + 1} | {self.target_variable}: {result[0]}", verbose=config["verbose"])
            self.log(
                f"Ind: {[f'{var}: {i:.4f}' for i, var in zip(ind, self.decision_variables.keys())]}",
                verbose=config["verbose"]
            )
            fitnesses.append(result)

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        self.log(f"Calculados {len(pop)} indivíduos")

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0
        gen_time_old = start_time
        rates = []
        gen_history = []
        fits_old = self.invalid_target_value
        max_same_target_count = 5
        same_target_count = 0

        # Begin the evolution
        while g < config["max_generation"]:
            # A new generation
            g += 1
            self.log(" ")
            self.log(f"------- Geração {g} -------")

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = []
            for i, ind in enumerate(invalid_ind):
                result = self.toolbox.evaluate(ind)
                self.log(f"Nº: {i + 1} | {self.target_variable}: {result[0]}", verbose=config["verbose"])
                self.log(
                    f"Ind: {[f'{var}: {i:.4f}' for i, var in zip(ind, self.decision_variables.keys())]}",
                    verbose=config["verbose"]
                )
                fitnesses.append(result)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.log(f"Calculados {len(invalid_ind)} indivíduos")

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            n_show = 15
            pop_best_inds = self.get_best_inds(pop, n_show)
            best_ind = tools.selBest(pop_best_inds, 1)[0]

            self.log(f"Somente mostrando os TOP {n_show} indivíduos:")

            pop_list = []
            for ind in pop:
                target, variables = self.ind_to_dict(ind)
                variables = {k: round(v, 4) for (k, v) in variables.items()}
                pop_list.append({**target, **variables})

            df = pd.DataFrame(pop_list)
            # Ordenar o DataFrame da de acordo com o problema de otimização (max ou min).
            if self.optimization_problem == "max":
                rev = False
            elif self.optimization_problem == "min":
                rev = True
            df = df.sort_values(by=self.target_variable, ascending=rev).head(n_show)
            self.log(f"Indivíduos:\n{df}")

            # Error calculation (Depende se é maximização ou minimização.)
            if self.optimization_problem == "min":
                error = abs(fits_old - min(fits))
                fits_old = min(fits)
            elif self.optimization_problem == "max":
                error = abs(fits_old - max(fits))
                fits_old = max(fits)

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            # Calculate generation rate in individuals / minute
            current_time = time.time()
            gen_time = current_time - gen_time_old
            gen_time_old = current_time
            rate = len(invalid_ind) / (gen_time / 60)  # Individuals / minute
            rates.append(rate)

            print(" ")
            self.log("Estatísticas da geração:")
            self.log(f'Err: {error:.5f}')
            self.log(f"Min: {min(fits):.5f}")
            self.log(f"Max: {max(fits):.5f}")
            self.log(f"Avg: {mean:.5f}")
            self.log(f"Std: {std:.5f}")
            self.log(f"Rate: {rate:.2f}")

            self.eval_EES_model(best_ind)
            gen_history.append({
                "best_target": best_ind.fitness.values[0],
                "best_individual": best_ind,
                "error": error,
                "stats": {
                    "min": min(fits),
                    "max": max(fits),
                    "avg": mean,
                    "std": std,
                    "rate": rate
                },
                "best_output": self.output_dict
            })

            # Critério de convergência
            if error < config["cvrg_tolerance"]:
                same_target_count += 1
                if same_target_count > max_same_target_count:
                    self.log(f">> Critério de convergência atingido na geração {g}")
                    break
            else:
                same_target_count = 0

        self.log("---- Fim da evolução ----")
        delta_t = time.time() - start_time
        target, variables = self.ind_to_dict(best_ind)
        results = {
            "run_ID": self.runID,
            "best_target": target,
            "best_individual": variables,
            "evolution_time": delta_t,
            "generations": g,
            "avg_rate": sum(rates) / len(rates),
            "config": config,
            "best_output": gen_history[-1]["best_output"],
            "gen_history": gen_history,
        }

        self.display_results(results)
        self.save_to_json(results, "results")
        return results

    def save_to_json(self, results, filename):
        with open(os.path.join(self.paths["results"], f"{filename}.json"), "w") as jsonfile:
            json.dump(results, jsonfile)
        with open(os.path.join(self.paths["results"], f"readable-{filename}.json"), "w") as jsonfile:
            json.dump(results, jsonfile, indent=4)

    def ind_to_dict(self, ind):
        target = {self.target_variable: ind.fitness.values[0]}

        variables = {}
        for variable, value in zip(self.decision_variables.keys(), ind):
            variables.update({variable: value})

        return target, variables

    def get_best_inds(self, pop, size):
        if self.optimization_problem == "max":
            rev = True
        elif self.optimization_problem == "min":
            rev = False
        tuple_list = [(ind, ind.fitness.values[0]) for ind in pop]
        ordered_list = sorted(tuple_list, key=lambda x: x[1], reverse=rev)
        return [ind for ind, fitness in ordered_list[:size]]

    def display_results(self, results):
        self.log(f"Run ID: {self.runID}")
        self.log(f"Tempo de Execução: {datetime.timedelta(seconds=results['evolution_time'])}")
        self.log(f"Gerações para a convergência: {results['generations']}")
        self.log(f"Taxa Média de cálculo de indivíduos: {results['avg_rate']} indivíduos/minuto")
        self.log(f"Melhor valor da função objetivo:")
        self.log(results["best_target"])
        self.log(f"Melhor Indivíduo (Conjunto de variáveis de decisão):")
        self.log({k: round(v, 4) for (k, v) in results["best_individual"].items()})
        self.log(f"Parâmetros do Algoritmo Genético:")
        self.log(results["config"])
        self.log("Output referente ao melhor indivíduo: ")
        self.log({k: round(v, 4) for (k, v) in results["best_output"].items()})


class OptGraph:

    def __init__(self, model_path: str, idx: str = None):
        if idx:
            self.idx = idx
        else:
            self.idx = self.last_generated_idx()

        self.result_folder = self.get_result_folder(model_path, self.idx)
        self.plots_folder = self.set_plots_folder()
        self.results = self.load_results()
        self.set_matplotlib_globalconfig()

    def get_result_folder(self, model_path: str, idx: str) -> str:
        """Gerar Path para a pasta de resultados."""

        base_folder = os.path.dirname(model_path)
        model_name = '.'.join(os.path.basename(model_path).split('.')[:-1])
        result_folder = os.path.join(base_folder, "results", model_name, ".opt", idx)

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        return result_folder

    def set_plots_folder(self) -> str:
        """Gerar Path para a pasta dos plots."""
        plots_folder = os.path.join(self.result_folder, ".plots")

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        return plots_folder

    def last_generated_idx(self) -> int:
        """Seleciona o último caso executado."""

        ids_folder = os.path.dirname(self.result_folder)
        idxs = []
        for directory in os.listdir(ids_folder):
            filepath = os.path.join(ids_folder, directory)
            if os.path.isdir(filepath):
                idxs.append(directory)
        last_generated_idx = sorted(idxs, reverse=True)[0]
        return last_generated_idx

    def load_results(self) -> dict:
        """Carregar resultados da otimização."""

        filename = os.path.join(self.result_folder, ".results", "results.json")
        with open(filename, "r") as jsonfile:
            results = json.load(jsonfile)
        return results

    def set_matplotlib_globalconfig(self):
        """Configuração de estilo dos gráficos."""

        plt.style.use("ggplot")

        font_dir = [r"font/computer-modern"]
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
        """Geração dos títulos dos gráficos."""

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
        """Geração dos gráficos."""

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
