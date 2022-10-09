from ctypes import ArgumentError
import os
import sys
import math
import time
import logging
import traceback
import win32ui
import dde
import pyperclip
import subprocess
import pandas as pd
from icecream import ic
from rich import print
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
