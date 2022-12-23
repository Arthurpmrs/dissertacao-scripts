import os
import time
import json
import pandas as pd
import subprocess
from icecream import ic
from .utilities import check_model_path


class SolveModel:

    def __init__(self, EES_exe, EES_model, inputs, outputs, runID=None):
        self.EES_exe = EES_exe
        self.EES_model = check_model_path(EES_model)
        self.runID = str(runID) if runID else str(round(time.time()))
        self.paths = self.set_paths(self.EES_model)
        self.inputs = inputs
        self.outputs = outputs
        self.datfiles = {}
        self.results = {}

    def set_paths(self, EES_model):
        """Set paths that will be used as a dictionary and creats them if not already."""

        model_folder = os.path.dirname(EES_model)
        model_filename = os.path.basename(EES_model)
        base_folder = os.path.join(
            os.path.dirname(EES_model),
            "results",
            '.'.join(model_filename.split('.')[:-1]),
            '.solver',
            self.runID
        )
        paths = {
            'model_path': EES_model,
            'model_folder': model_folder,
            'base_folder': base_folder,
        }

        # Check if folders exists and if not, creates them.
        for _, path in paths.items():
            if not os.path.exists(path):
                os.makedirs(path)
        return paths

    def handle_inputs(self):
        """Creation of .DAT files that will be imported by EES.

        Receives Macro string and adds to it.
        Creates list of output files.

        Returns:
            Macro String
        Sets:
            List of output files
        """
        input_string = self.prepare_input_strings()
        output_string = self.prepare_output_string()

        input_filepath = self.create_input_datfile(input_string)
        output_filepath = self.store_output_datfile(output_string)

        self.paths.update({
            'macro_path': self.setup_macro(
                input_filepath,
                output_filepath,
            )
        })

    def prepare_input_strings(self):
        """Creates string inputs for DAT files."""

        return ' '.join([str(var) for var in self.inputs.values()])

    def prepare_output_string(self):
        """Joins output variables together (Necessery for EES)."""

        return ' '.join(self.outputs)

    def create_input_datfile(self, input_string):
        """Creates input datfile and stores path in self.datfiles['inputs'] for each parametric value."""

        filepath = os.path.join(self.paths['base_folder'], f'input.dat')

        with open(filepath, 'w') as datfile:
            datfile.write(input_string)

        self.datfiles['input'] = filepath
        return filepath

    def store_output_datfile(self, output):
        """Stores output file path in self.datfiles['outputs']"""

        filepath = os.path.join(self.paths['base_folder'], f'OUTPUT.DAT')
        self.datfiles['output'] = filepath
        return filepath

    def setup_macro(self, input_filepath, output_filepath):
        """Updates macro string with EES commands to read from DATFILE, solve and export to DATFILE."""

        input_names_string = ' '.join(self.inputs.keys())
        output_names_string = ' '.join(self.outputs)
        csv_filepath = os.path.join(self.paths['base_folder'], 'ARRAYS.csv')

        macro_header = "//WINDOWSIZE 0 401 1496 317\n"
        macro_header += f'Open \'{self.EES_model}\'\n'
        macro_header += 'Units SI C kPa kJ Mass\n'
        macro_string = macro_header

        macro_string += f'Import \'{input_filepath}\' {input_names_string}\n'
        macro_string += 'Solve\n'
        macro_string += f'SaveArrays \'Main\' \'{csv_filepath}\' /N\n'
        macro_string += f'Export \'{output_filepath}\' {output_names_string}\n'
        macro_string += 'Quit'

        macro_filepath = os.path.join(self.paths['base_folder'], 'macro.emf')
        with open(macro_filepath, 'w') as emffile:
            emffile.write(macro_string)

        return macro_filepath

    def execute(self):
        """Executes the macro file on EES via subprocess module. Returns DataFrame with results."""

        if "EES.exe" in str(subprocess.check_output('tasklist')):
            print(">> Uma instância do EES foi encontrada aberta. Ela será fechada.")
            os.system("taskkill /f /im  EES.exe")

        # Set input datfile and output filename.
        self.handle_inputs()

        # Run EES and execute macro file
        subprocess.run([self.EES_exe, self.paths['macro_path'], '/hide', '/NOSPLASH'])

        return self.get_output()

    def get_output(self):
        """Read utput file created by EES. Returns Dictionary."""

        with open(self.datfiles['output'], 'r') as datfile:
            self.results = self.clean_up_outputs(datfile.read())

        self.save()
        return self.results

    def save(self):
        """Save inputs, outputs as JSON."""
        inputs_filename = f'inputs.json'
        results_filename = f'results.json'

        with open(os.path.join(self.paths['base_folder'], inputs_filename), 'w') as jsonfile:
            json.dump(self.inputs, jsonfile, indent=4)

        with open(os.path.join(self.paths['base_folder'], results_filename), 'w') as jsonfile:
            json.dump(self.results, jsonfile, indent=4)

    def clean_up_outputs(self, str_outputs):
        """Turns string from DAT file in a dict with variable name and value."""

        output_values = str_outputs.strip('\n').split('\t')

        output_dict = {}
        for var, value in zip(self.outputs, output_values):
            output_dict.update({var: float(value)})

        return output_dict
