import os
import time
import pandas as pd
import subprocess
from icecream import ic
from .utilities import check_model_path


class ParametricStudy:

    def __init__(self, paths, base_case_inputs, variable, parametric_inputs, outputs, run_id=None):
        self.run_id = run_id
        self.variable = variable
        self.paths = self.set_paths(paths)
        self.base_case_inputs = base_case_inputs
        self.parametric_inputs = parametric_inputs
        self.datfiles = {'inputs': [], 'outputs': []}
        self.outputs = outputs

    def set_paths(self, paths):
        """Set paths that will be used as a dictionary and creats them if not already."""

        new_paths = {}

        # Add to super Paths variable.
        results_folder = os.path.join(paths['base_folder'], self.run_id, '.results', self.variable)
        datfiles_folder = os.path.join(paths['base_folder'], self.run_id, '.datfiles', self.variable)
        plots_folder = os.path.join(paths['base_folder'], self.run_id, '.plots', self.variable)

        for name, path in paths.items():
            new_paths.update({name: path})

        new_paths.update({
            'results': results_folder,
            'datfiles': datfiles_folder,
            'plots': plots_folder
        })

        # Check if folders exists and if not, creates them.
        for _, path in new_paths.items():
            if not os.path.exists(path):
                os.makedirs(path)
        return new_paths

    def handle_inputs(self, macro_string):
        """Creation of .DAT files that will be imported by EES.

        Receives Macro string and adds to it.
        Creates list of output files.

        Returns:
            Macro String
        Sets:
            List of output files
        """
        input_strings = self.prepare_input_strings()
        output_string = self.prepare_output_string()

        for i, input_string in enumerate(input_strings):
            input_filepath = self.create_input_datfile(input_string, i)
            output_filepath = self.store_output_datfile(output_string, i)

            macro_string = self.update_macro_string(
                macro_string,
                input_filepath,
                output_filepath,
            )

        return macro_string

    def prepare_input_strings(self):
        """Creates a List of string inputs for DAT files."""

        inputs = []
        for value in self.parametric_inputs:
            mod_input = dict(self.base_case_inputs)
            mod_input.update({self.variable: value})
            inputs.append(' '.join([str(var) for var in mod_input.values()]))

        return inputs

    def prepare_output_string(self):
        """Joins output variables together (Necessery for EES)."""

        return ' '.join(self.outputs)

    def create_input_datfile(self, input_string, i):
        """Creates input datfile and stores path in self.datfiles['inputs'] for each parametric value."""

        filepath = os.path.join(self.paths['datfiles'], f'input_{i + 1}.dat')

        with open(filepath, 'w') as datfile:
            datfile.write(input_string)

        self.datfiles['inputs'].append(filepath)
        return filepath

    def store_output_datfile(self, output, i):
        """Stores output file path in self.datfiles['outputs']"""

        filepath = os.path.join(self.paths['datfiles'], f'OUTPUT_{i + 1}.DAT')
        self.datfiles['outputs'].append(filepath)
        return filepath

    def update_macro_string(self, macro_string, input_filepath, output_filepath):
        """Updates macro string with EES commands to read from DATFILE, solve and export to DATFILE."""

        input_names_string = ' '.join(self.base_case_inputs.keys())
        output_names_string = ' '.join(self.outputs)

        macro_string += f'Import \'{input_filepath}\' {input_names_string}\n'
        macro_string += 'Solve\n'
        macro_string += f'Export \'{output_filepath}\' {output_names_string}\n'

        return macro_string

    def get_outputs(self):
        """Read the output files created by EES. Returns Pandas DataFrame."""

        df = pd.DataFrame({})
        for output_path in self.datfiles['outputs']:
            with open(output_path, 'r') as datfile:
                df = df.append(self.clean_up_outputs(datfile.read()), ignore_index=True)

        df[self.variable] = pd.Series(self.parametric_inputs).values
        self.results = df
        self.save()
        return df

    def save(self):
        """Save outputs as JSON and CSV."""
        filename_json = f'parametric_result.json'
        filename_ijson = f'indented_parametric_result.json'
        filename_csv = f'parametric_result.csv'
        self.results.to_csv(os.path.join(self.paths['results'], filename_csv), sep=';')
        self.results.to_json(os.path.join(self.paths['results'], filename_ijson), orient="columns", indent=4)
        self.results.to_json(os.path.join(self.paths['results'], filename_json), orient="columns")

    def graphs(self, params):
        """Plots/saves graphs for each parameter parsed. Not customizable."""

    def clean_up_outputs(self, str_outputs):
        """Turns string from DAT file in a dict with variable name and value."""

        output_values = str_outputs.strip('\n').split('\t')

        output_dict = {}
        for var, value in zip(self.outputs, output_values):
            output_dict.update({var: float(value)})

        return output_dict


class ParametricStudies:

    def __init__(self, EES_exe, EES_model, base_case_inputs, parametric_inputs, outputs, run_id=None):
        self.EES_exe = EES_exe
        self.EES_model = check_model_path(EES_model)
        self.paths = self.set_paths(self.EES_model)
        self.base_case_inputs = base_case_inputs
        self.parametric_inputs = parametric_inputs
        self.variables = parametric_inputs.keys()
        self.outputs = outputs
        self.parametric_studies = {}
        self.macro_string = ''
        self.results = {}
        self.run_id = str(run_id) if run_id else str(round(time.time()))

    def set_paths(self, EES_model):
        """Set paths that will be used as a dictionary and creats them if not already."""

        model_folder = os.path.dirname(EES_model)
        model_filename = os.path.basename(EES_model)
        base_folder = os.path.join(
            model_folder,
            '.'.join(model_filename.split('.')[:-1]),
            '.ParamAnalysis'
        )
        paths = {
            'model_path': EES_model,
            'model_folder': model_folder,
            'base_folder': base_folder,
        }
        return paths

    def execute(self):
        """Executes the macro file on EES via subprocess module. Returns DataFrame with results."""

        # Initialize instances of ParametricStudy class and update macro string
        self.initialize()

        # Creates macro file (.emf)
        macro_filepath = self.setup_macro()

        # Run EES and execute macro file
        subprocess.run([self.EES_exe, macro_filepath, '/hide', '/NOSPLASH'])

        return self.get_output()

    def initialize(self):
        """Initialize instances for each parametric study that will be done."""

        for variable, parametric_input in self.parametric_inputs.items():
            self.parametric_studies.update({
                variable: ParametricStudy(
                    self.paths,
                    self.base_case_inputs,
                    variable,
                    parametric_input,
                    self.outputs,
                    self.run_id
                )
            })
            self.macro_string = self.parametric_studies[variable].handle_inputs(self.macro_string)

    def setup_macro(self):
        """Adds necessary header and Footer to macro string and create macro file."""

        macro_header = "//WINDOWSIZE 0 401 1496 317\n"
        macro_header += f'Open \'{self.EES_model}\'\n'
        macro_header += 'Units SI C kPa kJ Mass\n'
        self.macro_string = macro_header + self.macro_string
        self.macro_string += 'Quit'

        macro_filepath = os.path.join(self.paths['base_folder'], self.run_id, 'macro.emf')
        with open(macro_filepath, 'w') as emffile:
            emffile.write(self.macro_string)

        return macro_filepath

    def get_output(self):
        """
        Run get_outputs method from ParametricStudy for each variable studied. 
        Returns Dictionary of Dataframes.
        """
        for variable, studie in self.parametric_studies.items():
            self.results.update({variable: studie.get_outputs()})
        return self.results
