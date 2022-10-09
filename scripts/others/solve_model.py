import os
import time
import sys
import csv
sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
import pandas as pd
from ees.solvemodel import SolveModel
from ees.utilities import cleanup_csv


def libr():
    EES_exe = r'C:\Root\Universidade\EES\EES.exe'
    EES_model = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_LiBrH2O.EES'

    inputs = {
        'm_dot[9]': 0.0226,
        'T[1]': 25,
        'T[3]': 468,
        'T[4]': 763.4,
        'T[9]': 25,
        'eta_compressor': 0.85,
        'eta_turbina': 0.85,
        'rp': 3.22,
        'X_biogas_ch4': 0.6,
        'DeltaTmin': 10,
        'x[18]': 0,
        'Q_evaporador': 12,
        'epsilon_hx': 0.80,
        'eta_bomba': 0.95,
        'T[10]': 35,
        'T[13]': 85,
        'T[19]': 40,
        'T[22]': 5,
        'T[24]': 25,
        'T[25]': 30,
        'T[30]': 16,
        'T[31]': 10,
        'T[32]': 25,
        'T[34]': 80,
        'salinity': 3.535,
        'epsilon_u': 0.85,
        'epsilon_d': 0.85,
        'phi[36]': 0.9,
        'phi[37]': 0.9,
        'MR': 2.5,
        'T_0': 25,
        'P_0': 101.325
    }
    outputs = ['W_compressor', 'W_turbina', 'W_net', 'eta_brayton', 'Q_gerador', 'Q_absorvedor', 'Q_condensador', 'Q_evaporador',
               'UA_gerador', 'UA_absorvedor', 'UA_condensador', 'UA_evaporador', 'COP_1', 'COP_2', 'v_dot[38]', 'v_dot[32]',
               'm_dot[38]', 'm_dot[32]', 'Q_aquecedor', 'UA_aquecedor', 'RR', 'GOR', 'EUF_sys', 'Exd_compressor', 'psi_compressor',
               'Exd_regenerador', 'psi_regenerador', 'Exd_cc', 'psi_cc', 'Exd_turbina', 'psi_turbina', 'Exd_brayton', 'psi_brayton',
               'Exd_absorvedor', 'psi_absorvedor', 'Exd_gerador', 'psi_gerador', 'Exd_condensador', 'psi_condensador', 'Exd_evaporador',
               'psi_evaporador', 'Exd_vs', 'psi_vs', 'Exd_vr', 'psi_vr', 'Exd_hx', 'psi_hx', 'Exd_bomba', 'psi_bomba', 'psi_sra',
               'Exd_sra', 'Exd_umidificador', 'psi_umidificador', 'Exd_desumidificador', 'psi_desumidificador', 'Exd_aquecedor',
               'psi_aquecedor', 'Exd_hdh', 'psi_hdh', 'psi_sys_1', 'psi_sys_2', 'Exd_sys', 'delta_compressor', 'delta_regenerador',
               'delta_cc', 'delta_turbina', 'delta_absorvedor', 'delta_bomba', 'delta_vs', 'delta_vr', 'delta_hx', 'delta_gerador',
               'delta_condensador', 'delta_evaporador', 'delta_umidificador', 'delta_desumidificador', 'delta_aquecedor',
               'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh',
               'psi_partial', 'Exd_partial']

    casos = {
        "v2-caso1": {
            "T[10]": 35.00605725132968,
            "T[19]": 35.009395493143664,
            "T[13]": 84.65362090803661,
            "T[22]": 5.9929432412217265,
            "MR": 1.9869833698980435,
            "T[34]": 68.11813029159323
        },
        "v2-caso2": {
            "T[10]": 35.012024779165586,
            "T[19]": 35.10231867169098,
            "T[13]": 85.01460541391867,
            "T[22]": 5.980626043164863,
            "MR": 4.1054918454862985,
            "T[34]": 99.97410015687473
        },
        "v2-caso3": {
            "T[10]": 35.03834126958878,
            "T[19]": 35.010537539127725,
            "T[13]": 84.21670997033569,
            "T[22]": 5.991448318502584,
            "MR": 1.9840100708233024,
            "T[34]": 68.04435271411491
        }
    }
    for caso, case_vars in casos.items():
        modified_input = {}
        modified_input.update(inputs)
        modified_input.update(case_vars)

        eesmodel = SolveModel(EES_exe, EES_model, modified_input, outputs, runID=caso)
        result = eesmodel.execute()
        df = pd.DataFrame.from_dict(result, orient='index')
        print(caso)
        print(df)
        print(" ")

        inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
        inputs_df = pd.DataFrame.from_dict(modified_input, orient='index')
        inputs_df.to_excel(inputs_filename)

        excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
        df.to_excel(excel_filename)

        csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
        cleanup_csv(csv_filename)

        time.sleep(5)

    # CASO BASE
    eesmodel = SolveModel(EES_exe, EES_model, inputs, outputs, runID="v2-CASO BASE")
    result = eesmodel.execute()
    df = pd.DataFrame.from_dict(result, orient='index')
    print(df)

    inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
    inputs_df = pd.DataFrame.from_dict(inputs, orient='index')
    inputs_df.to_excel(inputs_filename)

    excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
    df.to_excel(excel_filename)

    csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
    cleanup_csv(csv_filename)

    # CASO LIMITE
    # inputs.update(
    #     {
    #         "T[10]": 35,
    #         "T[19]": 35,
    #         "T[13]": 90,
    #         "T[22]": 6,
    #         "MR": 2.6,
    #         "T[34]": 68
    #     }
    # )
    # eesmodel = SolveModel(EES_exe, EES_model, inputs, outputs, runID="Caso Limite")
    # result = eesmodel.execute()
    # df = pd.DataFrame.from_dict(result, orient='index')
    # print(df)

    # inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
    # inputs_df = pd.DataFrame.from_dict(inputs, orient='index')
    # inputs_df.to_excel(inputs_filename)

    # excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
    # df.to_excel(excel_filename)

    # csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
    # cleanup_csv(csv_filename)


def nh3():
    EES_exe = r'C:\Root\Universidade\EES\EES.exe'
    EES_model = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise\trigeracao_NH3H2O.EES'

    inputs = {
        'm_dot[9]': 0.0226,
        'T[1]': 25,
        'T[3]': 468,
        'T[4]': 763.4,
        'T[9]': 25,
        'eta_compressor': 0.85,
        'eta_turbina': 0.85,
        'rp': 3.22,
        'X_biogas_ch4': 0.6,
        'DeltaTmin': 10,
        'x[18]': 0.9996,
        'Q_evaporador': 12,
        'epsilon_hx': 0.80,
        'eta_bomba': 0.95,
        'T[10]': 35,
        'T[13]': 85,
        'T[19]': 40,
        'T[22]': 5,
        'T[24]': 25,
        'T[25]': 30,
        'T[30]': 16,
        'T[31]': 10,
        'T[32]': 25,
        'T[34]': 80,
        'salinity': 3.535,
        'epsilon_u': 0.85,
        'epsilon_d': 0.85,
        'phi[36]': 0.9,
        'phi[37]': 0.9,
        'MR': 2.5,
        'T_0': 25,
        'P_0': 101.325,
        'epsilon_rhx': 0.8,
        'Q[22]': 0.975
    }
    outputs = ['W_compressor', 'W_turbina', 'W_net', 'eta_brayton', 'Q_gerador', 'Q_absorvedor', 'Q_condensador', 'Q_evaporador',
               'UA_gerador', 'UA_absorvedor', 'UA_condensador', 'UA_evaporador', 'COP_1', 'COP_2', 'v_dot[38]', 'v_dot[32]',
               'm_dot[38]', 'm_dot[32]', 'Q_aquecedor', 'UA_aquecedor', 'RR', 'GOR', 'EUF_sys', 'Exd_compressor', 'psi_compressor',
               'Exd_regenerador', 'psi_regenerador', 'Exd_cc', 'psi_cc', 'Exd_turbina', 'psi_turbina', 'Exd_brayton', 'psi_brayton',
               'Exd_absorvedor', 'psi_absorvedor', 'Exd_gerador', 'psi_gerador', 'Exd_condensador', 'psi_condensador', 'Exd_evaporador',
               'psi_evaporador', 'Exd_vs', 'psi_vs', 'Exd_vr', 'psi_vr', 'Exd_hx', 'psi_hx', 'Exd_bomba', 'psi_bomba', 'psi_sra',
               'Exd_sra', 'Exd_umidificador', 'psi_umidificador', 'Exd_desumidificador', 'psi_desumidificador', 'Exd_aquecedor',
               'psi_aquecedor', 'Exd_hdh', 'psi_hdh', 'psi_sys_1', 'psi_sys_2', 'Exd_sys', 'delta_compressor', 'delta_regenerador',
               'delta_cc', 'delta_turbina', 'delta_absorvedor', 'delta_bomba', 'delta_vs', 'delta_vr', 'delta_hx', 'delta_gerador',
               'delta_condensador', 'delta_evaporador', 'delta_umidificador', 'delta_desumidificador', 'delta_aquecedor',
               'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh', 'Exd_retificador',
               'Exd_rhx', 'epsilon_rhx', 'psi_partial', 'Exd_partial']

    casos = {
        "v2-caso1": {
            "T[10]": 35.00503361381352,
            "T[19]": 35.00197668295226,
            "T[13]": 89.96853978405608,
            "T[22]": 5.999320298901701,
            "MR": 1.9838458708739872,
            "T[34]": 68.02894290613834
        },
        "v2-caso2": {
            "T[10]": 35.00292710645969,
            "T[19]": 35.00261818757181,
            "T[13]": 89.97643507841696,
            "T[22]": 5.9994117743926205,
            "MR": 4.087498338770087,
            "T[34]": 99.9742934894558
        },
        "v2-caso3": {
            "T[10]": 35.022784168424366,
            "T[19]": 35.00969985720541,
            "T[13]": 89.81685794790505,
            "T[22]": 5.994923574250652,
            "MR": 1.9840589454954465,
            "T[34]": 68.04985327128358
        },
    }
    for caso, case_vars in casos.items():
        modified_input = {}
        modified_input.update(inputs)
        modified_input.update(case_vars)

        eesmodel = SolveModel(EES_exe, EES_model, modified_input, outputs, runID=caso)
        result = eesmodel.execute()
        df = pd.DataFrame.from_dict(result, orient='index')
        print(caso)
        print(df)
        print(" ")

        inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
        inputs_df = pd.DataFrame.from_dict(modified_input, orient='index')
        inputs_df.to_excel(inputs_filename)

        excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
        df.to_excel(excel_filename)

        csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
        cleanup_csv(csv_filename)

        time.sleep(5)

    # CASO BASE

    eesmodel = SolveModel(EES_exe, EES_model, inputs, outputs, runID="v2-CASO BASE")
    result = eesmodel.execute()
    df = pd.DataFrame.from_dict(result, orient='index')
    print(df)

    inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
    inputs_df = pd.DataFrame.from_dict(inputs, orient='index')
    inputs_df.to_excel(inputs_filename)

    excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
    df.to_excel(excel_filename)

    csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
    cleanup_csv(csv_filename)

    # CASO LIMITE
    # inputs.update(
    #     {
    #         "T[10]": 35,
    #         "T[19]": 35,
    #         "T[13]": 90,
    #         "T[22]": 6,
    #         "MR": 2.6,
    #         "T[34]": 68
    #     }
    # )
    # eesmodel = SolveModel(EES_exe, EES_model, inputs, outputs, runID="Caso Limite")
    # result = eesmodel.execute()
    # df = pd.DataFrame.from_dict(result, orient='index')
    # print(df)

    # inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
    # inputs_df = pd.DataFrame.from_dict(inputs, orient='index')
    # inputs_df.to_excel(inputs_filename)

    # excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
    # df.to_excel(excel_filename)

    # csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
    # cleanup_csv(csv_filename)


if __name__ == "__main__":
    # libr()
    nh3()
