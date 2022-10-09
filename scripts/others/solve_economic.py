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
    EES_model = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise Economica\Cálculos\trigeracao_LiBrH2O_economic_py.EES'

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
        'P_0': 101.325,
        'dolar': 5.50,
        'tarifa_energia_eletrica': 0.62,
        'custo_agua': 13,
        'ir': 0.1
    }
    # outputs = ['W_compressor', 'W_turbina', 'W_net', 'eta_brayton', 'Q_gerador', 'Q_absorvedor', 'Q_condensador', 'Q_evaporador',
    #            'UA_gerador', 'UA_absorvedor', 'UA_condensador', 'UA_evaporador', 'COP_1', 'COP_2', 'v_dot[38]', 'v_dot[32]',
    #            'm_dot[38]', 'm_dot[32]', 'Q_aquecedor', 'UA_aquecedor', 'RR', 'GOR', 'EUF_sys', 'Exd_compressor', 'psi_compressor',
    #            'Exd_regenerador', 'psi_regenerador', 'Exd_cc', 'psi_cc', 'Exd_turbina', 'psi_turbina', 'Exd_brayton', 'psi_brayton',
    #            'Exd_absorvedor', 'psi_absorvedor', 'Exd_gerador', 'psi_gerador', 'Exd_condensador', 'psi_condensador', 'Exd_evaporador',
    #            'psi_evaporador', 'Exd_vs', 'psi_vs', 'Exd_vr', 'psi_vr', 'Exd_hx', 'psi_hx', 'Exd_bomba', 'psi_bomba', 'psi_sra',
    #            'Exd_sra', 'Exd_umidificador', 'psi_umidificador', 'Exd_desumidificador', 'psi_desumidificador', 'Exd_aquecedor',
    #            'psi_aquecedor', 'Exd_hdh', 'psi_hdh', 'psi_sys_1', 'psi_sys_2', 'Exd_sys', 'delta_compressor', 'delta_regenerador',
    #            'delta_cc', 'delta_turbina', 'delta_absorvedor', 'delta_bomba', 'delta_vs', 'delta_vr', 'delta_hx', 'delta_gerador',
    #            'delta_condensador', 'delta_evaporador', 'delta_umidificador', 'delta_desumidificador', 'delta_aquecedor',
    #            'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh', 'payback_simples_1',
    #            'payback_simples_2', 'payback_simples_3', 'cost_CAPEX_2019_real_trigen', 'cost_OPEX_real_anual', 'cost_prod',
    #            'cost_CAPEX_2019_real_total', 'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA']

    outputs = ['W_net', 'unit_cost_turbine', 'num_of_turbines', 'cost_turbine', 'cost_2019_turbine', 'cost_gerador', 'cost_absorvedor',
               'cost_condensador', 'cost_evaporador', 'cost_hx', 'cost_bomba', 'cost_vs', 'cost_vr', 'cost_2019_gerador', 'cost_sra',
               'cost_2019_absorvedor', 'cost_2019_condensador', 'cost_2019_evaporador', 'cost_2019_hx', 'cost_2019_bomba', 'cost_2019_vs',
               'cost_2019_vr', 'cost_2019_sra', 'A_evaporador', 'A_absorvedor', 'A_condensador', 'A_gerador', 'A_hx', 'W_bomba', 'm_dot[14]',
               'm_dot[19]', 'cost_u', 'cost_d', 'cost_aquecedor', 'cost_fan', 'cost_2019_u', 'cost_2019_d', 'cost_2019_aquecedor', 'cost_2019_fan',
               'm_dot[38]', 'A_aquecedor', 'v_dot[37]', 'cost_hdh', 'cost_2019_hdh', 'cost_CAPEX_2019_trigen', 'cost_CAPEX_2019_real_trigen',
               'cost_2019_turbine_real', 'cost_2019_sra_real', 'cost_2019_hdh_real', 'cost_op_MP', 'cost_op_fuel', 'cost_op_DR',
               'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA', 'cost_OPEX_real_anual', 'cost_prod_energia',
               'cost_prod_agua', 'cost_prod_refrigeracao', 'cost_prod', 'payback_simples_2']
    casos = {
        "caso_base": {
            'T[10]': 35,
            'T[19]': 40,
            'T[13]': 85,
            'T[22]': 5,
            'MR': 2.5,
            'T[34]': 80
        },
        "corr_caso1": {
            "T[10]": 35.00605725132968,
            "T[19]": 35.009395493143664,
            "T[13]": 84.65362090803661,
            "T[22]": 5.9929432412217265,
            "MR": 1.9869833698980435,
            "T[34]": 68.11813029159323
        },
        "corr_caso2": {
            "T[10]": 35.012024779165586,
            "T[19]": 35.10231867169098,
            "T[13]": 85.01460541391867,
            "T[22]": 5.980626043164863,
            "MR": 4.1054918454862985,
            "T[34]": 99.97410015687473
        },
        "corr_caso3": {
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
        print(eesmodel.paths["base_folder"])
        inputs_filename = os.path.join(eesmodel.paths["base_folder"], "displayed_inputs.xlsx")
        inputs_df = pd.DataFrame.from_dict(modified_input, orient='index')
        inputs_df.to_excel(inputs_filename)

        excel_filename = os.path.join(eesmodel.paths["base_folder"], "outputs.xlsx")
        df.to_excel(excel_filename)

        csv_filename = os.path.join(eesmodel.paths["base_folder"], "ARRAYS.csv")
        cleanup_csv(csv_filename)

        time.sleep(5)

def nh3():
    EES_exe = r'C:\Root\Universidade\EES\EES.exe'
    EES_model = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise Economica\Cálculos\trigeracao_NH3H2O_economic_py.EES'

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
        'Q[22]': 0.975,
        'dolar': 5.50,
        'tarifa_energia_eletrica': 0.62,
        'custo_agua': 13,
        'ir': 0.1
    }
    # outputs = ['W_compressor', 'W_turbina', 'W_net', 'eta_brayton', 'Q_gerador', 'Q_absorvedor', 'Q_condensador', 'Q_evaporador',
    #            'UA_gerador', 'UA_absorvedor', 'UA_condensador', 'UA_evaporador', 'COP_1', 'COP_2', 'v_dot[38]', 'v_dot[32]',
    #            'm_dot[38]', 'm_dot[32]', 'Q_aquecedor', 'UA_aquecedor', 'RR', 'GOR', 'EUF_sys', 'Exd_compressor', 'psi_compressor',
    #            'Exd_regenerador', 'psi_regenerador', 'Exd_cc', 'psi_cc', 'Exd_turbina', 'psi_turbina', 'Exd_brayton', 'psi_brayton',
    #            'Exd_absorvedor', 'psi_absorvedor', 'Exd_gerador', 'psi_gerador', 'Exd_condensador', 'psi_condensador', 'Exd_evaporador',
    #            'psi_evaporador', 'Exd_vs', 'psi_vs', 'Exd_vr', 'psi_vr', 'Exd_hx', 'psi_hx', 'Exd_bomba', 'psi_bomba', 'psi_sra',
    #            'Exd_sra', 'Exd_umidificador', 'psi_umidificador', 'Exd_desumidificador', 'psi_desumidificador', 'Exd_aquecedor',
    #            'psi_aquecedor', 'Exd_hdh', 'psi_hdh', 'psi_sys_1', 'psi_sys_2', 'Exd_sys', 'delta_compressor', 'delta_regenerador',
    #            'delta_cc', 'delta_turbina', 'delta_absorvedor', 'delta_bomba', 'delta_vs', 'delta_vr', 'delta_hx', 'delta_gerador',
    #            'delta_condensador', 'delta_evaporador', 'delta_umidificador', 'delta_desumidificador', 'delta_aquecedor',
    #            'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh', 'Exd_retificador',
    #            'Exd_rhx', 'epsilon_rhx', 'cost_CAPEX_2019_real_trigen',
    #            'cost_OPEX_real_anual', 'cost_prod', 'cost_CAPEX_2019_real_total', 'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS',
    #            'cost_op_DA', 'payback_simples_1', 'payback_simples_2', 'payback_simples_3']
    outputs = ['W_net', 'unit_cost_turbine', 'num_of_turbines', 'cost_turbine', 'cost_2019_turbine', 'cost_gerador', 'cost_absorvedor',
            'cost_condensador', 'cost_evaporador', 'cost_shx', 'cost_rhx', 'cost_retificador', 'cost_bomba', 'cost_vs', 'cost_vr',
            'cost_2019_gerador', 'cost_sra', 'cost_2019_absorvedor', 'cost_2019_condensador', 'cost_2019_evaporador', 'cost_2019_shx',
            'cost_2019_rhx', 'cost_2019_retificador', 'cost_2019_bomba', 'cost_2019_vs', 'cost_2019_vr', 'cost_2019_sra', 'A_evaporador',
            'A_absorvedor', 'A_condensador', 'A_gerador', 'A_shx', 'A_rhx', 'A_retificador', 'W_bomba', 'm_dot[14]',
            'm_dot[19]', 'cost_u', 'cost_d', 'cost_aquecedor', 'cost_fan', 'cost_2019_u', 'cost_2019_d', 'cost_2019_aquecedor', 'cost_2019_fan',
            'm_dot[38]', 'A_aquecedor', 'v_dot[37]', 'cost_hdh', 'cost_2019_hdh', 'cost_CAPEX_2019_trigen', 'cost_CAPEX_2019_real_trigen',
            'cost_2019_turbine_real', 'cost_2019_sra_real', 'cost_2019_hdh_real', 'cost_op_MP', 'cost_op_fuel', 'cost_op_DR',
            'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA', 'cost_OPEX_real_anual', 'cost_prod_energia',
            'cost_prod_agua', 'cost_prod_refrigeracao', 'cost_prod', 'tpb2']

    casos = {
        "caso_base": {
            'T[10]': 35,
            'T[19]': 40,
            'T[13]': 85,
            'T[22]': 5,
            'MR': 2.5,
            'T[34]': 80
        },
        "corr_caso1": {
            "T[10]": 35.00503361381352,
            "T[19]": 35.00197668295226,
            "T[13]": 89.96853978405608,
            "T[22]": 5.999320298901701,
            "MR": 1.9838458708739872,
            "T[34]": 68.02894290613834
        },
        "corr_caso2": {
            "T[10]": 35.00292710645969,
            "T[19]": 35.00261818757181,
            "T[13]": 89.97643507841696,
            "T[22]": 5.9994117743926205,
            "MR": 4.087498338770087,
            "T[34]": 99.9742934894558
        },
        "corr_caso3": {
            "T[10]": 35.022784168424366,
            "T[19]": 35.00969985720541,
            "T[13]": 89.81685794790505,
            "T[22]": 5.994923574250652,
            "MR": 1.9840589454954465,
            "T[34]": 68.04985327128358
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

if __name__ == "__main__":
    libr()
    # nh3()