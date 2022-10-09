import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
import pandas as pd
from ees.parametric import ParametricStudies
from ..others.graphs_dolar import DollarAnalisys

EES_exe = r'C:\Root\Universidade\EES\EES.exe'
# EES_exe = r'C:\Root\Universidade\EES_antigo\EES.exe'
EES_model_libr = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise Economica\Cálculos\trigeracao_LiBrH2O_economic_py.EES'
EES_model_nh3 = r'C:\Root\Drive\Unicamp\[Unicamp]\[Dissertação]\01 - Algoritmo\Analise Economica\Cálculos\trigeracao_NH3H2O_economic_py.EES'
inputs_libr = {
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
    'ir': 0.15
}
inputs_nh3 = {
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
outputs_libr = ['W_net', 'unit_cost_turbine', 'num_of_turbines', 'cost_turbine', 'cost_2019_turbine', 'cost_gerador', 'cost_absorvedor',
                'cost_condensador', 'cost_evaporador', 'cost_hx', 'cost_bomba', 'cost_vs', 'cost_vr', 'cost_2019_gerador', 'cost_sra',
                'cost_2019_absorvedor', 'cost_2019_condensador', 'cost_2019_evaporador', 'cost_2019_hx', 'cost_2019_bomba', 'cost_2019_vs',
                'cost_2019_vr', 'cost_2019_sra', 'A_evaporador', 'A_absorvedor', 'A_condensador', 'A_gerador', 'A_hx', 'W_bomba', 'm_dot[14]',
                'm_dot[19]', 'cost_u', 'cost_d', 'cost_aquecedor', 'cost_fan', 'cost_2019_u', 'cost_2019_d', 'cost_2019_aquecedor', 'cost_2019_fan',
                'm_dot[38]', 'A_aquecedor', 'v_dot[37]', 'cost_hdh', 'cost_2019_hdh', 'cost_CAPEX_2019_trigen', 'cost_CAPEX_2019_real_trigen',
                'cost_2019_turbine_real', 'cost_2019_sra_real', 'cost_2019_hdh_real', 'cost_op_MP', 'cost_op_fuel', 'cost_op_DR',
                'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA', 'cost_OPEX_real_anual', 'cost_prod_energia',
                'cost_prod_agua', 'cost_prod_refrigeracao', 'cost_prod', 'payback_simples_2']

outputs_nh3 = ['W_net', 'unit_cost_turbine', 'num_of_turbines', 'cost_turbine', 'cost_2019_turbine', 'cost_gerador', 'cost_absorvedor',
               'cost_condensador', 'cost_evaporador', 'cost_shx', 'cost_rhx', 'cost_retificador', 'cost_bomba', 'cost_vs', 'cost_vr',
               'cost_2019_gerador', 'cost_sra', 'cost_2019_absorvedor', 'cost_2019_condensador', 'cost_2019_evaporador', 'cost_2019_shx',
               'cost_2019_rhx', 'cost_2019_retificador', 'cost_2019_bomba', 'cost_2019_vs', 'cost_2019_vr', 'cost_2019_sra', 'A_evaporador',
               'A_absorvedor', 'A_condensador', 'A_gerador', 'A_shx', 'A_rhx', 'A_retificador', 'W_bomba', 'm_dot[14]',
               'm_dot[19]', 'cost_u', 'cost_d', 'cost_aquecedor', 'cost_fan', 'cost_2019_u', 'cost_2019_d', 'cost_2019_aquecedor', 'cost_2019_fan',
               'm_dot[38]', 'A_aquecedor', 'v_dot[37]', 'cost_hdh', 'cost_2019_hdh', 'cost_CAPEX_2019_trigen', 'cost_CAPEX_2019_real_trigen',
               'cost_2019_turbine_real', 'cost_2019_sra_real', 'cost_2019_hdh_real', 'cost_op_MP', 'cost_op_fuel', 'cost_op_DR',
               'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA', 'cost_OPEX_real_anual', 'cost_prod_energia',
               'cost_prod_agua', 'cost_prod_refrigeracao', 'cost_prod', 'tpb2']
parametric_inputs = {
    'dolar': np.linspace(3, 6, 20)
}

# eesmodel_libr = ParametricStudies(EES_exe, EES_model_libr, inputs_libr, parametric_inputs, outputs_libr)
# parametric_table = eesmodel_libr.execute()

# eesmodel_nh3 = ParametricStudies(EES_exe, EES_model_nh3, inputs_nh3, parametric_inputs, outputs_nh3)
# parametric_table = eesmodel_nh3.execute()

# for variable, df in parametric_table.items():
#     print(df)

EES_models = {'libr': EES_model_libr, 'nh3': EES_model_nh3}

graph = DollarAnalisys(EES_models)
# graph.generate(lang='en-US')
graph.generate(lang='pt-BR')
# graph.generate_artigo("libr")
