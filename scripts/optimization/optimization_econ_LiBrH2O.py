import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import json
import logging
import datetime
from rich import print
from icecream import ic
from ees.optimization import OptimizationStudy
from ees.optimization_ga import GAOptimizationStudy
from ees.optimization_graphs import OptGraph
from ees.utilities import get_base_folder, add_folder
from ees.optimization_param_analysis import OptParamAnalysis
from graphs_default_param_analysis import DefaultParamAnalysisGraph
from ees.utilities import get_base_folder


def optimization(EES_exe, EES_model, target_variable, inputs, outputs,
                 decision_variables, base_config, runID):
    """Run one optimization case."""

    eesopt = GAOptimizationStudy(EES_exe, EES_model, inputs, outputs, runID=runID)
    eesopt.set_decision_variables(decision_variables)
    eesopt.set_target_variable(**target_variable)
    eesopt.execute(base_config)
    graph = OptGraph(get_base_folder(EES_model), idx=runID)
    graph.generate(target_variable["target_variable_display"], lang="pt-BR")
    graph.generate(target_variable["target_variable_display"], lang="en-US")


def param_analysis(EES_exe, EES_model, target_variable, inputs, outputs,
                   decision_variables, base_config, params, runID):
    """Run parametric analysis."""

    paramAnalysis = OptParamAnalysis(EES_exe, EES_model, inputs, outputs,
                                     decision_variables, base_config, params, run_ID=runID)
    paramAnalysis.set_target_variable(**target_variable)
    paramAnalysis.set_optimizer(GAOptimizationStudy)
    results = paramAnalysis.param_analysis()
    # results = paramAnalysis.get_result_from_file()
    paramAnalysis.compute_best_results()

    # Geração dos Gráficos
    paramgraphs = DefaultParamAnalysisGraph(EES_model, runID, results)
    paramgraphs.set_target_variable(**target_variable)
    paramgraphs.generate(lang="pt-BR")
    paramgraphs.generate(lang="en-US")
    paramgraphs.generate_log()


def main():
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
        'ir': 0.15
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
               'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh', 'payback_simples_1',
               'payback_simples_2', 'payback_simples_3', 'cost_CAPEX_2019_real_trigen', 'cost_OPEX_real_anual', 'cost_prod',
               'cost_CAPEX_2019_real_total', 'cost_op_MR', 'cost_op_PR', 'cost_op_D', 'cost_op_IS', 'cost_op_DA']

    decision_variables = {
        'T[10]': (35, 44),
        'T[19]': (35, 48),
        'T[13]': (75, 90),
        'T[22]': (1, 6),
        'MR': (0.5, 4.5),
        'T[34]': (68, 100)
    }

    low = tuple([v[0] for _, v in decision_variables.items()])
    up = tuple([v[1] for _, v in decision_variables.items()])

    mu = [(x2 - x1) / 5 for x1, x2 in decision_variables.values()]

    int_low = tuple([int(l) for l in low])
    int_up = tuple([int(u) for u in up])

    base_config = {
        'seed': 5,
        'population': 50,
        'crossover': {'rate': 0.5, 'method': 'cxTwoPoint', 'params': {}},
        'mutation': {'rate': 0.10, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}},
        'selection': {'method': 'selTournament', 'params': {'tournsize': 5}},
        'max_generation': 35,
        'cvrg_tolerance': 1e-5,
        'verbose': True
    }
    best_config_1 = {
        'seed': 5,
        'population': 50,
        'crossover': {'rate': 0.5, 'method': 'cxTwoPoint', 'params': {}},
        'mutation': {'rate': 0.10, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}},
        'selection': {'method': 'selTournament', 'params': {'tournsize': 5}},
        'max_generation': 150,
        'cvrg_tolerance': 1e-5,
        'verbose': True
    }

    params = {
        "population": [
            {'population': 10},
            {'population': 15},
            {'population': 25},
            {'population': 50},
            {'population': 100},
            {'population': 150},
            {'population': 200},
        ],
        "crossover_rates": [
            {'crossover': {'rate': 0.2, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.3, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.4, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.5, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.6, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.7, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.8, 'method': 'cxTwoPoint', 'params': {}}}
        ],
        "crossover_methods": [
            {'crossover': {'rate': 0.5, 'method': 'cxTwoPoint', 'params': {}}},
            {'crossover': {'rate': 0.5, 'method': 'cxSimulatedBinaryBounded', 'params': {'eta': 3, 'low': low, 'up': up}}},
            {'crossover': {'rate': 0.5, 'method': 'cxBlend', 'params': {'alpha': 0.4}}}
        ],
        "mutation_rates": [
            {'mutation': {'rate': 0.05, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}},
            {'mutation': {'rate': 0.10, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}},
            {'mutation': {'rate': 0.15, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}},
            {'mutation': {'rate': 0.20, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}},
            {'mutation': {'rate': 0.25, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}}
        ],
        "mutation_methods": [
            {'mutation': {'rate': 0.10, 'method': 'mutGaussian', 'params': {'indpb': 0.05, 'mu': mu, 'sigma': 0.15}}},
            {'mutation': {'rate': 0.10, 'method': 'mutPolynomialBounded', 'params': {'indpb': 0.05, 'low': low, 'up': up, 'eta': 3}}},
            {'mutation': {'rate': 0.10, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}}},
        ],
        "selection_methods": [
            {'selection': {'method': 'selTournament', 'params': {'tournsize': 7}}},
            {'selection': {'method': 'selStochasticUniversalSampling', 'params': {}}},
        ]
    }

    # Otimizações específicas
    target_variable = {"target_variable": "payback_simples_2", "target_variable_display": r"$ T_{PB} $", "problem": "min"}
    optimization(EES_exe, EES_model, target_variable, inputs, outputs, decision_variables, base_config, runID="TPB_librh2o")

    # Análise de sensibilidade
    # target_variable = {"target_variable": "payback_simples_2", "target_variable_display": r"$ T_{PB} $", "problem": "min"}
    # param_analysis(EES_exe, EES_model, target_variable, inputs, outputs,
    #                decision_variables, base_config, params, runID="analise_TPB_librh2o_correcao")


if __name__ == "__main__":
    main()
