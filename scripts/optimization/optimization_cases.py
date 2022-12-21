import os
import sys
sys.path.append(os.path.join(os.getcwd()))
from ees.optimization import GAOptimizationStudy, OptGraph
from ees.optimization_param_analysis import OptParamAnalysis, OptParamAnalysisGraphs
from scripts.config import EES_exe


def optimization(EES_model, target_variable, inputs, outputs,
                 decision_variables, base_config, runID):
    """
    Executar um caso de otimização.

    EES_model (str): Path para o arquivo.EES.
    target_variable (dict): Especificar nome da variável como no EES, nome para display
                            e o tipo de problema (max ou min).
    inputs (dict): Nomes das variáveis de entrada do modelo como no EES e respectivos valores.
    outputs (list): Nomes das variáveis de saída desejadas como no EES.
    decision_variables (dict of tuples): Nomes das variáveis de decisão e suas faixas de operação.
    base_config (dict): Parâmetros do Algoritmo Genético.
    runID (str): Identificador da execução. 
    """

    eesopt = GAOptimizationStudy(EES_exe, EES_model, inputs, outputs, runID=runID)
    eesopt.set_decision_variables(decision_variables)
    eesopt.set_target_variable(**target_variable)
    eesopt.execute(base_config)

    graph = OptGraph(EES_model, idx=runID)
    graph.generate(target_variable["target_variable_display"], lang="pt-BR")
    graph.generate(target_variable["target_variable_display"], lang="en-US")


def optimization_param_analysis(EES_model, target_variable, inputs, outputs,
                                decision_variables, base_config, params, runID):
    """
    Executar análise de sensibilidade dos parâmetros do Algoritmo Genético.

    EES_model (str): Path para o arquivo.EES.
    target_variable (dict): Especificar nome da variável como no EES, nome para display
                            e o tipo de problema (max ou min).
    inputs (dict): Nomes das variáveis de entrada do modelo como no EES e respectivos valores.
    outputs (list): Nomes das variáveis de saída desejadas como no EES.
    decision_variables (dict of tuples): Nomes das variáveis de decisão e suas faixas de operação.
    base_config (dict): Parâmetros do configuração inicial Algoritmo Genético.
    params (dict): Parâmetros a serem testados na análise.
    runID (str): Identificador da execução.
    """

    analysis_case = OptParamAnalysis(EES_exe, EES_model, inputs, outputs,
                                     decision_variables, base_config, params, run_ID=runID)
    analysis_case.set_target_variable(**target_variable)
    analysis_case.set_optimizer(GAOptimizationStudy)

    try:
        results = analysis_case.get_result_from_file()
    except FileNotFoundError:
        results = analysis_case.param_analysis()

    analysis_case.compute_best_results()

    # Geração dos Gráficos
    paramgraphs = OptParamAnalysisGraphs(EES_model, runID, results)
    paramgraphs.set_target_variable(**target_variable)
    paramgraphs.generate(lang="pt-BR")
    paramgraphs.generate(lang="en-US")
    paramgraphs.generate_log()


def LiBr_model(execute="opt"):
    """Obtém resultados para o modelo do sistema de trigeração com SRA LiBr/H2O."""

    EES_model = r'C:\Root\Universidade\Mestrado\dissertacao-scripts\models\trigeracao_LiBrH2O.EES'

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
               'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh']

    decision_variables = {
        'T[10]': (35, 44),
        'T[19]': (35, 48),
        'T[13]': (75, 90),
        'T[22]': (1, 6),
        'MR': (0.5, 4.5),
        'T[34]': (68, 100)
    }

    # Otimização dos três casos
    if execute == "opt":
        low = tuple([v[0] for _, v in decision_variables.items()])
        up = tuple([v[1] for _, v in decision_variables.items()])

        cases = {
            "EUF_sys": {
                "runID": "EUF_LiBr",
                "target_variable": {"target_variable": "EUF_sys", "target_variable_display": r"$ EUF_{sys} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.6, 'method': 'cxBlend', 'params': {'alpha': 0.4}},
                    'mutation': {'rate': 0.2, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            },
            "psi_sys_1": {
                "runID": "psi_LiBr",
                "target_variable": {"target_variable": "psi_sys_1", "target_variable_display": r"$ \psi_{sys} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.7, 'method': 'cxSimulatedBinaryBounded', 'params': {'eta': 3, 'low': low, 'up': up}},
                    'mutation': {'rate': 0.15, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            },
            "m_dot[38]": {
                "runID": "m38_LiBr",
                "target_variable": {"target_variable": "m_dot[38]", "target_variable_display": r"$ \dot{m}_{38} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.4, 'method': 'cxBlend', 'params': {'alpha': 0.4}},
                    'mutation': {'rate': 0.15, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            }
        }

        for _, options in cases.items():
            optimization(EES_model, options["target_variable"], inputs, outputs,
                         decision_variables, options["config"], runID=options["runID"])

    elif execute == "param_analysis":
        cases = {
            "EUF_sys": {
                "runID": "param_analysis_EUF_LiBr",
                "target_variable": {"target_variable": "EUF_sys", "target_variable_display": r"$ EUF_{sys} $", "problem": "max"}
            },
            "psi_sys_1": {
                "runID": "param_analysis_psi_LiBr",
                "target_variable": {"target_variable": "psi_sys_1", "target_variable_display": r"$ \psi_{sys} $", "problem": "max"}
            },
            "m_dot[38]": {
                "runID": "param_analysis_m38_LiBr",
                "target_variable": {"target_variable": "m_dot[38]", "target_variable_display": r"$ \dot{m}_{38} $", "problem": "max"}
            }
        }
        optimization_param_analysis_cases(EES_model, inputs, outputs, decision_variables, cases)


def NH3_model(execute="opt"):
    """Obtém resultados para o modelo do sistema de trigeração com SRA NH3/H2O."""

    EES_model = r'C:\Root\Universidade\Mestrado\dissertacao-scripts\models\trigeracao_NH3H2O.EES'

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
               'Exd_rhx', 'epsilon_rhx']

    decision_variables = {
        'T[10]': (35, 42.5),
        'T[19]': (35, 45.5),
        'T[13]': (76.5, 90),
        'T[22]': (1, 6),
        'MR': (0.5, 4.5),
        'T[34]': (68, 100)
    }

    if execute == "opt":
        low = tuple([v[0] for _, v in decision_variables.items()])
        up = tuple([v[1] for _, v in decision_variables.items()])
        cases = {
            "EUF_sys": {
                "runID": "EUF_nh3",
                "target_variable": {"target_variable": "EUF_sys", "target_variable_display": r"$ EUF_{sys} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.6, 'method': 'cxBlend', 'params': {'alpha': 0.4}},
                    'mutation': {'rate': 0.2, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            },
            "psi_sys_1": {
                "runID": "psi_nh3",
                "target_variable": {"target_variable": "psi_sys_1", "target_variable_display": r"$ \psi_{sys} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.7, 'method': 'cxSimulatedBinaryBounded', 'params': {'eta': 3, 'low': low, 'up': up}},
                    'mutation': {'rate': 0.15, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            },
            "m_dot[38]": {
                "runID": "m38_nh3",
                "target_variable": {"target_variable": "m_dot[38]", "target_variable_display": r"$ \dot{m}_{38} $", "problem": "max"},
                "config": {
                    'seed': 5,
                    'population': 10,
                    'crossover': {'rate': 0.4, 'method': 'cxBlend', 'params': {'alpha': 0.4}},
                    'mutation': {'rate': 0.15, 'method': 'mutPolynomialBounded',
                                 'params': {'indpb': 0.15, 'low': low, 'up': up, 'eta': 3}},
                    'selection': {'method': 'selTournament', 'params': {'tournsize': 7}},
                    'max_generation': 5,
                    'cvrg_tolerance': 1e-05,
                    'verbose': True
                }
            }
        }

        optimization_cases(EES_model, inputs, outputs, decision_variables, cases)

    elif execute == "param_analysis":
        cases = {
            "EUF_sys": {
                "runID": "param_analysis_EUF_nh3",
                "target_variable": {"target_variable": "EUF_sys", "target_variable_display": r"$ EUF_{sys} $", "problem": "max"}
            },
            "psi_sys_1": {
                "runID": "param_analysis_psi_nh3",
                "target_variable": {"target_variable": "psi_sys_1", "target_variable_display": r"$ \psi_{sys} $", "problem": "max"}
            },
            "m_dot[38]": {
                "runID": "param_analysis_m38_nh3",
                "target_variable": {"target_variable": "m_dot[38]", "target_variable_display": r"$ \dot{m}_{38} $", "problem": "max"}
            }
        }
        optimization_param_analysis_cases(EES_model, inputs, outputs, decision_variables, cases)


def optimization_cases(EES_model, inputs, outputs, decision_variables, cases):
    """Executar todos os casos de otimização."""

    for _, options in cases.items():
        optimization(EES_model, options["target_variable"], inputs, outputs,
                     decision_variables, options["config"], runID=options["runID"])


def optimization_param_analysis_cases(EES_model, inputs, outputs, decision_variables, cases):
    """Executar as análises de sensibilidade de todos os casos de otimização."""
    low = tuple([v[0] for _, v in decision_variables.items()])
    up = tuple([v[1] for _, v in decision_variables.items()])
    mu = [(x2 - x1) / 5 for x1, x2 in decision_variables.values()]
    int_low = tuple([int(l) for l in low])
    int_up = tuple([int(u) for u in up])

    base_config = {
        'seed': 5,
        'population': 5,
        'crossover': {'rate': 0.5, 'method': 'cxTwoPoint', 'params': {}},
        'mutation': {'rate': 0.10, 'method': 'mutUniformInt', 'params': {'indpb': 0.05, 'low': int_low, 'up': int_up}},
        'selection': {'method': 'selTournament', 'params': {'tournsize': 5}},
        'max_generation': 3,
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
            {'selection': {'method': 'selRoulette', 'params': {}}},
            {'selection': {'method': 'selStochasticUniversalSampling', 'params': {}}},
        ]
    }

    for _, options in cases.items():
        optimization_param_analysis(EES_model, options["target_variable"], inputs, outputs,
                                    decision_variables, base_config, params, runID=options["runID"])


if __name__ == "__main__":
    LiBr_model(execute="opt")
    LiBr_model(execute="param_analysis")
    NH3_model(execute="opt")
    NH3_model(execute="param_analysis")
