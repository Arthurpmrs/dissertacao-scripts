import os
import sys
import numpy as np
sys.path.append(os.path.join(os.getcwd()))
from ees.parametric import ParametricStudies
from scripts.config import EES_exe


def LiBr_model():
    """Efetua a análise paramétrica do sistema de trigeração com SRA LiBr/H2O."""

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
               'EUF_sys_turbina', 'EUF_sys_sra', 'EUF_sys_hdh', 'psi_sys_turbina', 'psi_sys_sra', 'psi_sys_hdh',
               'psi_partial', 'Exd_partial']

    parametric_inputs = {
        'm_dot[9]': np.linspace(0.005, 0.035, 8),
        'X_biogas_ch4': np.linspace(0.4, 0.99, 8),
        'T[10]': np.linspace(35, 44, 8),
        'T[13]': np.linspace(75, 90, 8),
        'T[19]': np.linspace(35, 48, 8),
        'T[22]': np.linspace(1, 6, 8),
        'MR': np.linspace(0.5, 4.5, 8),
        'T[34]': np.linspace(68, 99, 8),
    }

    eesmodel = ParametricStudies(EES_exe, EES_model, inputs,
                                 parametric_inputs, outputs, run_id="param_analysis_v4")
    eesmodel.execute()

    hdh_effectiveness_analysis(EES_model, inputs, outputs)


def NH3_model():
    """Efetua a análise paramétrica do sistema de trigeração com SRA NH3/H2O."""

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
               'Exd_rhx', 'epsilon_rhx', 'psi_partial', 'Exd_partial']

    parametric_inputs = {
        'm_dot[9]': np.linspace(0.005, 0.035, 8),
        'X_biogas_ch4': np.linspace(0.4, 0.99, 8),
        'T[10]': np.linspace(35, 42.5, 8),
        'T[13]': np.linspace(75, 90, 8),
        'T[19]': np.linspace(35, 45.5, 8),
        'T[22]': np.linspace(1, 6, 8),
        'MR': np.linspace(0.5, 4.5, 8),
        'T[34]': np.linspace(68, 99.9, 8)
    }

    eesmodel = ParametricStudies(EES_exe, EES_model, inputs,
                                 parametric_inputs, outputs, run_id="param_analysis_v1")
    eesmodel.execute()

    hdh_effectiveness_analysis(EES_model, inputs, outputs)


def hdh_effectiveness_analysis(EES_model, inputs, outputs):
    """Efetua a análise paramétrica das efetividades do HDH em conjunto."""

    humidifier_effectiveness = [0.5, 0.6, 0.7, 0.8, 0.9]
    parametric_inputs = {
        'epsilon_d': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.90]
    }

    for e_h in humidifier_effectiveness:
        inputs.update({"epsilon_u": e_h})
        hdh_eff_analysis = ParametricStudies(EES_exe, EES_model, inputs,
                                             parametric_inputs, outputs, run_id=f"hdh_e_u_{e_h}")
        hdh_eff_analysis.execute()
        del hdh_eff_analysis


if __name__ == "__main__":
    LiBr_model()
    NH3_model()
