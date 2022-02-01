# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests related to the fitting functions that extract cell parameters from dis profiles"""

import os
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from beep.structure.cli import auto_load_processed
from beep.features.intracell_analysis import IntracellAnalysis, \
    get_halfcell_voltages
from beep.features.intracell_losses import IntracellCycles, IntracellFeatures

from beep.tests.constants import TEST_FILE_DIR


class IntracellAnalysisTest(unittest.TestCase):
    def setUp(self):
        run = os.path.join(TEST_FILE_DIR,
                           'PreDiag_000220_00005E_structure_omit.json')
        self.cell_struct = auto_load_processed(run)
        self.cathode_file = os.path.join(TEST_FILE_DIR,
                                         'data-share/raw/cell_info/cathode_test.csv')
        self.anode_file = os.path.join(TEST_FILE_DIR,
                                       'data-share/raw/cell_info/anode_test.csv')

    def tearDown(self):
        pass

    @unittest.skip
    def test_process_beep_cycle_data_for_initial_halfcell_analysis_mock(self):
        ia = IntracellAnalysis(
            self.cathode_file,
            self.anode_file,
            cycle_type='rpt_0.2C',
            step_type=0
        )
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(
                self.cell_struct, step_type=0)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile_aligned['Voltage_aligned'].min(),
            2.742084, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile_aligned['Voltage_aligned'].max(),
            4.196994, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['Voltage'].min(), 2.703006, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['Voltage'].max(), 4.196994, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['charge_capacity'].min(), 0.0, 4)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['charge_capacity'].max(), 4.539547,
            5)

    def test_intracell_halfcell_matching_v2_mock(self):
        ia = IntracellAnalysis(
            self.cathode_file,
            self.anode_file,
            cycle_type='rpt_0.2C',
            step_type=0
        )
        #
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(
                self.cell_struct)

        test_opt = np.array([0.999459, -4.1740795, 1.0, 0.1, 0.1])
        (PE_pristine_matched,
         NE_pristine_matched,
         df_real_interped,
         emulated_full_cell_interped) = ia.halfcell_initial_matching_v2(
            test_opt,
            real_cell_initial_charge_profile_aligned,
            ia.pe_pristine,
            ia.ne_1_pristine,
            ia.ne_2_pristine_pos,
            ia.ne_2_pristine_neg)

        self.assertAlmostEqual(PE_pristine_matched['SOC_aligned'].min(),
                               -93.08943089430896, 5)
        self.assertAlmostEqual(PE_pristine_matched['SOC_aligned'].max(),
                               110.16260162601628, 5)
        self.assertAlmostEqual(PE_pristine_matched['Voltage_aligned'].min(),
                               2.865916, 5)
        self.assertAlmostEqual(PE_pristine_matched['Voltage_aligned'].max(),
                               4.299219386998656, 5)

    def test_intracell_get_dq_dv_mock(self):
        ia = IntracellAnalysis(
            self.cathode_file,
            self.anode_file,
            cycle_type='rpt_0.2C',
            step_type=0
        )
        #
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(
                self.cell_struct)

        test_opt = np.array([0.999459, -4.1740795, 1.0, 0.1, 0.1])
        (PE_p_m, NE_p_m, df_real_i,
         em_interped) = ia.halfcell_initial_matching_v2(
            test_opt,
            real_cell_initial_charge_profile_aligned,
            ia.pe_pristine,
            ia.ne_1_pristine,
            ia.ne_2_pristine_pos,
            ia.ne_2_pristine_neg)

        real_cell_candidate_charge_profile_aligned = ia.process_beep_cycle_data_for_candidate_halfcell_analysis(
            self.cell_struct,
            real_cell_initial_charge_profile_aligned,
            real_cell_initial_charge_profile,
            3)

        self.assertAlmostEqual(
            real_cell_candidate_charge_profile_aligned["Voltage_aligned"].min(),
            2.742084, 5)

        (PE_out_centered,
         NE_out_centered,
         dVdQ_over_SOC_real,
         dVdQ_over_SOC_emulated,
         df_real_interped,
         emulated_full_cell_interped) = ia.get_dQdV_over_V_from_degradation_matching(
            test_opt, PE_p_m, NE_p_m,
            ia.ne_2_pristine_pos,
            ia.ne_2_pristine_neg,
            real_cell_candidate_charge_profile_aligned)
        print(PE_out_centered)
        self.assertAlmostEqual(PE_out_centered["Voltage_aligned"].min(),
                               2.86591646, 5)
        self.assertAlmostEqual(NE_out_centered["Voltage_aligned"].min(),
                               0.011226739, 5)
        self.assertAlmostEqual(dVdQ_over_SOC_real["Voltage_aligned"].mean(),
                               3.45, 5)
        self.assertAlmostEqual(dVdQ_over_SOC_emulated["Voltage_aligned"].min(),
                               2.7, 5)
        self.assertAlmostEqual(dVdQ_over_SOC_emulated["Voltage_aligned"].max(),
                               4.2, 5)
        #
        (
            PE_upper_voltage, PE_lower_voltage, PE_upper_SOC, PE_lower_SOC,
            PE_mass,
            NE_upper_voltage, NE_lower_voltage, NE_upper_SOC, NE_lower_SOC,
            NE_mass,
            SOC_upper, SOC_lower, Li_mass) = get_halfcell_voltages(
            PE_out_centered,
            NE_out_centered)
        self.assertAlmostEqual(PE_upper_voltage, 4.28427357, 5)
        self.assertAlmostEqual(PE_lower_voltage, 3.56538024, 5)
        self.assertAlmostEqual(PE_upper_SOC, 99.0605427, 5)
        self.assertAlmostEqual(PE_lower_SOC, 48.851774, 5)
        self.assertAlmostEqual(PE_mass, 202.843024, 5)
        self.assertAlmostEqual(NE_upper_voltage, 0.0828635, 5)
        self.assertAlmostEqual(NE_lower_voltage, 0.86598357, 5)
        self.assertAlmostEqual(NE_upper_SOC, 95.1648351, 5)
        self.assertAlmostEqual(NE_lower_SOC, 42.30769230, 5)
        self.assertAlmostEqual(NE_mass, 192.6796998, 5)
        self.assertAlmostEqual(SOC_upper, 183.363318, 5)
        self.assertAlmostEqual(SOC_lower, 81.518334, 5)
        self.assertAlmostEqual(PE_upper_SOC, 99.06054279, 5)
        self.assertAlmostEqual(PE_lower_SOC, 48.8517745, 5)
        self.assertAlmostEqual(Li_mass, 185.2689422, 5)

    def test_process_beep_cycle_data_for_initial_halfcell_analysis(self):
        ia = IntracellAnalysis(os.path.join(TEST_FILE_DIR,
                                            'cathode_clean_cc_charge_exptl_aligned.csv'),
                               os.path.join(TEST_FILE_DIR,
                                            'anode_secondMeasure_clean_cc_charge_exptl_aligned.csv'),
                               cycle_type='rpt_0.2C',
                               step_type=0
                               )
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(
                self.cell_struct, step_type=0)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile_aligned['Voltage_aligned'].min(),
            2.742084, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile_aligned['Voltage_aligned'].max(),
            4.196994, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['Voltage'].min(), 2.703006, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['Voltage'].max(), 4.196994, 5)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['charge_capacity'].min(), 0.0, 4)
        self.assertAlmostEqual(
            real_cell_initial_charge_profile['charge_capacity'].max(), 4.539547,
            5)

    def test_intracell(self):
        ia = IntracellAnalysis(os.path.join(TEST_FILE_DIR,
                                            'cathode_clean_cc_charge_exptl_aligned.csv'),
                               os.path.join(TEST_FILE_DIR,
                                            'anode_secondMeasure_clean_cc_charge_exptl_aligned.csv'),
                               cycle_type='rpt_0.2C',
                               step_type=0
                               )
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(
                self.cell_struct)

        # Solving initial electrode matching to real full cell
        electrode_matching_bounds = (
            (0.8, 1.2), (-20.0, 20.0), (1, 1), (0.1, 0.1), (0.1, 0.1))
        # (-1,1),(0.01,0.1),(0.01,0.1)
        opt_result_halfcell_initial_matching = differential_evolution(
            ia._get_error_from_halfcell_initial_matching,
            electrode_matching_bounds,
            args=(real_cell_initial_charge_profile_aligned,
                  ia.pe_pristine, ia.ne_1_pristine,
                  ia.ne_2_pristine_pos, ia.ne_2_pristine_neg),
            strategy='best1bin', maxiter=1000,
            popsize=15, tol=0.001, mutation=0.5,
            recombination=0.7, seed=1,
            callback=None, disp=False, polish=True,
            init='latinhypercube', atol=0,
            updating='deferred', workers=-1, constraints=())
        print(opt_result_halfcell_initial_matching)

        self.assertEqual(opt_result_halfcell_initial_matching.success, True)
        self.assertAlmostEqual(opt_result_halfcell_initial_matching.x[0],
                               0.999459, 5)
        self.assertAlmostEqual(opt_result_halfcell_initial_matching.x[1],
                               -4.1740795, 6)

        # test_opt = np.array([0.999459, -4.1740795, 1.0, 0.1, 0.1])
        (PE_pristine_matched,
         NE_pristine_matched,
         df_real_interped,
         emulated_full_cell_interped) = ia.halfcell_initial_matching_v2(
            opt_result_halfcell_initial_matching.x,
            real_cell_initial_charge_profile_aligned,
            ia.pe_pristine,
            ia.ne_1_pristine,
            ia.ne_2_pristine_pos,
            ia.ne_2_pristine_neg)
        print(PE_pristine_matched)
        self.assertAlmostEqual(PE_pristine_matched['SOC_aligned'].min(),
                               -4.675029, 5)
        self.assertAlmostEqual(PE_pristine_matched['SOC_aligned'].max(),
                               109.350057, 5)
        self.assertAlmostEqual(PE_pristine_matched['Voltage_aligned'].min(),
                               2.865916, 5)
        self.assertAlmostEqual(PE_pristine_matched['Voltage_aligned'].max(),
                               4.29917115, 5)

        eol_cycle_index_list = self.cell_struct.diagnostic_summary[
            (self.cell_struct.diagnostic_summary.cycle_type == ia.cycle_type) &
            (
                    self.cell_struct.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()
        #
        # initial bounds (for first cycle); they expand as the cell degrades (see update further down).
        # allow negative degradation, as it may reflect gained capacity from electrolyte wetting or other phenomena
        degradation_bounds = ((-10, 50),  # LLI
                              (-10, 50),  # LAM_PE
                              (-10, 50),  # LAM_NE
                              (1, 1),  # (-1,1) x_NE_2
                              (0.1, 0.1),  # (0.01,0.1)
                              (0.1, 0.1),  # (0.01,0.1)
                              )
        #
        # # initializations before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list):
            real_cell_candidate_charge_profile_aligned = ia.process_beep_cycle_data_for_candidate_halfcell_analysis(
                self.cell_struct,
                real_cell_initial_charge_profile_aligned,
                real_cell_initial_charge_profile,
                cycle_index)

            degradation_optimization_result = differential_evolution(
                ia._get_error_from_degradation_matching,
                degradation_bounds,
                args=(PE_pristine_matched,
                      NE_pristine_matched,
                      ia.ne_2_pristine_pos,
                      ia.ne_2_pristine_neg,
                      real_cell_candidate_charge_profile_aligned
                      ),
                strategy='best1bin', maxiter=100000,
                popsize=15, tol=0.001, mutation=0.5,
                recombination=0.7,
                seed=1,
                callback=None, disp=False, polish=True,
                init='latinhypercube',
                atol=0, updating='deferred', workers=-1,
                constraints=()
            )

            (PE_out_centered,
             NE_out_centered,
             dVdQ_over_SOC_real,
             dVdQ_over_SOC_emulated,
             df_real_interped,
             emulated_full_cell_interped) = ia.get_dQdV_over_V_from_degradation_matching(
                degradation_optimization_result.x,
                PE_pristine_matched,
                NE_pristine_matched,
                ia.ne_2_pristine_pos,
                ia.ne_2_pristine_neg,
                real_cell_candidate_charge_profile_aligned)
            #
            (PE_upper_voltage, PE_lower_voltage, PE_upper_SOC, PE_lower_SOC,
             PE_mass,
             NE_upper_voltage, NE_lower_voltage, NE_upper_SOC, NE_lower_SOC,
             NE_mass,
             SOC_upper, SOC_lower, Li_mass) = get_halfcell_voltages(
                PE_out_centered, NE_out_centered)
            #
            LLI = degradation_optimization_result.x[0]
            LAM_PE = degradation_optimization_result.x[1]
            LAM_NE = degradation_optimization_result.x[2]
            x_NE_2 = degradation_optimization_result.x[3]
            alpha_real = degradation_optimization_result.x[4]
            alpha_emulated = degradation_optimization_result.x[5]

            tmp_dict = {cycle_index: [LLI, LAM_PE, LAM_NE, x_NE_2, alpha_real,
                                      alpha_emulated,
                                      PE_upper_voltage, PE_lower_voltage,
                                      PE_upper_SOC, PE_lower_SOC, PE_mass,
                                      NE_upper_voltage, NE_lower_voltage,
                                      NE_upper_SOC, NE_lower_SOC, NE_mass,
                                      Li_mass
                                      ]
                        }

            dataset_dict_of_cell_degradation_path.update(tmp_dict)
            real_cell_dict_of_profiles.update(
                {cycle_index: real_cell_candidate_charge_profile_aligned})

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['LLI', 'LAM_PE', 'LAM_NE',
                                             'x_NE_2', 'alpha_real',
                                             'alpha_emulated',
                                             'PE_upper_voltage',
                                             'PE_lower_voltage', 'PE_upper_SOC',
                                             'PE_lower_SOC',
                                             'PE_mass', 'NE_upper_voltage',
                                             'NE_lower_voltage', 'NE_upper_SOC',
                                             'NE_lower_SOC', 'NE_mass',
                                             'Li_mass'
                                             ]).T
        # print(degradation_df.iloc[0].to_list())
        self.assertAlmostEqual(degradation_df['LLI'].iloc[0], -0.027076, 5)
        self.assertAlmostEqual(degradation_df['LAM_PE'].iloc[0], 0.06750165, 5)
        self.assertAlmostEqual(degradation_df['LAM_NE'].iloc[0], 0.3055425, 5)
        self.assertAlmostEqual(degradation_df['PE_upper_voltage'].iloc[0],
                               4.25095116, 5)
        self.assertAlmostEqual(degradation_df['PE_lower_voltage'].iloc[0],
                               3.62354913951, 5)
        self.assertAlmostEqual(degradation_df['PE_upper_SOC'].iloc[0],
                               95.7202505, 5)
        self.assertAlmostEqual(degradation_df['PE_mass'].iloc[0], 109.24224079,
                               5)

        self.assertAlmostEqual(degradation_df['NE_upper_voltage'].iloc[0],
                               0.050515360, 5)
        self.assertAlmostEqual(degradation_df['NE_lower_voltage'].iloc[0],
                               0.8564127166, 5)
        self.assertAlmostEqual(degradation_df['NE_upper_SOC'].iloc[0],
                               91.832460, 5)
        self.assertAlmostEqual(degradation_df['NE_mass'].iloc[0], 108.900146, 5)

        self.assertAlmostEqual(degradation_df['Li_mass'].iloc[0], 104.680978, 5)

    def test_intracell_wrappers(self):
        ia = IntracellAnalysis(
            os.path.join(TEST_FILE_DIR, 'data-share', 'raw', 'cell_info',
                         'cathode_test.csv'),
            os.path.join(TEST_FILE_DIR, 'data-share', 'raw', 'cell_info',
                         'anode_test.csv'),
            cycle_type='rpt_0.2C',
            step_type=0
        )

        (cell_init_aligned, cell_init_profile, PE_matched,
         NE_matched) = ia.intracell_wrapper_init(self.cell_struct)

        eol_cycle_index_list = self.cell_struct.diagnostic_summary[
            (self.cell_struct.diagnostic_summary.cycle_type == ia.cycle_type) &
            (
                    self.cell_struct.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()

        # # initializations before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list):
            loss_dict, profiles_dict = ia.intracell_values_wrapper(cycle_index,
                                                                   self.cell_struct,
                                                                   cell_init_aligned,
                                                                   cell_init_profile,
                                                                   PE_matched,
                                                                   NE_matched,
                                                                   )
            dataset_dict_of_cell_degradation_path.update(loss_dict)
            real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['LLI', 'LAM_PE', 'LAM_NE',
                                             'x_NE_2', 'alpha_real',
                                             'alpha_emulated',
                                             'PE_upper_voltage',
                                             'PE_lower_voltage', 'PE_upper_SOC',
                                             'PE_lower_SOC',
                                             'PE_mass', 'NE_upper_voltage',
                                             'NE_lower_voltage', 'NE_upper_SOC',
                                             'NE_lower_SOC', 'NE_mass',
                                             'Li_mass'
                                             ]).T
        print(degradation_df['LLI'])
        print(degradation_df['LAM_PE'])
        print(degradation_df['Li_mass'])
        self.assertAlmostEqual(degradation_df['LLI'].iloc[0], -9.999983, 5)
        self.assertAlmostEqual(degradation_df['LLI'].iloc[1], -9.999556, 5)
        self.assertAlmostEqual(degradation_df['LAM_PE'].iloc[0], 49.984768, 5)
        self.assertAlmostEqual(degradation_df['LAM_PE'].iloc[1], 49.984877, 5)
        self.assertAlmostEqual(degradation_df["Li_mass"].iloc[1], 12.312480, 3)

        # Values for real anode and cathode measurements
        # self.assertAlmostEqual(degradation_df['LLI'].iloc[0], -0.027076, 5)
        # self.assertAlmostEqual(degradation_df['LLI'].iloc[1], 2.712016, 5)
        # self.assertAlmostEqual(degradation_df['LAM_PE'].iloc[0], 0.06750165, 5)
        # self.assertAlmostEqual(degradation_df['LAM_PE'].iloc[1], 2.745720, 5)
        # self.assertAlmostEqual(degradation_df["Li_mass"].iloc[1], 101.82022, 3)


class IntracellFeaturesTest(unittest.TestCase):
    def setUp(self):
        run_path = os.path.join(TEST_FILE_DIR,
                                'PreDiag_000220_00005E_structure_omit.json')
        self.datapath = auto_load_processed(run_path)
        cathode_file = os.path.join(TEST_FILE_DIR,
                                    'data-share/raw/cell_info/cathode_test.csv')
        anode_file = os.path.join(TEST_FILE_DIR,
                                  'data-share/raw/cell_info/anode_test.csv')
        self.params = {
            'diagnostic_cycle_type': 'rpt_0.2C',
            'step_type': 0,
            "anode_file": anode_file,
            "cathode_file": cathode_file
        }

    def test_IntracellCycles(self):
        featurizer = IntracellCycles(self.datapath, self.params)
        featurizer.create_features()
        X = featurizer.features
        self.assertEqual(X.shape, (2, 17))
        self.assertAlmostEqual(X["LLI"].iloc[0], -9.999983, 5)
        self.assertAlmostEqual(X["Li_mass"].iloc[1], 12.312480, 3)

    def test_IntracellFeatures(self):
        featurizer = IntracellFeatures(self.datapath, self.params)
        featurizer.create_features()
        X = featurizer.features
        self.assertEqual(X.shape, (1, 34))
        self.assertAlmostEqual(X["diag_0_LLI"].iloc[0], -9.999983, 5)
        self.assertAlmostEqual(X["diag_1_Li_mass"].iloc[0], 12.312480, 3)

    def test_validation(self):
        # Modify datapath_run to be invalid
        mask = self.datapath.diagnostic_summary.cycle_type == "rpt_0.2C"
        self.datapath.diagnostic_summary.loc[mask, "discharge_capacity"] = 3.35

        featurizer = IntracellFeatures(self.datapath, self.params)
        val, msg = featurizer.validate()
        self.assertFalse(val)
