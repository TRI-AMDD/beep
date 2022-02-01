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
import pandas as pd
from beep.structure.cli import auto_load_processed
from beep.features.intracell_analysis_v2 import IntracellAnalysisV2
from beep.features.intracell_losses_v2 import IntracellCyclesV2, IntracellFeaturesV2

from beep.tests.constants import TEST_FILE_DIR


class IntracellAnalysisV2Test(unittest.TestCase):
    def setUp(self):
        run = os.path.join(TEST_FILE_DIR,
                           'PreDiag_000220_00005E_structure_omit.json')
        self.cell_struct = auto_load_processed(run)

    def test_intracell_wrappers(self):
        ia = IntracellAnalysisV2(
            os.path.join(TEST_FILE_DIR, 'cathode_clean_cc_charge_exptl_aligned.csv'),
            os.path.join(TEST_FILE_DIR, 'anode_secondMeasure_clean_cc_charge_exptl_aligned.csv'),
            cycle_type='rpt_0.2C',
            error_type='dVdQ'
        )

        cycle_index = 3

        loss_dict, profiles_dict = ia.intracell_values_wrapper_ah(cycle_index, self.cell_struct)

        loss_df = pd.DataFrame(loss_dict, index=['rmse_error', 'LLI_opt', 'Q_pe_opt', 'Q_ne_opt', 'x_NE_2',
                                                 'pe_voltage_FC4p2V', 'pe_voltage_FC4p1V', 'pe_voltage_FC4p0V',
                                                 'pe_voltage_FC3p9V', 'pe_voltage_FC3p8V', 'pe_voltage_FC3p7V',
                                                 'pe_voltage_FC3p6V', 'pe_voltage_FC3p5V', 'pe_voltage_FC3p4V',
                                                 'pe_voltage_FC3p3V', 'pe_voltage_FC3p2V', 'pe_voltage_FC3p1V',
                                                 'pe_voltage_FC3p0V', 'pe_voltage_FC2p9V', 'pe_voltage_FC2p8V',
                                                 'pe_voltage_FC2p7V',
                                                 'pe_soc_FC4p2V', 'pe_soc_FC4p1V', 'pe_soc_FC4p0V', 'pe_soc_FC3p9V',
                                                 'pe_soc_FC3p8V', 'pe_soc_FC3p7V',
                                                 'pe_soc_FC3p6V', 'pe_soc_FC3p5V', 'pe_soc_FC3p4V', 'pe_soc_FC3p3V',
                                                 'pe_soc_FC3p2V', 'pe_soc_FC3p1V',
                                                 'pe_soc_FC3p0V', 'pe_soc_FC2p9V', 'pe_soc_FC2p8V', 'pe_soc_FC2p7V',
                                                 'ne_voltage_FC4p2V', 'ne_voltage_FC4p1V', 'ne_voltage_FC4p0V',
                                                 'ne_voltage_FC3p9V', 'ne_voltage_FC3p8V', 'ne_voltage_FC3p7V',
                                                 'ne_voltage_FC3p6V', 'ne_voltage_FC3p5V', 'ne_voltage_FC3p4V',
                                                 'ne_voltage_FC3p3V', 'ne_voltage_FC3p2V', 'ne_voltage_FC3p1V',
                                                 'ne_voltage_FC3p0V', 'ne_voltage_FC2p9V', 'ne_voltage_FC2p8V',
                                                 'ne_voltage_FC2p7V',
                                                 'ne_soc_FC4p2V', 'ne_soc_FC4p1V', 'ne_soc_FC4p0V', 'ne_soc_FC3p9V',
                                                 'ne_soc_FC3p8V', 'ne_soc_FC3p7V',
                                                 'ne_soc_FC3p6V', 'ne_soc_FC3p5V', 'ne_soc_FC3p4V', 'ne_soc_FC3p3V',
                                                 'ne_soc_FC3p2V', 'ne_soc_FC3p1V',
                                                 'ne_soc_FC3p0V', 'ne_soc_FC2p9V', 'ne_soc_FC2p8V', 'ne_soc_FC2p7V',
                                                 'Q_fc', 'Q_pe', 'Q_ne', 'Q_li']).T

        self.assertAlmostEqual(loss_df['Q_pe'].iloc[0], 4.971001250967573, 3)
        self.assertAlmostEqual(loss_df['Q_ne'].iloc[0], 5.077560763314145, 3)
        self.assertAlmostEqual(loss_df['Q_li'].iloc[0], 4.72058639695313, 3)
        self.assertAlmostEqual(loss_df['pe_voltage_FC4p2V'].iloc[0], 4.2569909388206595, 3)
        self.assertAlmostEqual(loss_df['pe_voltage_FC2p7V'].iloc[0], 3.627263943506632, 3)
        self.assertAlmostEqual(loss_df['ne_voltage_FC4p2V'].iloc[0], 0.05688501257876296, 5)
        self.assertAlmostEqual(loss_df['ne_voltage_FC2p7V'].iloc[0], 0.7952915950646048, 5)
        self.assertAlmostEqual(loss_df['pe_soc_FC4p2V'].iloc[0], 0.9646302250803859, 5)
        self.assertAlmostEqual(loss_df['pe_soc_FC2p7V'].iloc[0], 0.05144694533762058, 5)
        self.assertAlmostEqual(loss_df['ne_soc_FC4p2V'].iloc[0], 0.895068205666317, 5)
        self.assertAlmostEqual(loss_df['ne_soc_FC2p7V'].iloc[0], 0.0010493179433368257, 5)


class IntracellFeaturesTestV2(unittest.TestCase):
    def setUp(self):
        run_path = os.path.join(TEST_FILE_DIR, 'PreDiag_000220_00005E_structure_omit.json')
        self.datapath = auto_load_processed(run_path)
        cathode_file = os.path.join(TEST_FILE_DIR, 'cathode_clean_cc_charge_exptl_aligned.csv')
        anode_file = os.path.join(TEST_FILE_DIR, 'anode_secondMeasure_clean_cc_charge_exptl_aligned.csv')

        self.params = {
            'diagnostic_cycle_type': 'rpt_0.2C',
            'step_type': 0,
            "anode_file": anode_file,
            "cathode_file": cathode_file
        }

    def test_IntracellCycles(self):
        featurizer = IntracellCyclesV2(self.datapath, self.params)
        featurizer.create_features()
        X = featurizer.features
        self.assertEqual(X.shape, (2, 73))
        self.assertAlmostEqual(X["Q_li"].iloc[0], 4.743450821877655, 5)
        self.assertAlmostEqual(X["Q_ne"].iloc[1], 5.101834164537508, 3)

    def test_IntracellFeatures(self):
        featurizer = IntracellFeaturesV2(self.datapath, self.params)
        featurizer.create_features()
        X = featurizer.features
        self.assertEqual(X.shape, (1, 146))
        self.assertAlmostEqual(X["diag_0_LLI_opt"].iloc[0], 0.1929531780384086, 5)
        self.assertAlmostEqual(X["diag_1_LLI_opt"].iloc[0], 0.2080220199958258, 3)

    def test_validation(self):
        # Modify datapath_run to be invalid
        mask = self.datapath.diagnostic_summary.cycle_type == "rpt_0.2C"
        self.datapath.diagnostic_summary.loc[mask, "discharge_capacity"] = 0

        featurizer = IntracellFeaturesV2(self.datapath, self.params)
        val, msg = featurizer.validate()
        self.assertFalse(val)
