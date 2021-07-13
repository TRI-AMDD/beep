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
"""Unit tests related to the capacity fade degradation behavior classification"""

import os
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from beep.structure.cli import auto_load_processed
from beep.features.knee_identification import bacon_watts_heaviside, get_error_bacon_watts_heaviside, get_knee_value

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class KneeIdentificationTest(unittest.TestCase):
    def setUp(self):
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        file_name = 'PreDiag_000220_00005E_structure_omit.json'
        run = os.path.join(TEST_FILE_DIR, file_name)
        self.cell_struct = auto_load_processed(run)
        self.seq_num = self.cell_struct.paths['raw'][-14:-11]

    def tearDown(self):
        pass
    
    def test_load_capacity_fade_from_run(self):
        
        
    def test_bacon_watts_heaviside(self):
        ia = IntracellAnalysis('cathode_test.csv',
                               'anode_test.csv',
                               cycle_type='rpt_0.2C',
                               step_type=0
                               )
        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            ia.process_beep_cycle_data_for_initial_halfcell_analysis(self.cell_struct, step_type=0)
        self.assertAlmostEqual(real_cell_initial_charge_profile_aligned['Voltage_aligned'].min(), 2.742084, 5)
        self.assertAlmostEqual(real_cell_initial_charge_profile_aligned['Voltage_aligned'].max(), 4.196994, 5)
        self.assertAlmostEqual(real_cell_initial_charge_profile['Voltage'].min(), 2.703006, 5)
        self.assertAlmostEqual(real_cell_initial_charge_profile['Voltage'].max(), 4.196994, 5)
        self.assertAlmostEqual(real_cell_initial_charge_profile['charge_capacity'].min(), 0.0, 4)
        self.assertAlmostEqual(real_cell_initial_charge_profile['charge_capacity'].max(), 4.539547, 5)
