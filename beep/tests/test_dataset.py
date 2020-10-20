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
"""Unit tests related to feature generation"""

import unittest
import os
import json
import numpy as np
from beep.featurize import (
    RPTdQdVFeatures,
    HPPCResistanceVoltageFeatures,
    DiagnosticSummaryStats,
)
from beep.dataset import BeepDataset
from monty.tempfile import ScratchDir

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")
DIAGNOSTIC_PROCESSED = os.path.join(TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json")
FASTCHARGE_PROCESSED = os.path.join(TEST_FILE_DIR, '2017-06-30_2C-10per_6C_CH10_structure.json')

BIG_FILE_TESTS = os.environ.get("BEEP_BIG_TESTS", False)
SKIP_MSG = "Tests requiring large files with diagnostic cycles are disabled, set BIG_FILE_TESTS to run full tests"
FEATURIZER_CLASSES = [RPTdQdVFeatures, HPPCResistanceVoltageFeatures, DiagnosticSummaryStats]


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_from_features(self):
        dataset = BeepDataset.from_features('test_dataset', ['PreDiag'], FEATURIZER_CLASSES,
                                            feature_dir=os.path.join(TEST_FILE_DIR, 'data-share/features'))
        self.assertEqual(dataset.name, 'test_dataset')
        self.assertEqual(dataset.data.shape, (2, 55))
        self.assertIsNone(dataset.X_test)
        self.assertSetEqual(set(dataset.feature_sets.keys()), {'RPTdQdVFeatures', 'DiagnosticSummaryStats'})
        self.assertEqual(dataset.missing.feature_class.iloc[0], 'HPPCResistanceVoltageFeatures')

    def test_from_processed_cycler_run_list(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            dataset = BeepDataset.from_processed_cycler_runs('test_dataset',
                                                             project_list=None,
                                                             processed_run_list=[DIAGNOSTIC_PROCESSED,
                                                                                 FASTCHARGE_PROCESSED],
                                                             feature_class_list=FEATURIZER_CLASSES,
                                                             processed_dir=TEST_FILE_DIR,
                                                             feature_dir='data-share/features')
            self.assertEqual(dataset.name, 'test_dataset')
            self.assertEqual(dataset.data.shape, (1, 118))
            self.assertIsNone(dataset.X_test)

            self.assertEqual(dataset.missing.shape, (3, 2))
            self.assertEqual(dataset.missing.filename.iloc[0],
                             os.path.split(FASTCHARGE_PROCESSED)[1])

    def test_train_test_split(self):
        dataset = BeepDataset.from_features('test_dataset', ['PreDiag'], FEATURIZER_CLASSES,
                                            feature_dir=os.path.join(TEST_FILE_DIR, 'data-share', 'features'))
        predictors = dataset.feature_sets['RPTdQdVFeatures'][0:3] + \
                     dataset.feature_sets['DiagnosticSummaryStats'][-3:]

        X_train, X_test, y_train, y_test = \
            dataset.generate_train_test_split(predictors=predictors,
                                              outcomes=dataset.feature_sets['RPTdQdVFeatures'][-1],
                                              test_size=0.5, seed=123,
                                              parameters_path=os.path.join(TEST_FILE_DIR,
                                                                           'data-share',
                                                                           'raw',
                                                                           'parameters'))

        self.assertEqual(dataset.data.shape, (2, 55))
        self.assertEqual(dataset.X_test.shape, (1, 6))
        self.assertEqual(dataset.X_train.shape, (1, 6))

        parameter_dict = {'PreDiag_000197':
                              {'project_name': 'PreDiag',
                               'seq_num': 197,
                               'template': 'diagnosticV3.000',
                               'charge_constant_current_1': 0.2,
                               'charge_percent_limit_1': 30,
                               'charge_constant_current_2': 0.2,
                               'charge_cutoff_voltage': 3.7,
                               'charge_constant_voltage_time': 30,
                               'charge_rest_time': 5, 'discharge_constant_current': 0.2,
                               'discharge_cutoff_voltage': 3.5, 'discharge_rest_time': 15,
                               'cell_temperature_nominal': 25, 'cell_type': 'Tesla_Model3_21700',
                               'capacity_nominal': 4.84, 'diagnostic_type': 'HPPC+RPT',
                               'diagnostic_parameter_set': 'Tesla21700',
                               'diagnostic_start_cycle': 30,
                               'diagnostic_interval': 100}
                          }

        self.assertDictEqual(dataset.train_cells_parameter_dict, parameter_dict)
