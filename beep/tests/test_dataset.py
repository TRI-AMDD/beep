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
"""Unit tests related to dataset generation"""

import unittest
import os
import shutil

from monty.tempfile import ScratchDir
from monty.serialization import dumpfn, loadfn


from beep.featurize import (
    HPPCResistanceVoltageFeatures,
    DiagnosticSummaryStats,
    CycleSummaryStats,
    DiagnosticProperties
)
from beep import MODULE_DIR
from beep.dataset import BeepDataset, get_threshold_targets
from beep.tests.constants import TEST_FILE_DIR

DIAGNOSTIC_PROCESSED = os.path.join(TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json")
FASTCHARGE_PROCESSED = os.path.join(TEST_FILE_DIR, '2017-06-30_2C-10per_6C_CH10_structure.json')

FEATURIZER_CLASSES = [HPPCResistanceVoltageFeatures, DiagnosticSummaryStats]
FEATURE_HYPERPARAMS = loadfn(
    os.path.join(MODULE_DIR, "features/feature_hyperparameters.yaml")
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_from_features(self):
        dataset = BeepDataset.from_features('test_dataset', ['PreDiag'], FEATURIZER_CLASSES,
                                            feature_dir=os.path.join(TEST_FILE_DIR, 'data-share/features'))
        self.assertEqual(dataset.name, 'test_dataset')
        self.assertEqual(dataset.data.shape, (2, 44))
        #from pdb import set_trace; set_trace()
        self.assertListEqual(list(dataset.data.seq_num), [196, 197])
        self.assertIsNone(dataset.X_test)
        print(set(dataset.feature_sets.keys()))
        self.assertSetEqual(set(dataset.feature_sets.keys()), {'DiagnosticSummaryStats'})
        self.assertEqual(dataset.missing.feature_class.iloc[0], 'HPPCResistanceVoltageFeatures')

    def test_serialization(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            dataset = BeepDataset.from_features('test_dataset', ['PreDiag'], FEATURIZER_CLASSES,
                                                feature_dir=os.path.join(TEST_FILE_DIR, 'data-share/features'))
            dumpfn(dataset, 'temp_dataset.json')
            dataset = loadfn('temp_dataset.json')
            self.assertEqual(dataset.name, 'test_dataset')
            self.assertEqual(dataset.data.shape, (2, 44))
            # from pdb import set_trace; set_trace()
            self.assertListEqual(list(dataset.data.seq_num), [196, 197])
            self.assertIsNone(dataset.X_test)
            self.assertSetEqual(set(dataset.feature_sets.keys()), {'DiagnosticSummaryStats'})
            self.assertEqual(dataset.missing.feature_class.iloc[0], 'HPPCResistanceVoltageFeatures')
            self.assertIsInstance(dataset.filenames, list)

            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            dataset2 = BeepDataset.from_features('test_dataset', ['PreDiag'], [DiagnosticSummaryStats],
                                                  feature_dir=os.path.join(TEST_FILE_DIR, 'data-share/features'))
            dumpfn(dataset2, "temp_dataset_2.json")
            dataset2 = loadfn('temp_dataset_2.json')
            self.assertEqual(dataset2.missing.columns.to_list(), ["filename", "feature_class"])

    def test_from_processed_cycler_run_list(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            os.makedirs(os.path.join(os.getcwd(), "data-share", "raw", "parameters"))
            parameter_files = os.listdir(os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters"))
            for file in parameter_files:
                shutil.copy(os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters", file),
                            os.path.join(os.getcwd(), "data-share", "raw", "parameters"))
            dataset = BeepDataset.from_processed_cycler_runs('test_dataset',
                                                             project_list=None,
                                                             processed_run_list=[DIAGNOSTIC_PROCESSED,
                                                                                 FASTCHARGE_PROCESSED],
                                                             feature_class_list=FEATURIZER_CLASSES,
                                                             processed_dir=TEST_FILE_DIR,
                                                             feature_dir='data-share/features')
            self.assertEqual(dataset.name, 'test_dataset')
            self.assertEqual(dataset.data.shape, (1, 132))
            self.assertEqual(dataset.data.seq_num.iloc[0], 240)
            self.assertIsNone(dataset.X_test)

            self.assertEqual(dataset.missing.shape, (2, 2))
            self.assertEqual(dataset.missing.filename.iloc[0],
                             os.path.split(FASTCHARGE_PROCESSED)[1])

    def test_dataset_with_custom_feature_hyperparameters(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            os.makedirs(os.path.join(os.getcwd(), "data-share", "raw", "parameters"))
            parameter_files = os.listdir(os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters"))
            for file in parameter_files:
                shutil.copy(os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters", file),
                            os.path.join(os.getcwd(), "data-share", "raw", "parameters"))
            hyperparameter_dict = {'DiagnosticSummaryStats': [
                {'test_time_filter_sec': 1000000, 'cycle_index_filter': 6,
                 'diagnostic_cycle_type': 'rpt_0.2C', 'diag_pos_list': [0, 1]},
                {'test_time_filter_sec': 1000000, 'cycle_index_filter': 6,
                 'diagnostic_cycle_type': 'rpt_1C', 'diag_pos_list': [0, 1]},
                {'test_time_filter_sec': 1000000, 'cycle_index_filter': 6,
                 'diagnostic_cycle_type': 'rpt_2C', 'diag_pos_list': [0, 1]}],
                                   'HPPCResistanceVoltageFeatures': [
                                       FEATURE_HYPERPARAMS['HPPCResistanceVoltageFeatures']],
                                   # 'DiagnosticProperties': [FEATURE_HYPERPARAMS['DiagnosticProperties']]
                                   }
            dataset = BeepDataset.from_processed_cycler_runs('test_dataset',
                                                             project_list=None,
                                                             processed_run_list=[DIAGNOSTIC_PROCESSED,
                                                                                 FASTCHARGE_PROCESSED],
                                                             feature_class_list=FEATURIZER_CLASSES,
                                                             processed_dir=TEST_FILE_DIR,
                                                             hyperparameter_dict=hyperparameter_dict,
                                                             feature_dir='data-share/features')
            self.assertEqual(dataset.name, 'test_dataset')
            self.assertEqual(dataset.data.shape, (1, 240))
            self.assertEqual(dataset.data.seq_num.iloc[0], 240)
            self.assertIsNone(dataset.X_test)

            self.assertEqual(dataset.missing.shape, (4, 2))
            self.assertEqual(dataset.missing.filename.iloc[0],
                             os.path.split(FASTCHARGE_PROCESSED)[1])

    def test_train_test_split(self):
        dataset = BeepDataset.from_features('test_dataset', ['PreDiag'], FEATURIZER_CLASSES,
                                            feature_dir=os.path.join(TEST_FILE_DIR, 'data-share', 'features'))
        predictors = dataset.feature_sets['DiagnosticSummaryStats'][-3:]

        X_train, X_test, y_train, y_test = \
            dataset.generate_train_test_split(predictors=predictors,
                                              outcomes=dataset.feature_sets['DiagnosticSummaryStats'][-1],
                                              test_size=0.5, seed=123,
                                              parameters_path=os.path.join(TEST_FILE_DIR,
                                                                           'data-share',
                                                                           'raw',
                                                                           'parameters'))

        self.assertEqual(dataset.data.shape, (2, 44))
        self.assertEqual(dataset.X_test.shape, (1, 3))
        self.assertEqual(dataset.X_train.shape, (1, 3))

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

    def test_get_threshold_targets(self):
        dataset_diagnostic_properties = loadfn(os.path.join(TEST_FILE_DIR, "diagnostic_properties_test.json"))
        threshold_targets_df = get_threshold_targets(dataset_diagnostic_properties.data,
                                                     cycle_type="rpt_1C")
        self.assertEqual(len(threshold_targets_df), 92)
        print(threshold_targets_df.columns.to_list())
        self.assertEqual(threshold_targets_df.columns.to_list(),
                         ['file',
                          'seq_num',
                          'initial_regular_throughput',
                          'rpt_1Cdischarge_energy0.8_normalized_regular_throughput',
                          'rpt_1Cdischarge_energy0.8_cycle_index',
                          'rpt_1Cdischarge_energy0.8_real_regular_throughput']
                         )
        self.assertEqual(threshold_targets_df[threshold_targets_df['seq_num'] == 154].round(decimals=3).to_dict("list"),
                         {
                             'file': ['PredictionDiagnostics_000154'],
                             'seq_num': [154],
                             'initial_regular_throughput': [489.31],
                             'rpt_1Cdischarge_energy0.8_normalized_regular_throughput': [4.453],
                             'rpt_1Cdischarge_energy0.8_cycle_index': [159.766],
                             'rpt_1Cdischarge_energy0.8_real_regular_throughput': [2178.925]
                          }
                         )
        threshold_targets_df = get_threshold_targets(dataset_diagnostic_properties.data,
                                                     cycle_type="rpt_1C",
                                                     extrapolate_threshold=False)
        self.assertEqual(len(threshold_targets_df), 64)
        self.assertEqual(threshold_targets_df['rpt_1Cdischarge_energy0.8_real_regular_throughput'].round(decimals=3)
                         .median(), 2016.976)
