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
import tempfile
from pathlib import Path
from beep.featurize import (
    process_file_list_from_json,
    DeltaQFastCharge,
    TrajectoryFastCharge,
    DegradationPredictor,
    RPTdQdVFeatures,
    HPPCResistanceVoltageFeatures,
    HPPCRelaxationFeatures,
    DiagnosticProperties,
    DiagnosticSummaryStats,
)
from beep.features import featurizer_helpers
from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")
PROCESSED_CYCLER_FILE = "2017-06-30_2C-10per_6C_CH10_structure.json"
PROCESSED_CYCLER_FILE_INSUF = "structure_insufficient.json"
MACCOR_FILE_W_DIAGNOSTICS = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000020_CH71.071")
MACCOR_FILE_W_PARAMETERS = os.path.join(
    TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010"
)

BIG_FILE_TESTS = os.environ.get("BEEP_BIG_TESTS", False)
SKIP_MSG = "Tests requiring large files with diagnostic cycles are disabled, set BIG_FILE_TESTS to run full tests"


class TestFeaturizer(unittest.TestCase):
    def setUp(self):
        pass

    def test_feature_generation_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, PROCESSED_CYCLER_FILE)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = loadfn(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            self.assertEqual(len(featurizer.X), 1)  # just test if works for now
            # Ensure no NaN values
            self.assertFalse(np.any(featurizer.X.isnull()))

    def test_feature_old_class(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, PROCESSED_CYCLER_FILE)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            predictor = DegradationPredictor.from_processed_cycler_run_file(
                processed_cycler_run_path, features_label="full_model"
            )
            self.assertEqual(predictor.feature_labels[4], "charge_time_cycles_1:5")

    def test_feature_label_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, PROCESSED_CYCLER_FILE)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = loadfn(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            self.assertEqual(featurizer.X.columns.tolist()[4], "charge_time_cycles_1:5")

    def test_feature_serialization(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, PROCESSED_CYCLER_FILE)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = loadfn(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            dumpfn(featurizer, featurizer.name)
            features_reloaded = loadfn(featurizer.name)
            self.assertIsInstance(features_reloaded, DeltaQFastCharge)
            # test nominal capacity is being generated
            self.assertEqual(
                features_reloaded.X.loc[0, "nominal_capacity_by_median"],
                1.0628421000000001
            )

    def test_feature_serialization_for_training(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, PROCESSED_CYCLER_FILE)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = loadfn(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            dumpfn(featurizer, featurizer.name)
            features_reloaded = loadfn(featurizer.name)
            self.assertIsInstance(features_reloaded, DeltaQFastCharge)

    def test_feature_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()

            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = DeltaQFastCharge.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            self.assertEqual(
                local_filename,
                "2017-06-30_2C-10per_6C_CH10_features_DeltaQFastCharge.json",
            )
            self.assertEqual(folder, "DeltaQFastCharge")
            dumpfn(featurizer, featurizer.name)

            processed_run_list = []
            processed_result_list = []
            processed_message_list = []
            processed_paths_list = []
            run_id = 1

            featurizer_classes = [DeltaQFastCharge, TrajectoryFastCharge]
            for featurizer_class in featurizer_classes:
                featurizer = featurizer_class.from_run(
                    pcycler_run_loc, os.getcwd(), pcycler_run
                )
                if featurizer:
                    self.assertEqual(featurizer.metadata["channel_id"], 9)
                    self.assertEqual(featurizer.metadata["protocol"], None)
                    self.assertEqual(featurizer.metadata["barcode"], None)
                    dumpfn(featurizer, featurizer.name)
                    processed_paths_list.append(featurizer.name)
                    processed_run_list.append(run_id)
                    processed_result_list.append("success")
                    processed_message_list.append({"comment": "", "error": ""})
                else:
                    processed_paths_list.append(pcycler_run_loc)
                    processed_run_list.append(run_id)
                    processed_result_list.append("incomplete")
                    processed_message_list.append(
                        {
                            "comment": "Insufficient or incorrect data for featurization",
                            "error": "",
                        }
                    )

            self.assertEqual(processed_result_list, ["success", "success"])
            trajectory = loadfn(
                os.path.join(
                    "TrajectoryFastCharge",
                    "2017-06-30_2C-10per_6C_CH10_features_TrajectoryFastCharge.json",
                )
            )
            self.assertEqual(trajectory.X.loc[0, "capacity_0.8"], 161)

    def test_feature_generation_list_to_json(self):
        processed_cycler_run_path = os.path.join(
            TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
        )
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            # os.environ['BEEP_PROCESSING_DIR'] = os.getcwd()

            # Create dummy json obj
            json_obj = {
                "file_list": [processed_cycler_run_path, processed_cycler_run_path],
                "run_list": [0, 1],
            }
            json_string = json.dumps(json_obj)

            newjsonpaths = process_file_list_from_json(
                json_string, processed_dir=os.getcwd()
            )
            reloaded = json.loads(newjsonpaths)

            # Check that at least strings are output
            self.assertIsInstance(reloaded["file_list"][-1], str)

            # Ensure first is correct
            features_reloaded = loadfn(reloaded["file_list"][4])
            self.assertIsInstance(features_reloaded, DeltaQFastCharge)
            self.assertEqual(
                features_reloaded.X.loc[0, "nominal_capacity_by_median"],
                0.07114775279999999,
            )
            features_reloaded = loadfn(reloaded["file_list"][-1])
            self.assertIsInstance(features_reloaded, DiagnosticProperties)
            self.assertListEqual(
                list(features_reloaded.X.iloc[2, :]),
                [141, 0.9859837086597274, 91.17758004259996, 2.578137278917377, 'reset', 'discharge_energy'],
            )

            # Workflow output
            output_file_path = Path(tempfile.gettempdir()) / "results.json"
            self.assertTrue(output_file_path.exists())

            output_data = json.loads(output_file_path.read_text())
            output_json = output_data[0]

            self.assertEqual(reloaded["file_list"][0], output_json["filename"])
            self.assertEqual(os.path.getsize(output_json["filename"]), output_json["size"])
            self.assertEqual(0, output_json["run_id"])
            self.assertEqual("featurizing", output_json["action"])
            self.assertEqual("success", output_json["status"])

    def test_insufficient_data_file(self):
        processed_cycler_run_path = os.path.join(
            TEST_FILE_DIR, PROCESSED_CYCLER_FILE_INSUF
        )
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()

            json_obj = {
                "file_list": [processed_cycler_run_path],
                "run_list": [1],
            }
            json_string = json.dumps(json_obj)

            json_path = process_file_list_from_json(
                json_string, processed_dir=os.getcwd()
            )
            output_obj = json.loads(json_path)
            self.assertEqual(output_obj["result_list"][0], "incomplete")
            self.assertEqual(
                output_obj["message_list"][0]["comment"],
                "Insufficient or incorrect data for featurization",
            )

            # Workflow output
            output_file_path = Path(tempfile.gettempdir()) / "results.json"
            self.assertTrue(output_file_path.exists())

            output_data = json.loads(output_file_path.read_text())
            output_json = output_data[0]

            self.assertEqual(output_obj["file_list"][0], output_json["filename"])
            self.assertEqual(os.path.getsize(output_json["filename"]), output_json["size"])
            self.assertEqual(1, output_json["run_id"])
            self.assertEqual("featurizing", output_json["action"])
            self.assertEqual("incomplete", output_json["status"])

    def test_RPTdQdVFeatures_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            params_dict = {
                "diag_ref": 0,
                "diag_nr": 2,
                "charge_y_n": 1,
                "rpt_type": "rpt_2C",
                "plotting_y_n": 0,
            }
            featurizer = RPTdQdVFeatures.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run, params_dict
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "RPTdQdVFeatures")
            self.assertEqual(featurizer.X.shape[1], 11)
            self.assertEqual(featurizer.metadata["parameters"], params_dict)

    def test_HPPCResistanceVoltageFeatures_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = HPPCResistanceVoltageFeatures.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "HPPCResistanceVoltageFeatures")
            self.assertEqual(featurizer.X.shape[1], 64)
            self.assertListEqual(
                [featurizer.X.columns[0], featurizer.X.columns[-1]],
                ["ohmic_r_d0", "D_8"],
            )

    def test_HPPCRelaxationFeatures_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = HPPCRelaxationFeatures.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            params_dict = {
                "n_soc_windows": 8,
                "soc_list": [90, 80, 70, 60, 50, 40, 30, 20, 10],
                "percentage_list": [50, 80, 99],
                "hppc_list": [0, 1],
            }

            self.assertEqual(folder, "HPPCRelaxationFeatures")
            self.assertEqual(featurizer.X.shape[1], 30)
            self.assertListEqual(
                [featurizer.X.columns[0], featurizer.X.columns[-1]],
                ["var_50%", "SOC10%_degrad99%"],
            )
            self.assertEqual(featurizer.metadata["parameters"], params_dict)

    def test_DiagnosticSummaryStats_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = DiagnosticSummaryStats.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "DiagnosticSummaryStats")
            self.assertEqual(featurizer.X.shape[1], 42)
            self.assertListEqual(
                [featurizer.X.columns[0], featurizer.X.columns[-1]],
                ["var_charging_capacity", "square_discharging_dQdV"],
            )

    def test_DiagnosticProperties_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = DiagnosticProperties.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "DiagnosticProperties")
            self.assertEqual(featurizer.X.shape, (30, 6))
            self.assertListEqual(
                list(featurizer.X.iloc[2, :]),
                [141, 0.9859837086597274, 91.17758004259996, 2.578137278917377, 'reset', 'discharge_energy']
            )

    def test_get_fractional_quantity_remaining_nx(self):
        processed_cycler_run_path = os.path.join(
            TEST_FILE_DIR, "PreDiag_000233_00021F_truncated_structure.json"
        )
        pcycler_run = loadfn(processed_cycler_run_path)
        sum_diag = featurizer_helpers.get_fractional_quantity_remaining_nx(pcycler_run,
                                                                           metric="discharge_energy",
                                                                           diagnostic_cycle_type="hppc")
        self.assertEqual(len(sum_diag.index), 16)
        self.assertEqual(sum_diag.cycle_index.max(), 1507)
        self.assertEqual(np.around(sum_diag.x.iloc[0], 3), np.around(320.629961, 3))
        self.assertEqual(np.around(sum_diag.n.iloc[15], 3), np.around(37.241178, 3))

    def test_generate_dQdV_peak_fits(self):
        processed_cycler_run_path = os.path.join(
            TEST_FILE_DIR, "PreDiag_000304_000153_truncated_structure.json"
        )
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run = loadfn(processed_cycler_run_path)
            peaks_df = featurizer_helpers.generate_dQdV_peak_fits(pcycler_run, 'rpt_0.2C', 0, 1, plotting_y_n=1)
            print(len(peaks_df.columns))
            self.assertEqual(peaks_df.columns.tolist(),
                             ['m0_Amp_rpt_0.2C_1', 'm0_Mu_rpt_0.2C_1', 'm1_Amp_rpt_0.2C_1',
                              'm1_Mu_rpt_0.2C_1', 'm2_Amp_rpt_0.2C_1', 'm2_Mu_rpt_0.2C_1',
                              'trough_height_0_rpt_0.2C_1', 'trough_height_1_rpt_0.2C_1'])

    def test_get_v_diff(self):
        processed_cycler_run_path_1 = os.path.join(
            TEST_FILE_DIR, "Talos_001383_NCR18650618001_CH31_truncated_structure.json"
        )
        processed_cycler_run_path_2 = os.path.join(
            TEST_FILE_DIR, "PreDiag_000304_000153_truncated_structure.json"
        )
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run = loadfn(processed_cycler_run_path_1)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(0.00850563, decimals=8))

            pcycler_run = loadfn(processed_cycler_run_path_2)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(2.664685514952474e-05, decimals=8))
