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
    CycleSummaryStats
)
from beep.structure import RawCyclerRun
from beep.features import featurizer_helpers
from beep.utils import parameters_lookup
from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir
from beep.utils.s3 import download_s3_object

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")
PROCESSED_CYCLER_FILE = "2017-06-30_2C-10per_6C_CH10_structure.json"
PROCESSED_CYCLER_FILE_INSUF = "structure_insufficient.json"
MACCOR_FILE_W_DIAGNOSTICS = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000020_CH71.071")
MACCOR_FILE_W_PARAMETERS = os.path.join(
    TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010"
)

BIG_FILE_TESTS = os.environ.get("BIG_FILE_TESTS", None) == "True"
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
                [141, 0.9859837086597274, 7.885284043, 4.323121513988055,
                 21.12108276469096, 30, 100, 'reset', 'discharge_energy'],
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

    def test_get_hppc_ocv(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
        )
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = loadfn(pcycler_run_loc)
        hppc_ocv_features = featurizer_helpers.get_hppc_ocv(pcycler_run, 1)
        self.assertEqual(np.round(hppc_ocv_features['variance of ocv'].iloc[0], 6), 0.000016)

    def test_get_step_index(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
        )

        parameters_path = os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters")
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = loadfn(pcycler_run_loc)
        data = pcycler_run.diagnostic_interpolated
        hppc_cycles = data.loc[data.cycle_type == "hppc"]
        print(hppc_cycles.step_index.unique())
        _, protocol_name = os.path.split(pcycler_run.protocol)
        parameter_row, _ = parameters_lookup.get_protocol_parameters(protocol_name, parameters_path=parameters_path)

        for cycle in hppc_cycles.cycle_index.unique():
            hppc_cycle = hppc_cycles[hppc_cycles.cycle_index == cycle]
            for step in hppc_cycle.step_index.unique():
                hppc_cycle_step = hppc_cycle[(hppc_cycle.step_index == step)]
                for step_iter in hppc_cycle_step.step_index_counter.unique():
                    hppc_cycle_step_iter = hppc_cycle_step[(hppc_cycle_step.step_index_counter == step_iter)]
                    duration = hppc_cycle_step_iter.test_time.max() - hppc_cycle_step_iter.test_time.min()
                    median_crate = np.round(hppc_cycle_step.current.median() /
                                            parameter_row["capacity_nominal"].iloc[0], 2)
                    print(step, median_crate, duration)

        step_ind = featurizer_helpers.get_step_index(pcycler_run,
                                                     cycle_type="hppc",
                                                     diag_pos=0)
        self.assertEqual(len(step_ind.values()), 6)
        print([step_ind["hppc_long_rest"],
               step_ind["hppc_discharge_pulse"],
               step_ind["hppc_short_rest"],
               step_ind["hppc_charge_pulse"],
               step_ind["hppc_discharge_to_next_soc"]])

        self.assertEqual(step_ind, {
            'hppc_charge_to_soc': 9,
            'hppc_long_rest': 11,
            'hppc_discharge_pulse': 12,
            'hppc_short_rest': 13,
            'hppc_charge_pulse': 14,
            'hppc_discharge_to_next_soc': 15
        })
        step_ind = featurizer_helpers.get_step_index(pcycler_run,
                                                     cycle_type="hppc",
                                                     diag_pos=1)
        self.assertEqual(len(step_ind.values()), 6)
        self.assertEqual(step_ind, {
            'hppc_charge_to_soc': 41,
            'hppc_long_rest': 43,
            'hppc_discharge_pulse': 44,
            'hppc_short_rest': 45,
            'hppc_charge_pulse': 46,
            'hppc_discharge_to_next_soc': 47
        })

    def test_get_step_index_2(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PreDiag_000400_000084_truncated_structure.json"
        )
        parameters_path = os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters")
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = loadfn(pcycler_run_loc)
        _, protocol_name = os.path.split(pcycler_run.protocol)
        parameter_row, _ = parameters_lookup.get_protocol_parameters(protocol_name, parameters_path=parameters_path)

        step_ind = featurizer_helpers.get_step_index(pcycler_run,
                                                     cycle_type="hppc",
                                                     diag_pos=0)
        self.assertEqual(len(step_ind.values()), 7)

        self.assertEqual(step_ind, {
            'hppc_charge_to_soc': 9,
            'hppc_long_rest': 11,
            'hppc_discharge_pulse': 12,
            'hppc_short_rest': 13,
            'hppc_charge_pulse': 14,
            'hppc_discharge_to_next_soc': 15,
            'hppc_final_discharge': 17
        })
        step_ind = featurizer_helpers.get_step_index(pcycler_run,
                                                     cycle_type="hppc",
                                                     diag_pos=1)
        self.assertEqual(len(step_ind.values()), 7)
        self.assertEqual(step_ind, {
            'hppc_charge_to_soc': 41,
            'hppc_long_rest': 43,
            'hppc_discharge_pulse': 44,
            'hppc_short_rest': 45,
            'hppc_charge_pulse': 46,
            'hppc_discharge_to_next_soc': 47,
            'hppc_final_discharge': 49
        })

    def test_get_diffusion_coeff(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = loadfn(pcycler_run_loc)
            diffusion_df = featurizer_helpers.get_diffusion_coeff(pcycler_run, 1)
            print(np.round(diffusion_df.iloc[0].to_list(), 3))
            self.assertEqual(np.round(diffusion_df.iloc[0].to_list(), 3)[0], -0.016)
            self.assertEqual(np.round(diffusion_df.iloc[0].to_list(), 3)[5], -0.011)

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
            x = [-3.622991274215596, -1.4948801528128568, -2.441732890889216, -0.794422489658189, 0.4889470327970021,
                 0.7562360890191123, -0.9122534588595697, -3.771727344982484, -1.6613278517299095, -3.9279757071656616,
                 0.1418911233780052, 0.7493913209640308, 0.6755655006191633, -1.0823827139302122, -2.484906394983077,
                 -0.8949449222504844, -1.7523322777749897, -1.4575307327423712, 0.4467463228405364, 1.3265006178265961,
                 0.2422557417274141, -2.6373799375134594, -1.230847957965504, -2.046540216421213, 0.2334339752067063,
                 0.8239822694093881, 1.2085578295115413, 0.06687710057927358, -1.0135736732168983, 0.12101479889802537,
                 -2.2735196264247866, 0.37844357940755063, 1.425189114118929, 1.8786507359201035, 1.6731897281287798,
                 -1.1875358619917917, 0.1361208058450041, -1.8275104616090456, -0.2665523054105704, 1.1375831683815445,
                 1.84972885518774, 1.5023615714170622]
            computed = featurizer.X.iloc[0].tolist()
            for indx, value in enumerate(x):
                precision = 6
                self.assertEqual(np.round(value, precision), np.round(computed[indx], precision))

            self.assertEqual(np.round(featurizer.X['var_discharging_capacity'].iloc[0], 3),
                             np.round(-3.771727344982484, 3))

    def test_CycleSummaryStats_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000296_00270E_truncated_structure.json"
            )

            # Test diagnostic with regular cycles
            pcycler_run = loadfn(pcycler_run_loc)
            featurizer = CycleSummaryStats.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            self.assertAlmostEqual(featurizer.X['square_discharging_capacity'].iloc[0], 0.764316, 6)

            # Test diagnostic with regular cycles with different index
            params_dict = {
                "cycle_comp_num": [11, 100]
            }
            features = CycleSummaryStats.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run, params_dict
            )
            self.assertAlmostEqual(features.X['square_discharging_capacity'].iloc[0], 0.7519596, 6)

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
            self.assertEqual(featurizer.X.shape, (30, 9))
            print(list(featurizer.X.iloc[2, :]))
            self.assertListEqual(
                list(featurizer.X.iloc[2, :]),
                [141, 0.9859837086597274, 7.885284043, 4.323121513988055,
                 21.12108276469096, 30, 100, 'reset', 'discharge_energy']
            )

    def test_get_fractional_quantity_remaining_nx(self):
        processed_cycler_run_path_1 = os.path.join(
            TEST_FILE_DIR, "PreDiag_000233_00021F_truncated_structure.json"
        )
        pcycler_run = loadfn(processed_cycler_run_path_1)
        pcycler_run.summary = pcycler_run.summary[~pcycler_run.summary.cycle_index.isin(pcycler_run.diagnostic_summary.cycle_index)]

        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR

        sum_diag = featurizer_helpers.get_fractional_quantity_remaining_nx(pcycler_run,
                                                                           metric="discharge_energy",
                                                                           diagnostic_cycle_type="hppc")
        print(sum_diag["normalized_regular_throughput"])
        self.assertEqual(len(sum_diag.index), 16)
        self.assertEqual(sum_diag.cycle_index.max(), 1507)
        self.assertEqual(np.around(sum_diag["initial_regular_throughput"].iloc[0], 3), np.around(237.001769, 3))
        self.assertEqual(np.around(sum_diag["normalized_regular_throughput"].iloc[15], 3), np.around(45.145, 3))
        self.assertEqual(np.around(sum_diag["normalized_diagnostic_throughput"].iloc[15], 3), np.around(5.098, 3))
        self.assertFalse(sum_diag.isnull().values.any())
        self.assertEqual(sum_diag['diagnostic_start_cycle'].iloc[0], 30)
        self.assertEqual(sum_diag['diagnostic_interval'].iloc[0], 100)

        sum_diag = featurizer_helpers.get_fractional_quantity_remaining_nx(pcycler_run,
                                                                           metric="discharge_energy",
                                                                           diagnostic_cycle_type="rpt_1C")
        self.assertEqual(len(sum_diag.index), 16)
        self.assertEqual(sum_diag.cycle_index.max(), 1509)
        self.assertEqual(np.around(sum_diag["initial_regular_throughput"].iloc[0], 3), np.around(237.001769, 3))
        self.assertEqual(np.around(sum_diag["normalized_regular_throughput"].iloc[15], 3), np.around(45.145, 3))
        self.assertEqual(np.around(sum_diag["normalized_diagnostic_throughput"].iloc[15], 3), np.around(5.229, 3))
        self.assertEqual(sum_diag['diagnostic_start_cycle'].iloc[0], 30)
        self.assertEqual(sum_diag['diagnostic_interval'].iloc[0], 100)

        processed_cycler_run_path_2 = os.path.join(
            TEST_FILE_DIR, "Talos_001383_NCR18650618001_CH31_truncated_structure.json"
        )
        pcycler_run = loadfn(processed_cycler_run_path_2)

        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR

        sum_diag = featurizer_helpers.get_fractional_quantity_remaining_nx(pcycler_run,
                                                                           metric="discharge_energy",
                                                                           diagnostic_cycle_type="hppc")
        self.assertEqual(len(sum_diag.index), 3)
        self.assertEqual(sum_diag.cycle_index.max(), 242)
        self.assertEqual(np.around(sum_diag["initial_regular_throughput"].iloc[0], 3), np.around(331.428, 3))
        self.assertEqual(np.around(sum_diag["normalized_regular_throughput"].iloc[2], 3), np.around(6.817, 3))
        self.assertEqual(np.around(sum_diag["normalized_diagnostic_throughput"].iloc[2], 3), np.around(0.385, 3))
        self.assertEqual(sum_diag['diagnostic_start_cycle'].iloc[0], 30)
        self.assertEqual(sum_diag['diagnostic_interval'].iloc[0], 200)

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
        processed_cycler_run_path_3 = os.path.join(
            TEST_FILE_DIR, "Talos_001380_ICR1865026JM001_CH28_truncated_structure.json"
        )
        processed_cycler_run_path_4 = os.path.join(
            TEST_FILE_DIR, "Talos_001375_NCR18650319002_CH15_truncated_structure.json"
        )
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run = loadfn(processed_cycler_run_path_1)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(0.00472705, decimals=8))

            pcycler_run = loadfn(processed_cycler_run_path_2)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(2.664e-05, decimals=8))

            pcycler_run = loadfn(processed_cycler_run_path_3)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(4.82e-06, decimals=8))

            pcycler_run = loadfn(processed_cycler_run_path_4)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var(v_diff)'], decimals=8),
                             np.round(9.71e-06, decimals=8))


class TestRawToFeatures(unittest.TestCase):
    def setUp(self):
        self.maccor_file_w_parameters_s3 = {
            "bucket": "beep-sync-test-stage",
            "key": "big_file_tests/PreDiag_000287_000128.092"
        }
        self.maccor_file_w_parameters = os.path.join(
            TEST_FILE_DIR, "PreDiag_000287_000128.092"
        )

    @unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
    def test_raw_to_features(self):
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR

        download_s3_object(bucket=self.maccor_file_w_parameters_s3["bucket"],
                           key=self.maccor_file_w_parameters_s3["key"],
                           destination_path=self.maccor_file_w_parameters)

        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            # os.environ['BEEP_PROCESSING_DIR'] = os.getcwd()
            cycler_run = RawCyclerRun.from_file(self.maccor_file_w_parameters)
            processed_cycler_run = cycler_run.to_processed_cycler_run()
            processed_cycler_run_path = os.path.join(
                TEST_FILE_DIR, "processed_diagnostic.json"
            )
            # Dump to the structured file and check the file size
            dumpfn(processed_cycler_run, processed_cycler_run_path)
            # Create dummy json obj
            json_obj = {
                "file_list": [processed_cycler_run_path],
                "run_list": [0],
            }
            json_string = json.dumps(json_obj)

            newjsonpaths = process_file_list_from_json(
                json_string, processed_dir=os.getcwd()
            )
            reloaded = json.loads(newjsonpaths)
            result_list = ['success', 'success', 'success', 'success', 'success', 'success', 'success']
            self.assertEqual(reloaded['result_list'], result_list)
            rpt_df = loadfn(reloaded['file_list'][0])
            self.assertEqual(np.round(rpt_df.X['m0_Amp_rpt_0.2C_1'].iloc[0], 6), 0.867371)
