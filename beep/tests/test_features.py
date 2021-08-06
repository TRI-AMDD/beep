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
import shutil
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
    HPPCResistanceVoltageFeatures,
    DiagnosticProperties,
    DiagnosticSummaryStats,
    CycleSummaryStats
)
from beep.structure.maccor import MaccorDatapath
from beep.structure.cli import auto_load_processed
from beep.features import featurizer_helpers
from beep.utils import parameters_lookup
from monty.serialization import dumpfn, loadfn
from monty.tempfile import ScratchDir
from beep.utils.s3 import download_s3_object
from beep.tests.constants import TEST_FILE_DIR, BIG_FILE_TESTS, SKIP_MSG
from beep import MODULE_DIR


MACCOR_FILE_W_DIAGNOSTICS = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000020_CH71.071")
MACCOR_FILE_W_PARAMETERS = os.path.join(
    TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010"
)
FEATURE_HYPERPARAMS = loadfn(
    os.path.join(MODULE_DIR, "features/feature_hyperparameters.yaml")
)


class TestFeaturizer(unittest.TestCase):
    def setUp(self):
        self.processed_cycler_file = "2017-06-30_2C-10per_6C_CH10_structure.json"
        self.processed_cycler_file_insuf = "structure_insufficient.json"

    def test_feature_generation_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, self.processed_cycler_file)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = auto_load_processed(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            self.assertEqual(len(featurizer.X), 1)  # just test if works for now
            # Ensure no NaN values
            # print(featurizer.X.to_dict())
            self.assertFalse(np.any(featurizer.X.isnull()))
            self.assertEqual(np.round(featurizer.X.loc[0, 'intercept_discharge_capacity_cycle_number_91:100'], 6),
                             np.round(1.1050065801818196, 6))

    def test_feature_old_class(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, self.processed_cycler_file)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            predictor = DegradationPredictor.from_processed_cycler_run_file(
                processed_cycler_run_path, features_label="full_model"
            )
            self.assertEqual(predictor.feature_labels[4], "charge_time_cycles_1:5")

    def test_feature_label_full_model(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, self.processed_cycler_file)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = auto_load_processed(processed_cycler_run_path)
            featurizer = DeltaQFastCharge.from_run(
                processed_cycler_run_path, os.getcwd(), pcycler_run
            )

            self.assertEqual(featurizer.X.columns.tolist()[4], "charge_time_cycles_1:5")

    def test_feature_serialization(self):
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, self.processed_cycler_file)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = auto_load_processed(processed_cycler_run_path)
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
        processed_cycler_run_path = os.path.join(TEST_FILE_DIR, self.processed_cycler_file)
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            pcycler_run = auto_load_processed(processed_cycler_run_path)
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
            pcycler_run = auto_load_processed(pcycler_run_loc)
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
                    self.assertEqual(featurizer.metadata["protocol"], '2017-06-30_tests\\20170629-2C_10per_6C.sdu')
                    self.assertEqual(featurizer.metadata["barcode"], 'el150800460605')
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
            shutil.copy(os.path.join(TEST_FILE_DIR, "data-share", "raw", "cell_info", "anode_test.csv"),
                        os.path.join(TEST_FILE_DIR, "data-share", "raw", "cell_info",
                                     FEATURE_HYPERPARAMS["IntracellFeatures"]["anode_file"])
                        )
            shutil.copy(os.path.join(TEST_FILE_DIR, "data-share", "raw", "cell_info", "cathode_test.csv"),
                        os.path.join(TEST_FILE_DIR, "data-share", "raw", "cell_info",
                                     FEATURE_HYPERPARAMS["IntracellFeatures"]["cathode_file"])
                        )

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
            features_reloaded = loadfn(reloaded["file_list"][2])
            self.assertIsInstance(features_reloaded, DeltaQFastCharge)
            self.assertEqual(
                features_reloaded.X.loc[0, "nominal_capacity_by_median"],
                0.07114775279999999,
            )
            features_reloaded = loadfn(reloaded["file_list"][4])
            self.assertIsInstance(features_reloaded, DiagnosticProperties)
            self.assertListEqual(
                list(features_reloaded.X.iloc[2, :]),
                [141, 0.9859837086597274, 7.885284043, 4.323121513988055,
                 21.12108276469096, 30, 100, 1577338063, 'reset', 'discharge_energy'],
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
            TEST_FILE_DIR, self.processed_cycler_file_insuf
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


    def test_HPPCResistanceVoltageFeatures_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = auto_load_processed(pcycler_run_loc)
            featurizer = HPPCResistanceVoltageFeatures.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "HPPCResistanceVoltageFeatures")
            self.assertEqual(featurizer.X.shape[1], 76)
            self.assertEqual(featurizer.X.columns[0], "r_c_0s_00")
            self.assertEqual(featurizer.X.columns[-1], "D_8")

            self.assertAlmostEqual(featurizer.X.iloc[0, 0], -0.08845776922490017, 6)
            self.assertAlmostEqual(featurizer.X.iloc[0, 5], -0.1280224700339366, 6)
            self.assertAlmostEqual(featurizer.X.iloc[0, 27], -0.10378359476555565, 6)

    def test_DiagnosticSummaryStats_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json")
            pcycler_run = auto_load_processed(pcycler_run_loc)
            featurizer = DiagnosticSummaryStats.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "DiagnosticSummaryStats")
            self.assertEqual(featurizer.X.shape[1], 54)
            self.assertListEqual(
                [featurizer.X.columns[0], featurizer.X.columns[41]],
                ["var_charging_capacity", "square_discharging_dQdV"],
            )
            self.assertListEqual(
                [featurizer.X.columns[42], featurizer.X.columns[53]],
                ["diag_sum_diff_0_1_rpt_0.2Cdischarge_capacity", "diag_sum_diff_0_1_rpt_2Ccharge_energy"],
            )
            x = [-3.622991274215596, -1.4948801528128568, -2.441732890889216, -0.794422489658189, 0.4889470327970021,
                 0.7562360890191123, -0.9122534588595697, -3.771727344982484, -1.6613278517299095, -3.9279757071656616,
                 0.1418911233780052, 0.7493913209640308, 0.6755655006191633, -1.0823827139302122, -2.484906394983077,
                 -0.8949449222504844, -1.7523322777749897, -1.4575307327423712, 0.4467463228405364, 1.3265006178265961,
                 0.2422557417274141, -2.6373799375134594, -1.230847957965504, -2.046540216421213, 0.2334339752067063,
                 0.8239822694093881, 1.2085578295115413, 0.06687710057927358, -1.0135736732168983, 0.12101479889802537,
                 -2.2735196264247866, 0.37844357940755063, 1.425189114118929, 1.8786507359201035, 1.6731897281287798,
                 -1.1875358619917917, 0.1361208058450041, -1.8275104616090456, -0.2665523054105704, 1.1375831683815445,
                 1.84972885518774, 1.5023615714170622, -0.00472514151532623, -0.003475275535937185,
                 -0.008076419207993832, -0.008621551983451683, 7.413107429038043e-05, 0.0013748657878274915,
                 -0.005084993748595586, -0.005675990891556979, -0.002536196993382343, -0.0018987653783979423,
                 -0.00016598153694586686, -0.00105148083990717]
            computed = featurizer.X.iloc[0].tolist()
            for indx, value in enumerate(x):
                precision = 5
                self.assertEqual(np.around(np.float32(value), precision),
                                 np.around(np.float32(computed[indx]), precision))

            self.assertEqual(np.around(featurizer.X['var_discharging_capacity'].iloc[0], 6),
                             np.around(-3.771727344982484, 6))

            pcycler_run_loc = os.path.join(TEST_FILE_DIR,
                                           "PredictionDiagnostics_000136_00002D_truncated_structure.json")
            pcycler_run = auto_load_processed(pcycler_run_loc)
            featurizer = DiagnosticSummaryStats.from_run(pcycler_run_loc, os.getcwd(), pcycler_run)
            x = [-2.4602845133649374, -0.7912059829821004, -1.3246516129064152, -0.5577484175221676,
                 0.22558675296269257, 1.4107424811304434, 0.44307560772987753, -2.968731527885897,
                 -1.003386799815887, -1.2861922579124305, 0.010393880890967514, 0.4995216948726259,
                 1.4292366107477192, 0.2643953383205679, -1.3377336978836682, -0.21470956778563194,
                 -0.7617667690573674, -0.47886877345098366, 0.23547492071796852, 1.9699615602673914,
                 1.566893893282218, -1.8282011110054657, -0.46311299104523346, -0.7166620260036703,
                 0.06268262404068164, 0.5400910355865228, 2.00139593781454, 1.4038773986895716,
                 0.46799197793006897, 0.5117431282997131, -1.4615182876586914, 1.2889420237956628,
                 2.6205135712205725, 2.176016330718994, 3.1539101600646973, -0.9218153953552246,
                 0.23360896110534668, -1.1706260442733765, -0.5070897459236073, 1.1722059184617377,
                 2.0029776096343994, 1.7837194204330444, -0.021425815851990795, -0.020270314430328763,
                 -0.028696091773302315, -0.02782930233422708, -0.017478835661355316, -0.019788159842565697,
                 -0.021354840746757066, -0.021056601447539146, -0.026599426370616085, -0.03017946374275189,
                 -0.017983518726387225, -0.01771638489069907]
            computed = featurizer.X.iloc[0].tolist()
            for indx, value in enumerate(x):
                precision = 5
                self.assertEqual(np.around(np.float32(value), precision),
                                 np.around(np.float32(computed[indx]), precision))

    def test_CycleSummaryStats_class(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000296_00270E_truncated_structure.json"
            )

            # Test diagnostic with regular cycles
            pcycler_run = auto_load_processed(pcycler_run_loc)
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
            pcycler_run = auto_load_processed(pcycler_run_loc)
            featurizer = DiagnosticProperties.from_run(
                pcycler_run_loc, os.getcwd(), pcycler_run
            )
            path, local_filename = os.path.split(featurizer.name)
            folder = os.path.split(path)[-1]
            dumpfn(featurizer, featurizer.name)
            self.assertEqual(folder, "DiagnosticProperties")
            self.assertEqual(featurizer.X.shape, (30, 10))
            print(list(featurizer.X.iloc[2, :]))
            self.assertListEqual(
                list(featurizer.X.iloc[2, :]),
                [141, 0.9859837086597274, 7.885284043, 4.323121513988055,
                 21.12108276469096, 30, 100, 1577338063, 'reset', 'discharge_energy']
            )

    @unittest.skip
    def test_features_on_list(self):
        files = [
            "PredictionDiagnostics_000102_0001B1_structure.json",
            "PredictionDiagnostics_000103_0001B3_structure.json",
            "PredictionDiagnostics_000114_00003C_structure.json",
            "PredictionDiagnostics_000117_00003E_structure.json",
            "PredictionDiagnostics_000120_000041_structure.json",
            "PredictionDiagnostics_000122_000043_structure.json",
            "PredictionDiagnostics_000124_000049_structure.json",
            "PredictionDiagnostics_000130_000044_structure.json",
            "PredictionDiagnostics_000133_00004D_structure (2).json",
            "PredictionDiagnostics_000136_00002D_structure (1).json",
            "PredictionDiagnostics_000139_000034_structure.json",
            "PredictionDiagnostics_000144_00002E_structure.json",
            "PredictionDiagnostics_000148_000038_structure.json",
            "PredictionDiagnostics_000150_00003B_structure.json",

            "PredictionDiagnostics_000156_000023_structure.json",
            "PredictionDiagnostics_000160_000251_structure.json",
            "PredictionDiagnostics_000163_000022_structure.json",
            "PredictionDiagnostics_000163_000022_structure.json",
            "PredictionDiagnostics_000164_000239_structure.json",
            "PredictionDiagnostics_000167_000255_structure.json",
            "PredictionDiagnostics_000168_000253_structure.json",
            "PredictionDiagnostics_000175_000247_structure.json",
            "PredictionDiagnostics_000178_00023B_structure.json",
            "PredictionDiagnostics_000181_00023A_structure.json",

            "PredictionDiagnostics_000184_000244_structure.json",
            "PredictionDiagnostics_000186_00024E_structure.json",
            "PredictionDiagnostics_000194_000242_structure.json",
        ]
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            for file in files:
                pcycler_run_path = os.path.join(TEST_FILE_DIR, file)
                json_obj = {
                    "file_list": [pcycler_run_path],
                    "run_list": [0],
                }
                json_string = json.dumps(json_obj)

                newjsonpaths = process_file_list_from_json(
                    json_string, processed_dir=os.getcwd()
                )


class TestFeaturizerHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_fractional_quantity_remaining_nx(self):
        processed_cycler_run_path_1 = os.path.join(
            TEST_FILE_DIR, "PreDiag_000233_00021F_truncated_structure.json"
        )
        pcycler_run = auto_load_processed(processed_cycler_run_path_1)
        pcycler_run.structured_summary = pcycler_run.structured_summary[
            ~pcycler_run.structured_summary.cycle_index.isin(pcycler_run.diagnostic_summary.cycle_index)]

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
        self.assertEqual(sum_diag['epoch_time'].iloc[0], 1576641695)

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
        self.assertEqual(sum_diag['epoch_time'].iloc[0], 1576736230)

        processed_cycler_run_path_2 = os.path.join(
            TEST_FILE_DIR, "Talos_001383_NCR18650618001_CH31_truncated_structure.json"
        )
        pcycler_run = auto_load_processed(processed_cycler_run_path_2)

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
        self.assertEqual(sum_diag['epoch_time'].iloc[0], 1598156928)

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

            # processed_cycler_run_path_1
            pcycler_run = auto_load_processed(processed_cycler_run_path_1)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var_v_diff'], decimals=8),
                             np.round(0.00472705, decimals=8))
            self.assertListEqual(list(v_vars_df.columns),
                                 ["var_v_diff", "min_v_diff", "mean_v_diff", "skew_v_diff", "kurtosis_v_diff",
                                  "sum_v_diff", "sum_square_v_diff"])

            temp_list = v_vars_df.iloc[0, :].to_list()
            temp_list = [np.round(float(x), 8) for x in temp_list]
            self.assertListEqual(temp_list,
                                 [0.00472705, 0.0108896, 0.13865059, 0.59427689, 2.36743208, 176.50219843, 30.4896637])

            # processed_cycler_run_path_2
            pcycler_run = auto_load_processed(processed_cycler_run_path_2)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var_v_diff'], decimals=8),
                             np.round(2.664e-05, decimals=8))
            self.assertListEqual(list(v_vars_df.columns),
                                 ["var_v_diff", "min_v_diff", "mean_v_diff", "skew_v_diff", "kurtosis_v_diff",
                                  "sum_v_diff", "sum_square_v_diff"])

            temp_list = v_vars_df.iloc[0, :].to_list()
            temp_list = [np.round(float(x), 8) for x in temp_list]
            self.assertListEqual(temp_list,
                                 [2.664e-05, 0.01481062, 0.01993318, 1.70458503, 4.89453871, 6.83708111, 0.14542267])

            # processed_cycler_run_path_3
            pcycler_run = auto_load_processed(processed_cycler_run_path_3)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var_v_diff'], decimals=8),
                             np.round(4.82e-06, decimals=8))
            self.assertListEqual(list(v_vars_df.columns),
                                 ["var_v_diff", "min_v_diff", "mean_v_diff", "skew_v_diff", "kurtosis_v_diff",
                                  "sum_v_diff", "sum_square_v_diff"])

            temp_list = v_vars_df.iloc[0, :].to_list()
            temp_list = [np.round(float(x), 8) for x in temp_list]
            self.assertListEqual(temp_list,
                                 [4.82e-06, 0.01134005, 0.01569094, -0.01052989, 3.25562527, 4.07964428, 0.06526675])

            # processed_cycler_run_path_4
            pcycler_run = auto_load_processed(processed_cycler_run_path_4)
            v_vars_df = featurizer_helpers.get_v_diff(pcycler_run, 1, 8)
            print(v_vars_df)
            self.assertEqual(np.round(v_vars_df.iloc[0]['var_v_diff'], decimals=8),
                             np.round(9.71e-06, decimals=8))
            self.assertListEqual(list(v_vars_df.columns),
                                 ["var_v_diff", "min_v_diff", "mean_v_diff", "skew_v_diff", "kurtosis_v_diff",
                                  "sum_v_diff", "sum_square_v_diff"])

            temp_list = v_vars_df.iloc[0, :].to_list()
            temp_list = [np.round(float(x), 8) for x in temp_list]
            self.assertListEqual(temp_list,
                                 [9.71e-06, -0.01138431, 0.00490308, -3.09586327, 13.72199015, 2.16744705, 0.01306312])

    def test_get_hppc_ocv(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
        )
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = auto_load_processed(pcycler_run_loc)
        hppc_ocv_features = featurizer_helpers.get_hppc_ocv(pcycler_run, 1)
        self.assertAlmostEqual(hppc_ocv_features['var_ocv'].iloc[0], 0.000016, 6)
        self.assertAlmostEqual(hppc_ocv_features['min_ocv'].iloc[0], -0.001291, 6)
        self.assertAlmostEqual(hppc_ocv_features['mean_ocv'].iloc[0], 0.002221, 6)
        self.assertAlmostEqual(hppc_ocv_features['skew_ocv'].iloc[0], 1.589392, 6)
        self.assertAlmostEqual(hppc_ocv_features['kurtosis_ocv'].iloc[0], 7.041016, 6)
        self.assertAlmostEqual(hppc_ocv_features['sum_ocv'].iloc[0], 0.025126, 6)
        self.assertAlmostEqual(hppc_ocv_features['sum_square_ocv'].iloc[0], 0.000188, 6)

    def test_get_step_index(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
        )

        parameters_path = os.path.join(TEST_FILE_DIR, "data-share", "raw", "parameters")
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = auto_load_processed(pcycler_run_loc)
        data = pcycler_run.diagnostic_data
        hppc_cycles = data.loc[data.cycle_type == "hppc"]
        print(hppc_cycles.step_index.unique())
        _, protocol_name = os.path.split(pcycler_run.metadata.protocol)
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
        pcycler_run = auto_load_processed(pcycler_run_loc)
        _, protocol_name = os.path.split(pcycler_run.metadata.protocol)
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
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="reset", diag_pos=0)
        self.assertEqual(step_ind, {'reset_charge': 5, 'reset_discharge': 6})
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="reset", diag_pos=1)
        self.assertEqual(step_ind, {'reset_charge': 38, 'reset_discharge': 39})

        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_0.2C", diag_pos=0)
        self.assertEqual(step_ind, {'rpt_0.2C_charge': 19, 'rpt_0.2C_discharge': 20})
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_0.2C", diag_pos=1)
        self.assertEqual(step_ind, {'rpt_0.2C_charge': 51, 'rpt_0.2C_discharge': 52})

        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_1C", diag_pos=0)
        self.assertEqual(step_ind, {'rpt_1C_charge': 22, 'rpt_1C_discharge': 23})
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_1C", diag_pos=1)
        self.assertEqual(step_ind, {'rpt_1C_charge': 54, 'rpt_1C_discharge': 55})

        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_2C", diag_pos=0)
        self.assertEqual(step_ind, {'rpt_2C_charge': 25, 'rpt_2C_discharge': 26})
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="rpt_2C", diag_pos=1)
        self.assertEqual(step_ind, {'rpt_2C_charge': 57, 'rpt_2C_discharge': 58})

    def test_get_step_index_3(self):
        pcycler_run_loc = os.path.join(
            TEST_FILE_DIR, "PredictionDiagnostics_000136_00002D_truncated_structure.json"
        )
        os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
        pcycler_run = auto_load_processed(pcycler_run_loc)
        step_ind = featurizer_helpers.get_step_index(pcycler_run, cycle_type="hppc", diag_pos=0)
        self.assertEqual(len(step_ind.values()), 6)

    def test_get_diffusion_coeff(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = TEST_FILE_DIR
            pcycler_run_loc = os.path.join(
                TEST_FILE_DIR, "PreDiag_000240_000227_truncated_structure.json"
            )
            pcycler_run = auto_load_processed(pcycler_run_loc)
            diffusion_df = featurizer_helpers.get_diffusion_coeff(pcycler_run, 1)
            print(np.round(diffusion_df.iloc[0].to_list(), 3))
            self.assertEqual(np.round(diffusion_df.iloc[0].to_list(), 3)[0], -0.016)
            self.assertEqual(np.round(diffusion_df.iloc[0].to_list(), 3)[5], -0.011)


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
            dp = MaccorDatapath.from_file(self.maccor_file_w_parameters)
            dp.autostructure()
            processed_run_path = os.path.join(
                TEST_FILE_DIR, "processed_diagnostic.json"
            )
            # Dump to the structured file and check the file size
            dumpfn(dp, processed_run_path)
            # Create dummy json obj
            json_obj = {
                "file_list": [processed_run_path],
                "run_list": [0],
            }
            json_string = json.dumps(json_obj)

            newjsonpaths = process_file_list_from_json(
                json_string, processed_dir=os.getcwd()
            )

            reloaded = json.loads(newjsonpaths)

            import pprint
            pprint.pprint(reloaded)

            result_list = ['success'] * 7
            self.assertEqual(reloaded['result_list'], result_list)
            res_df = loadfn(reloaded['file_list'][0])
            self.assertEqual(res_df.class_feature_name, "HPPCResistanceVoltageFeatures")
            print(res_df.X)
            self.assertAlmostEqual(res_df.X['r_c_0s_00'].iloc[0], -0.159771397, 5)
            self.assertAlmostEqual(res_df.X['r_c_0s_10'].iloc[0], -0.143679, 5)
            self.assertAlmostEqual(res_df.X['r_c_0s_20'].iloc[0], -0.146345, 5)
            self.assertAlmostEqual(res_df.X['D_6'].iloc[0], -0.167919, 5)
            self.assertAlmostEqual(res_df.X['D_7'].iloc[0], 0.094136, 5)
            self.assertAlmostEqual(res_df.X['D_8'].iloc[0], 0.172496, 5)
