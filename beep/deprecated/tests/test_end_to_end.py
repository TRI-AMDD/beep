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

import unittest
import os
import shutil
import subprocess
import json

import pandas as pd

from tempfile import mkdtemp
from monty.serialization import loadfn

from beep import collate, validate, structure, featurize
from beep.utils import os_format
from beep.tests.constants import TEST_FILE_DIR


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        # Get cwd, create and enter scratch dir
        self.cwd = os.getcwd()
        scratch_dir = mkdtemp()
        os.chdir(scratch_dir)
        self.scratch_dir = scratch_dir

        # Set BEEP_PROCESSING_DIR directory to scratch_dir
        os.environ["BEEP_PROCESSING_DIR"] = scratch_dir

        # Create data-share and subfolders
        os.mkdir("data-share")
        os.chdir("data-share")

        # Set up directory structure and specify the test files
        os.mkdir("raw_cycler_files")

        # Copy starting files into raw_cycler_files directory
        starting_files = [
            "2017-12-04_4_65C-69per_6C_CH29.csv",
            "2017-05-09_test-TC-contact_CH33.csv",  # Fails for not meeting naming convention
            "2017-08-14_8C-5per_3_47C_CH44.csv",
        ]
        starting_files = [os.path.join(TEST_FILE_DIR, path) for path in starting_files]
        for file in starting_files:
            shutil.copy(file, "raw_cycler_files")
            shutil.copy(file.replace(".csv", "_Metadata.csv"), "raw_cycler_files")

        # Go back into test directory
        os.chdir("..")

    def tearDown(self):
        # Go back to initial directory and tear down test dir
        os.chdir(self.cwd)
        shutil.rmtree(self.scratch_dir)

    def test_python(self):
        """Python script for end to end test"""
        # Copy
        mapped = collate.process_files_json()
        rename_output = json.loads(mapped)
        rename_output["run_list"] = list(range(len(rename_output["file_list"])))
        mapped = json.dumps(rename_output)

        # Validation
        validated = validate.validate_file_list_from_json(mapped)
        validated_output = json.loads(validated)

        validated_output["run_list"] = list(range(len(validated_output["file_list"])))
        validated = json.dumps(validated_output)

        # Data structuring
        structured = structure.process_file_list_from_json(validated)
        structured_output = json.loads(structured)

        structured_output["run_list"] = list(range(len(structured_output["file_list"])))
        structured = json.dumps(structured_output)

        # Featurization
        featurized = featurize.process_file_list_from_json(structured)
        featurized_output = json.loads(featurized)

        featurized_output["run_list"] = list(range(len(featurized_output["file_list"])))
        featurized = json.dumps(featurized_output)

        # Prediction
        # predictions = run_model.process_file_list_from_json(
        #     featurized, model_dir=MODEL_DIR)

        # Validate output files
        self._check_result_file_validity()

    def test_console(self):
        """
        Console command for end to end test, run by passing the output of
        `program_executable [JSON_STRING]` from module to module, essentially simulating
        > collate | validate | structure | featurize | run_model
        """

        rename_output = subprocess.check_output("collate", shell=True).decode("utf-8")

        # Validation console test
        rename_output = json.loads(rename_output)
        validation_input = {
            "file_list": rename_output[
                "file_list"
            ],  # list of file paths ['path/test1.csv', 'path/test2.csv']
            "run_list": list(
                range(len(rename_output["file_list"]))
            ),  # list of run_ids [0, 1]
        }
        validation_input = os_format(json.dumps(validation_input))
        validation_output = subprocess.check_output(
            "validate {}".format(validation_input), shell=True
        ).decode("utf-8")

        # Structure console test
        validation_output = json.loads(validation_output)
        structure_input = {
            "file_list": validation_output[
                "file_list"
            ],  # list of file paths ['path/test1.json', 'path/test2.json']
            "run_list": list(
                range(len(validation_output["file_list"]))
            ),  # list of run_ids [0, 1]
            "validity": validation_output[
                "validity"
            ],  # list of validities ['valid', 'invalid']
        }
        structure_input = os_format(json.dumps(structure_input))
        structure_output = subprocess.check_output(
            "structure {}".format(structure_input), shell=True
        ).decode("utf-8")

        # Featurizing console test
        structure_output = json.loads(structure_output)
        feature_input = {
            "file_list": structure_output[
                "file_list"
            ],  # list of file paths ['path/test1.json', 'path/test2.json']
            "run_list": list(
                range(len(structure_output["file_list"]))
            ),  # list of run_ids [0, 1]
        }
        feature_input = os_format(json.dumps(feature_input))
        feature_output = subprocess.check_output(
            "featurize {}".format(feature_input), shell=True
        ).decode("utf-8")

        # Fitting console test
        # feature_output = json.loads(feature_output)
        # fitting_input = {
        #     "mode": self.events_mode,  # mode run|test|events_off
        #     "file_list": feature_output['file_list'],  # list of file paths ['path/test1.json', 'path/test2.json']
        #     'run_list': list(range(len(feature_output['file_list'])))  # list of run_ids [0, 1]
        #     }
        # fitting_input = os_format(json.dumps(fitting_input))
        # model_output = subprocess.check_output("run_model {}".format(fitting_input),
        #                                        shell=True).decode('utf-8')

        # Validate output files
        self._check_result_file_validity()

    def _check_result_file_validity(self):
        """Single routine to validate file results from end-to-end tests"""
        # Validate that files are in the right place
        rename_map = pd.read_csv(
            os.path.join(
                "data-share",
                "renamed_cycler_files",
                "FastCharge",
                "FastCharge" + "map.csv",
            )
        )

        self.assertEqual(rename_map.channel_no.tolist(), ["CH33", "CH44", "CH29"])

        loaded_structure = loadfn(
            os.path.join(
                "data-share", "structure", "FastCharge_000002_CH29_structure.json"
            )
        )
        self.assertIsInstance(loaded_structure, structure.BEEPDatapath)

        loaded_features = loadfn(
            os.path.join(
                "data-share",
                "features",
                "DeltaQFastCharge",
                "FastCharge_000002_CH29_features_DeltaQFastCharge.json",
            )
        )
        self.assertIsInstance(loaded_features, featurize.DeltaQFastCharge)

        # loaded_prediction = loadfn(
        #     os.path.join("data-share", "predictions", "FastCharge_000002_CH29_full_model_multi_predictions.json"))
        # self.assertAlmostEqual(np.floor(loaded_prediction['cycle_number'][0]), 121)


if __name__ == "__main__":
    unittest.main()
