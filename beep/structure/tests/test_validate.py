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
"""Unit tests related to batch validation"""

import json
import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from monty.tempfile import ScratchDir
from beep.structure.validate import SimpleValidator
from beep import S3_CACHE, VALIDATION_SCHEMA_DIR
from beep.tests.constants import TEST_FILE_DIR


@unittest.skip("Needs to be integrated with new structuring")
class ValidationMaccorTest(unittest.TestCase):
    # To further develop as Maccor data / schema becomes available
    def setUp(self):
        pass

    def test_validation_maccor(self):
        path = "xTESLADIAG_000019_CH70.070"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator(
            schema_filename=os.path.join(
                VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml"
            )
        )
        v.allow_unknown = True
        header = pd.read_csv(path, delimiter="\t", nrows=0)
        df = pd.read_csv(path, delimiter="\t", skiprows=1)
        df["State"] = df["State"].astype(str)
        df["current"] = df["Amps"]
        validity, reason = v.validate(df)
        self.assertTrue(validity)

    def test_invalidation_maccor(self):
        path = "PredictionDiagnostics_000109_tztest.010"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator(
            schema_filename=os.path.join(
                VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml"
            )
        )
        v.allow_unknown = True
        header = pd.read_csv(path, delimiter="\t", nrows=0)
        print(header)
        df = pd.read_csv(path, delimiter="\t", skiprows=1)
        df["State"] = df["State"].astype(str)
        df["current"] = df["Amps"]
        print(df.dtypes)
        validity, reason = v.validate(df)
        print(validity, reason)
        self.assertFalse(validity)

    def test_validate_from_paths_maccor(self):
        paths = [os.path.join(TEST_FILE_DIR, "xTESLADIAG_000019_CH70.070")]
        with ScratchDir(".") as scratch_dir:
            # Run validation on everything
            v = SimpleValidator()
            validate_record = v.validate_from_paths(
                paths,
                record_results=True,
                skip_existing=False,
                record_path=os.path.join(scratch_dir, "validation_records.json"),
            )
            df = pd.DataFrame(v.validation_records)
            df = df.transpose()
            self.assertEqual(
                df.loc["xTESLADIAG_000019_CH70.070", "method"], "simple_maccor"
            )
            self.assertEqual(df.loc["xTESLADIAG_000019_CH70.070", "validated"], True)

    def test_monotonic_cycle_index_maccor(self):
        path = "PredictionDiagnostics_000109_tztest.010"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator(
            schema_filename=os.path.join(
                VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml"
            )
        )
        v.allow_unknown = True
        header = pd.read_csv(path, delimiter="\t", nrows=0)
        print(header)
        df = pd.read_csv(path, delimiter="\t", skiprows=1)
        df["State"] = df["State"].astype(str)
        df["current"] = df["Amps"]
        df.loc[df["Cyc#"] == 89, "Cyc#"] = 0
        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(reason, "cyc# needs to be monotonically increasing for processing")


@unittest.skip("Needs to be integrated with new structuring")
class ValidationEisTest(unittest.TestCase):
    # To further develop
    def setUp(self):
        pass

    def test_validation_maccor(self):
        path = "maccor_test_file_4267-66-6519.EDA0001.041"
        path = os.path.join(TEST_FILE_DIR, path)

        v = ValidatorBeep()
        v.allow_unknown = True

        df = pd.read_csv(path, delimiter="\t", skip_blank_lines=True, skiprows=10)

        self.assertTrue(v.validate_eis_dataframe(df))


@unittest.skip("Needs to be integrated with new structuring")
class SimpleValidatorTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_file_incomplete(self):
        path = "FastCharge_000025_CH8.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)

        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(
            reason,
            "cycle_index needs to reach at least 1 "
            "for processing, instead found:value=0.0",
        )

    def test_basic(self):
        path = "2017-05-09_test-TC-contact_CH33.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)

        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(
            reason,
            "Column cycle_index: integer type check failed "
            "at index 0 with value nan",
        )

        # Test bigger file, with float/numeric type
        path = "2017-08-14_8C-5per_3_47C_CH44.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)
        v.schema["cycle_index"]["schema"]["type"] = "float"
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, "")

        v.schema["cycle_index"]["schema"]["type"] = "numeric"
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, "")

        # Test good file
        path = "2017-12-04_4_65C-69per_6C_CH29.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, "")

        # Alter the schema on-the-fly to induce error
        v.schema["discharge_capacity"]["schema"]["max"] = 1.8
        validity, reason = v.validate(df)
        self.assertFalse(validity)
        # Cut off reasons to prevent floating format discrepancies
        self.assertEqual(
            reason[:80],
            "discharge_capacity is higher than allowed max 1.8 "
            "at index 11154: value=1.801418",
        )

        # Alter the schema on-the-fly to move on to the next errors
        v.schema["discharge_capacity"]["schema"]["max"] = 2.1
        v.schema["step_time"] = {"schema": {"min": 0.0, "type": "float"}}
        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(
            reason[:70],
            "step_time is lower than allowed min 0.0 " "at index 104416:value=-450.945",
        )

        # Alter schema once more to recover validation
        del v.schema["step_time"]["schema"]["min"]
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, "")

    def test_project_name(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "validation"))

            v = SimpleValidator()
            paths = [
                "FastCharge_000000_CH29.csv",
                "FastCharge_000025_CH8.csv",
                "PredictionDiagnostics_000151_test.052",
                "PriorLAMVal_000101_000080_CH2.csv",
            ]
            paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
            validate_record = v.validate_from_paths(
                paths, record_results=True, skip_existing=False
            )

            self.assertEqual(
                validate_record["FastCharge_000000_CH29.csv"]["method"],
                "schema-arbin-lfp.yaml",
            )
            self.assertEqual(
                validate_record["FastCharge_000025_CH8.csv"]["method"],
                "schema-arbin-lfp.yaml",
            )
            self.assertEqual(
                validate_record["PredictionDiagnostics_000151_test.052"]["method"],
                "schema-maccor-2170.yaml",
            )
            self.assertEqual(
                validate_record["PriorLAMVal_000101_000080_CH2.csv"]["method"],
                "schema-arbin-18650-2170.yaml",
            )

    def test_validation_from_json(self):
        with ScratchDir("."):
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "validation"))
            paths = [
                "2017-05-09_test-TC-contact_CH33.csv",
                "2017-12-04_4_65C-69per_6C_CH29.csv",
            ]
            paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
            # Create dummy json obj
            json_obj = {
                "file_list": paths,
                "run_list": list(range(len(paths))),
            }
            json_string = json.dumps(json_obj)
            json_output = validate_file_list_from_json(json_string)
            loaded = json.loads(json_output)
        self.assertEqual(loaded["validity"][0], "invalid")
        self.assertEqual(loaded["validity"][1], "valid")

        # Workflow output
        output_file_path = Path(tempfile.gettempdir()) / "results.json"
        self.assertTrue(output_file_path.exists())

        output_json = json.loads(output_file_path.read_text())

        self.assertEqual(paths[0], output_json["filename"])
        self.assertEqual(os.path.getsize(output_json["filename"]), output_json["size"])
        self.assertEqual(0, output_json["run_id"])
        self.assertEqual("validating", output_json["action"])
        self.assertEqual("invalid", output_json["status"])

    @unittest.skipUnless(False, "toggle this test")
    def test_heavy(self):
        # Sync all S3 objects
        from beep.utils.memprof import cache_all_kitware_data

        cache_all_kitware_data()
        paths = os.listdir(os.path.join(S3_CACHE, "D3Batt_Data_publication"))
        paths = [path for path in paths if not "Metadata" in path]
        paths = [
            os.path.join(S3_CACHE, "D3Batt_Data_publication", path) for path in paths
        ]

        # Run validation on everything
        v = SimpleValidator()
        validate_record = v.validate_from_paths(
            paths, record_results=True, skip_existing=True
        )
        df = pd.DataFrame(v.validation_records)
        df = df.transpose()
        print(df)
        print(
            "{} valid, {} invalid".format(
                len([x for x in df.validated if x]),
                len([x for x in df.validated if not x]),
            )
        )
        invalid_runs = df[np.logical_not(df.validated)]
        print(invalid_runs)
        self.assertEqual(len(invalid_runs), 3)


if __name__ == "__main__":
    unittest.main()
