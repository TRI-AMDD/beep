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
"""Unit tests related to Splicing files"""

import os
import unittest

import numpy as np

from beep.utils import MaccorSplice
from beep.utils.s3 import download_s3_object
from beep import MODULE_DIR
from beep.utils import parameters_lookup
from beep.tests.constants import BIG_FILE_TESTS, TEST_FILE_DIR, SKIP_MSG
from beep import PROTOCOL_PARAMETERS_DIR


class SpliceTest(unittest.TestCase):
    def setUp(self):
        self.arbin_file = os.path.join(TEST_FILE_DIR, "FastCharge_000000_CH29.csv")
        self.filename_part_1 = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038.078")
        self.filename_part_2 = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038con.078")
        self.output = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038joined.078")
        self.test = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038test.078")

    def test_maccor_read_write(self):
        splicer = MaccorSplice(self.filename_part_1, self.filename_part_2, self.output)

        meta_1, data_1 = splicer.read_maccor_file(self.filename_part_1)
        splicer.write_maccor_file(meta_1, data_1, self.test)
        meta_test, data_test = splicer.read_maccor_file(self.test)

        assert meta_1 == meta_test
        assert np.allclose(data_1["Volts"].to_numpy(), data_test["Volts"].to_numpy())
        assert np.allclose(data_1["Amps"].to_numpy(), data_test["Amps"].to_numpy())
        assert np.allclose(
            data_1["Test (Sec)"].to_numpy(), data_test["Test (Sec)"].to_numpy()
        )

    def test_column_increment(self):
        splicer = MaccorSplice(self.filename_part_1, self.filename_part_2, self.output)
        meta_1, data_1 = splicer.read_maccor_file(self.filename_part_1)
        meta_2, data_2 = splicer.read_maccor_file(self.filename_part_2)
        data_1, data_2 = splicer.column_increment(data_1, data_2)

        assert data_1["Rec#"].max() < data_2["Rec#"].min()


@unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
class S3Test(unittest.TestCase):

    bucket = "beep-sync-test-stage"
    key = "test_util/test_file.txt"
    destination_path = "test_util_s3_file.txt"

    def test_download_s3_object(self):

        download_s3_object(bucket=self.bucket,
                           key=self.key,
                           destination_path=self.destination_path)

        os._exists(self.destination_path)

    def tearDown(self) -> None:
        os.remove(self.destination_path)


class TestStructuringUtils(unittest.TestCase):
    """
    Tests related to utils only used in structuring.
    """

    # based on RCRT.test_get_protocol_parameters
    def test_get_protocol_parameters(self):
        filepath = os.path.join(
            TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010"
        )
        parameters, _ = parameters_lookup.get_protocol_parameters(filepath, parameters_path=PROTOCOL_PARAMETERS_DIR)

        self.assertEqual(parameters["diagnostic_type"].iloc[0], "HPPC+RPT")
        self.assertEqual(parameters["diagnostic_parameter_set"].iloc[0], "Tesla21700")
        self.assertEqual(parameters["seq_num"].iloc[0], 109)
        self.assertEqual(len(parameters.index), 1)

        parameters_missing, project_missing = parameters_lookup.get_protocol_parameters(
            "Fake", parameters_path=PROTOCOL_PARAMETERS_DIR
        )
        self.assertEqual(parameters_missing, None)
        self.assertEqual(project_missing, None)

        filepath = os.path.join(TEST_FILE_DIR, "PreDiag_000292_tztest.010")
        parameters, _ = parameters_lookup.get_protocol_parameters(filepath, parameters_path=PROTOCOL_PARAMETERS_DIR)
        self.assertEqual(parameters["diagnostic_type"].iloc[0], "HPPC+RPT")
        self.assertEqual(parameters["seq_num"].iloc[0], 292)

    # based on RCRT.test_get_project_name
    def test_get_project_name(self):
        project_name_parts = parameters_lookup.get_project_sequence(
            os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010")
        )
        project_name = project_name_parts[0]
        self.assertEqual(project_name, "PredictionDiagnostics")

    # based on RCRT.test_get_diagnostic_parameters
    def test_get_diagnostic_parameters(self):
        diagnostic_available = {
            "parameter_set": "Tesla21700",
            "cycle_type": ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
            "length": 5,
            "diagnostic_starts_at": [1, 36, 141],
        }
        diagnostic_parameter_path = os.path.join(MODULE_DIR, "procedure_templates")
        project_name = "PreDiag"
        v_range = parameters_lookup.get_diagnostic_parameters(
            diagnostic_available, diagnostic_parameter_path, project_name
        )
        self.assertEqual(v_range, [2.7, 4.2])
