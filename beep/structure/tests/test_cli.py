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
"""Unit tests related to cycler run data structures"""

import os
import unittest

from beep.structure.arbin import ArbinDatapath
from beep.structure.cli import auto_load, auto_load_processed
from beep.tests.constants import TEST_FILE_DIR


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.arbin_file = os.path.join(
            TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv"
        )

        self.processed_file = os.path.join(
            TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_structure.json"
        )

        self.processed_maccor_file = os.path.join(
            TEST_FILE_DIR, "PredictionDiagnostics_000132_00004C_structure.json"
        )

    # todo: could be more comprehensive
    # based on PCRT.test_auto_load
    def test_auto_load(self):
        dp = auto_load(self.arbin_file)
        self.assertIsInstance(dp, ArbinDatapath)

    def test_auto_load_processed(self):
        dp = auto_load_processed(self.processed_file)
        self.assertIsInstance(dp, ArbinDatapath)
        self.assertIsNotNone(dp.structured_summary)

    def test_auto_load_maccor(self):
        dp = auto_load_processed(self.processed_maccor_file)
        self.assertEqual(dp.paths["structured"], self.processed_maccor_file)



if __name__ == "__main__":
    unittest.main()
