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

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir

from beep.utils import os_format
from beep.structure.base import BEEPDatapath
from beep.structure.arbin import ArbinDatapath
from beep.structure.cli import auto_load
from beep.tests.constants import TEST_FILE_DIR


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.arbin_file = os.path.join(
            TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv"
        )

    # based on CliTest.test_simple_conversion
    def test_structure_command_simple_conversion(self):
        with ScratchDir("."):
            # Set root env
            os.environ["BEEP_PROCESSING_DIR"] = os.getcwd()
            # Make necessary directories
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "structure"))
            # Create dummy json obj
            json_obj = {
                "file_list": [self.arbin_file],
                "run_list": [0],
                "validity": ["valid"],
            }
            json_string = json.dumps(json_obj)

            command = "structure {}".format(os_format(json_string))
            result = subprocess.check_call(command, shell=True)
            self.assertEqual(result, 0)
            # print(os.listdir(os.path.join("data-share", "structure")))
            processed = loadfn(
                os.path.join(
                    "data-share",
                    "structure",
                    "2017-12-04_4_65C-69per_6C_CH29_structure.json",
                )
            )

        self.assertIsInstance(processed, BEEPDatapath)

    # todo: could be more comprehensive
    # based on PCRT.test_auto_load
    def test_auto_load(self):
        dp = auto_load(self.arbin_file)
        self.assertIsInstance(dp, ArbinDatapath)


if __name__ == "__main__":
    unittest.main()
