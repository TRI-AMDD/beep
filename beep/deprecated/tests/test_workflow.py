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
"""Unit tests related to Events and Logging"""

import os
import unittest
import warnings
import datetime
import json
import pytz
import boto3
import tempfile
from platform import system
from pathlib import Path
from beep.utils import WorkflowOutputs, Logger
from beep import ENVIRONMENT, __version__

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class WorkflowOutputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs = WorkflowOutputs()
        self.tmp_dir = Path(tempfile.gettempdir())

        self.filename_path = self.tmp_dir / "filename.txt"
        self.size_path = self.tmp_dir / "size.txt"
        self.run_id_path = self.tmp_dir / "run_id.txt"
        self.action_path = self.tmp_dir / "action.txt"
        self.status_path = self.tmp_dir / "status.txt"

        self.output_data = {
            "filename": str(Path(TEST_FILE_DIR) / "2017-12-04_4_65C-69per_6C_CH29.csv"),
            "run_id": 123,
            "result": "valid",
        }

        self.test_file_size = os.path.getsize(str(self.output_data["filename"]))

        self.output_data_list = {
            "file_list": [
                str(Path(TEST_FILE_DIR) / "2017-08-14_8C-5per_3_47C_CH44.csv"),
                str(Path(TEST_FILE_DIR) / "2017-12-04_4_65C-69per_6C_CH29.csv"),
            ],
            "run_list": [123, 456],
            "result_list": ["valid", "invalid"],
        }

        self.action = "structuring"

        self.output_file_path = self.tmp_dir / "results.json"

    def tearDown(self):
        if self.output_file_path.exists():
            self.output_file_path.unlink()

        if self.filename_path.exists():
            self.filename_path.unlink()

        if self.size_path.exists():
            self.size_path.unlink()

        if self.run_id_path.exists():
            self.run_id_path.unlink()

        if self.action_path.exists():
            self.action_path.unlink()

        if self.status_path.exists():
            self.status_path.unlink()

    def test_get_local_file_size(self):
        file_size = self.outputs.get_local_file_size(str(self.output_data["filename"]))

        self.assertEqual(self.test_file_size, file_size)

    def test_put_workflow_outputs(self):
        self.outputs.put_workflow_outputs(self.output_data, self.action)

        self.assertTrue(self.output_file_path.exists())
        output_json = json.loads(self.output_file_path.read_text())

        self.assertEqual(self.output_data["filename"], output_json["filename"])
        self.assertEqual(self.test_file_size, output_json["size"])
        self.assertEqual(self.output_data["run_id"], output_json["run_id"])
        self.assertEqual(self.action, output_json["action"])
        self.assertEqual(self.output_data["result"], output_json["status"])

        # Split outputs
        self.assertTrue(self.filename_path.exists())
        self.assertTrue(self.size_path.exists())
        self.assertTrue(self.run_id_path.exists())
        self.assertTrue(self.action_path.exists())
        self.assertTrue(self.status_path.exists())

        self.assertEqual(self.output_data["filename"], self.filename_path.read_text())
        self.assertEqual(self.test_file_size, int(self.size_path.read_text()))
        self.assertEqual(self.output_data["run_id"], int(self.run_id_path.read_text()))
        self.assertEqual(self.action, self.action_path.read_text())
        self.assertEqual(self.output_data["result"], self.status_path.read_text())

    def test_split_workflow_outputs(self):
        result = {
            "filename": self.output_data["filename"],
            "size": self.test_file_size,
            "run_id": self.output_data["run_id"],
            "action": self.action,
            "status": self.output_data["result"],
        }

        self.outputs.split_workflow_outputs(self.tmp_dir, result)

        self.assertTrue(self.filename_path.exists())
        self.assertTrue(self.size_path.exists())
        self.assertTrue(self.run_id_path.exists())
        self.assertTrue(self.action_path.exists())
        self.assertTrue(self.status_path.exists())

        self.assertEqual(self.output_data["filename"], self.filename_path.read_text())
        self.assertEqual(self.test_file_size, int(self.size_path.read_text()))
        self.assertEqual(self.output_data["run_id"], int(self.run_id_path.read_text()))
        self.assertEqual(self.action, self.action_path.read_text())
        self.assertEqual(self.output_data["result"], self.status_path.read_text())

    def test_put_generate_outputs_list(self):
        result = "success"
        status = "complete"
        parameter_file_size = 142 if system() == "Windows" else 138

        all_output_files = [
            str(Path(TEST_FILE_DIR) / "data-share/protocols/procedures/name_000000.000"),
            str(Path(TEST_FILE_DIR) / "data-share/protocols/procedures/name_000007.000"),
            str(Path(TEST_FILE_DIR) / "data-share/protocols/procedures/name_000014.000"),
        ]

        output_data = {
            "file_list": all_output_files,
            "result": result,
            "message": "",
        }

        self.outputs.put_generate_outputs_list(output_data, status)

        self.assertTrue(self.output_file_path.exists())
        output_list = json.loads(self.output_file_path.read_text())
        output_json = output_list[1]

        self.assertEqual(all_output_files[1], output_json["filename"])
        self.assertEqual(parameter_file_size, output_json["size"])
        self.assertEqual(result, output_json["result"])
        self.assertEqual(status, output_json["status"])

    def test_put_generate_output_list_no_file(self):
        result = "success"
        status = "complete"

        all_output_files = [
            str(Path(TEST_FILE_DIR) / "data-share/protocols/procedures/not_a_real_file"),
        ]

        output_data = {
            "file_list": all_output_files,
            "result": result,
            "message": "",
        }

        self.assertRaises(FileNotFoundError, self.outputs.put_generate_outputs_list, output_data, status)

    def test_put_workflow_outputs_list(self):
        self.outputs.put_workflow_outputs_list(self.output_data_list, self.action)

        self.assertTrue(self.output_file_path.exists())
        output_list = json.loads(self.output_file_path.read_text())
        output_json = output_list[1]

        self.assertEqual(self.output_data_list["file_list"][1], output_json["filename"])
        self.assertEqual(self.test_file_size, output_json["size"])
        self.assertEqual(self.output_data_list["run_list"][1], output_json["run_id"])
        self.assertEqual(self.action, output_json["action"])
        self.assertEqual(self.output_data_list["result_list"][1], output_json["status"])


class CloudWatchLoggingTest(unittest.TestCase):
    # Test to see if the connection to AWS Cloudwatch is available and only
    # run tests if describe_alarms() does not return an error
    try:
        cloudwatch = boto3.client("cloudwatch")
        cloudwatch.describe_alarms()
        beep_cloudwatch_connection_broken = False
        print("here")
    except Exception as e:
        warnings.warn("Cloud resources not configured")
        beep_cloudwatch_connection_broken = True

    def setUp(self):
        pass

    @unittest.skipUnless(ENVIRONMENT == "test", "Non-test environment")
    def test_versioning(self):
        # Ensure something is appended to version
        self.assertGreater(len(__version__), 11)

    def test_log(self):
        logger = Logger(
            log_file=os.path.join(TEST_FILE_DIR, "Testing_logger.log"),
            log_cloudwatch=False,
        )
        logger.info("Hi! Your local test log is working")

    @unittest.skipIf(
        beep_cloudwatch_connection_broken, "Unable to connect to Cloudwatch"
    )
    def test_cloudwatch_log(self):
        logger = Logger(
            log_file=os.path.join(TEST_FILE_DIR, "Testing_logger.log"),
            log_cloudwatch=True,
        )
        logger.info(
            dict(level="INFO", details={datetime.datetime.now(pytz.utc).isoformat()})
        )
        logger.error(
            dict(level="ERROR", details={datetime.datetime.now(pytz.utc).isoformat()})
        )
        logger.warning(
            dict(level="WARNING", details={datetime.datetime.now(pytz.utc).isoformat()})
        )
        logger.critical(
            dict(
                level="CRITICAL", details={datetime.datetime.now(pytz.utc).isoformat()}
            )
        )
