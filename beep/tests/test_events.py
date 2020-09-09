# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Events and Logging"""

import os
import unittest
import warnings
import datetime
import json
import pytz
import numpy as np
import boto3
import tempfile
from pathlib import Path
from dateutil.tz import tzutc
from beep.utils import KinesisEvents, WorkflowOutputs, Logger
from beep.utils.secrets_manager import get_secret
from beep.config import config
from beep.utils.secrets_manager import event_setup
from beep import ENVIRONMENT, __version__

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class KinesisEventsTest(unittest.TestCase):
    # Test to see if the connection to AWS Kinesis is available and only
    # run tests if list_streams() does not return an error
    try:
        kinesis = boto3.client("kinesis")
        response = kinesis.list_streams()
        stream_name = get_secret(config[ENVIRONMENT]["kinesis"]["stream"])["streamName"]
        assert "kinesis-test" == stream_name
        beep_aws_disconnected = False
    except Exception as e:
        warnings.warn("Cloud resources not configured")
        beep_aws_disconnected = True

    def setUp(self):
        pass

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_get_file_size(self):
        events = KinesisEvents(service="Testing", mode="test")
        file_list = [
            os.path.join(TEST_FILE_DIR, "2017-05-09_test-TC-contact_CH33.csv"),
            os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv"),
            os.path.join(TEST_FILE_DIR, "xTESLADIAG_000019_CH70.070"),
        ]
        file_sizes = events.get_file_size(file_list)
        print(file_sizes)
        assert file_sizes[0] == 54620
        assert file_sizes[1] == 37878198
        assert file_sizes[2] == 3019440

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_basic_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        response = events.put_basic_event("test_events", "This is a basic event test")
        assert response["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_service_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        response_valid = events.put_service_event(
            "Test", "starting", {"String": "test"}
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

        response_type_error = events.put_service_event(
            "Test", "starting", np.array([1, 2, 3])
        )
        self.assertRaises(TypeError, response_type_error)
        # Test list variable type
        response_valid = events.put_service_event(
            "Test", "starting", {"List": [1, 2, 3]}
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200
        # Test float variable type
        response_valid = events.put_service_event(
            "Test", "starting", {"Float": 1238.1231234}
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200
        # Test np.array variable type
        response_valid = events.put_service_event(
            "Test", "starting", {"Array": np.random.rand(10, 10).tolist()}
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200
        # Test dictionary variable type
        response_valid = events.put_service_event(
            "Test", "starting", {"Dict": {"key": "value"}}
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_service_event_stress(self):
        events = KinesisEvents(service="Testing", mode="test")
        for i in range(10):
            array = np.random.rand(5, 5, 3)
            print(array.tolist())
            response_valid = events.put_service_event(
                "Test", "starting", {"Stress test array": array.tolist()}
            )
            assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_upload_retrigger_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        s3_bucket = "beep-input-data"
        obj = {
            "Key": "d3Batt/raw/arbin/FastCharge_000002_CH2_Metadata.csv",
            "LastModified": datetime.datetime(2019, 4, 4, 23, 19, 20, tzinfo=tzutc()),
            "ETag": '"37677ae6b73034197d59cf3075f6fb98"',
            "Size": 615,
            "StorageClass": "STANDARD",
            "Owner": {
                "DisplayName": "it-admin+materials-admin",
                "ID": "02d8b24e2f66c2b5937f391b7c87406d4eeab68cf887bd9933d6631536959f24",
            },
        }
        retrigger_data = {
            "filename": obj["Key"],
            "bucket": s3_bucket,
            "size": obj["Size"],
            "hash": obj["ETag"].strip('"'),
        }
        response_valid = events.put_upload_retrigger_event("complete", retrigger_data)
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_validation_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        file_list = [os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv")]
        file_list_data = {"run_list": [24]}
        validity = ["valid"]
        messages = [{"comment": "", "error": ""}]
        output_json = {
            "file_list": file_list,
            "run_list": file_list_data["run_list"],
            "validity": validity,
            "message_list": messages,
        }
        response_valid = events.put_validation_event(output_json, "complete")
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_structuring_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        processed_file_list = [
            os.path.join(TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_structure.json")
        ]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{"comment": "", "error": ""}]
        invalid_file_list = []
        output_json = {
            "file_list": processed_file_list,
            "run_list": processed_run_list,
            "result_list": processed_result_list,
            "message_list": processed_message_list,
            "invalid_file_list": invalid_file_list,
        }

        response_valid = events.put_structuring_event(output_json, "complete")
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_analyzing_event(self):
        events = KinesisEvents(service="Testing", mode="test")
        processed_paths_list = [
            os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_features.json")
        ]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{"comment": "", "error": ""}]

        output_data = {
            "file_list": processed_paths_list,
            "run_list": processed_run_list,
            "result_list": processed_result_list,
            "message_list": processed_message_list,
        }

        response_valid = events.put_analyzing_event(
            output_data, "featurizing", "complete"
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

        processed_paths_list = [
            os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_predictions.json")
        ]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{"comment": "", "error": ""}]

        output_data = {
            "file_list": processed_paths_list,
            "run_list": processed_run_list,
            "result_list": processed_result_list,
            "message_list": processed_message_list,
        }

        response_valid = events.put_analyzing_event(
            output_data, "predicting", "complete"
        )
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200

    @unittest.skipIf(beep_aws_disconnected, "Unable to connect to Kinesis")
    def test_kinesis_put_generate_event(self):
        events = KinesisEvents(service="Testing", mode="test")

        all_output_files = [
            "/data-share/protocols/procedures/name_000000.000",
            "/data-share/protocols/procedures/name_000007.000",
            "/data-share/protocols/procedures/name_000014.000",
        ]
        result = "success"
        message = {"comment": "", "error": ""}

        output_data = {
            "file_list": all_output_files,
            "result": result,
            "message": message,
        }

        response_valid = events.put_generate_event(output_data, "complete")
        assert response_valid["ResponseMetadata"]["HTTPStatusCode"] == 200


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
        parameter_file_size = 138

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
