# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Events and Logging"""

import os
import unittest
import datetime
import pytz
import numpy as np
import boto3
from dateutil.tz import tzutc
from botocore.exceptions import NoRegionError, NoCredentialsError
from beep.utils import KinesisEvents, Logger
from beep.utils.secrets_manager import get_secret
from beep.config import config
from beep import ENVIRONMENT, __version__

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class KinesisEventsTest(unittest.TestCase):
    # Test to see if the connection to AWS Kinesis is available and only
    # run tests if list_streams() does not return an error
    try:
        kinesis = boto3.client('kinesis')
        response = kinesis.list_streams()
        print(response)
        beep_kinesis_connection_broken = False
    except NoRegionError or NoCredentialsError as e:
        beep_kinesis_connection_broken = True

    def setUp(self):
        try:
            stream_name = get_secret(config[ENVIRONMENT]['kinesis']['stream'])['streamName']
            self.assertEqual('kinesis-test', stream_name)
        except NoRegionError or NoCredentialsError as e:
            beep_secrets_connection_broken = True

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_get_file_size(self):
        events = KinesisEvents(service='Testing', mode='test')
        file_list = [os.path.join(TEST_FILE_DIR, "2017-05-09_test-TC-contact_CH33.csv"),
                     os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv"),
                     os.path.join(TEST_FILE_DIR, "xTESLADIAG_000019_CH70.070")]
        file_sizes = events.get_file_size(file_list)
        print(file_sizes)
        assert file_sizes[0] == 54620
        assert file_sizes[1] == 37878198
        assert file_sizes[2] == 3019440

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_basic_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        response = events.put_basic_event('test_events', 'This is a basic event test')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_service_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        response_valid = events.put_service_event('Test', 'starting', {"String": "test"})
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

        response_type_error = events.put_service_event('Test', 'starting', np.array([1, 2, 3]))
        self.assertRaises(TypeError, response_type_error)
        # Test list variable type
        response_valid = events.put_service_event('Test', 'starting', {"List": [1, 2, 3]})
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200
        # Test float variable type
        response_valid = events.put_service_event('Test', 'starting', {"Float": 1238.1231234})
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200
        # Test np.array variable type
        response_valid = events.put_service_event('Test', 'starting', {"Array": np.random.rand(10, 10).tolist()})
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200
        # Test dictionary variable type
        response_valid = events.put_service_event('Test', 'starting', {"Dict": {"key": "value"}})
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_service_event_stress(self):
        events = KinesisEvents(service='Testing', mode='test')
        for i in range(10):
            array = np.random.rand(5, 5, 3)
            print(array.tolist())
            response_valid = events.put_service_event('Test', 'starting', {"Stress test array": array.tolist()})
            assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_upload_retrigger_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        s3_bucket = "beep-input-data"
        obj = {
            'Key': 'd3Batt/raw/arbin/FastCharge_000002_CH2_Metadata.csv',
            'LastModified': datetime.datetime(2019, 4, 4, 23, 19, 20, tzinfo=tzutc()),
            'ETag': '"37677ae6b73034197d59cf3075f6fb98"',
            'Size': 615,
            'StorageClass': 'STANDARD',
            'Owner': {
                   'DisplayName': 'it-admin+materials-admin',
                   'ID': '02d8b24e2f66c2b5937f391b7c87406d4eeab68cf887bd9933d6631536959f24'
                    }
              }
        retrigger_data = {
            "filename": obj['Key'],
            "bucket": s3_bucket,
            "size": obj['Size'],
            "hash": obj["ETag"].strip('\"')
        }
        response_valid = events.put_upload_retrigger_event('complete', retrigger_data)
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_validation_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        file_list = [os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv")]
        file_list_data = {"run_list": [24]}
        validity = ["valid"]
        messages = [{'comment': '',
                     'error': ''}]
        output_json = {'file_list': file_list, 'run_list': file_list_data['run_list'],
                       'validity': validity, 'message_list': messages}
        response_valid = events.put_validation_event(output_json, 'complete')
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_structuring_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        processed_file_list = [os.path.join(TEST_FILE_DIR, "2017-06-30_2C-10per_6C_CH10_structure.json")]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{'comment': '',
                                   'error': ''}]
        invalid_file_list = []
        output_json = {"file_list": processed_file_list,
                       "run_list": processed_run_list,
                       "result_list": processed_result_list,
                       "message_list": processed_message_list,
                       "invalid_file_list": invalid_file_list}

        response_valid = events.put_structuring_event(output_json, 'complete')
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_analyzing_event(self):
        events = KinesisEvents(service='Testing', mode='test')
        processed_paths_list = [os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_features.json")]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{'comment': '',
                                   'error': ''}]

        output_data = {"file_list": processed_paths_list,
                       "run_list": processed_run_list,
                       "result_list": processed_result_list,
                       "message_list": processed_message_list
                       }

        response_valid = events.put_analyzing_event(output_data, 'featurizing', 'complete')
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

        processed_paths_list = [os.path.join(TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_predictions.json")]
        processed_run_list = [24]
        processed_result_list = ["success"]
        processed_message_list = [{'comment': '',
                                   'error': ''}]

        output_data = {"file_list": processed_paths_list,
                       "run_list": processed_run_list,
                       "result_list": processed_result_list,
                       "message_list": processed_message_list
                       }

        response_valid = events.put_analyzing_event(output_data, 'predicting', 'complete')
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200

    @unittest.skipIf(beep_kinesis_connection_broken, "Unable to connect to Kinesis")
    def test_kinesis_put_generate_event(self):
        events = KinesisEvents(service='Testing', mode='test')

        all_output_files = \
            ['/data-share/protocols/procedures/name_000000.000',
             '/data-share/protocols/procedures/name_000007.000',
             '/data-share/protocols/procedures/name_000014.000']
        result = 'success'
        message = {'comment': '',
                   'error': ''}

        output_data = {"file_list": all_output_files,
                       "result": result,
                       "message": message
                       }

        response_valid = events.put_generate_event(output_data, 'complete')
        assert response_valid['ResponseMetadata']['HTTPStatusCode'] == 200


class CloudWatchLoggingTest(unittest.TestCase):
    # Test to see if the connection to AWS Cloudwatch is available and only
    # run tests if describe_alarms() does not return an error
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.describe_alarms()
        beep_cloudwatch_connection_broken = False
    except NoRegionError or NoCredentialsError as e:
        beep_cloudwatch_connection_broken = True

    def setUp(self):
        pass

    @unittest.skipUnless(ENVIRONMENT == "test", "Non-test environment")
    def test_versioning(self):
        # Ensure something is appended to version
        self.assertGreater(len(__version__), 11)

    def test_log(self):
        logger = Logger(log_file=os.path.join(TEST_FILE_DIR, "Testing_logger.log"))
        logger.info('Hi! Your local test log is working')

    @unittest.skipIf(beep_cloudwatch_connection_broken, "Unable to connect to Cloudwatch")
    def test_cloudwatch_log(self):
        logger = Logger(log_file=os.path.join(TEST_FILE_DIR, "Testing_logger.log"),
                        log_cloudwatch=True)
        logger.info(dict(level="INFO",
                         details={datetime.datetime.now(pytz.utc).isoformat()}))
        logger.error(dict(level="ERROR",
                          details={datetime.datetime.now(pytz.utc).isoformat()}))
        logger.warning(dict(level="WARNING",
                            details={datetime.datetime.now(pytz.utc).isoformat()}))
        logger.critical(dict(level="CRITICAL",
                             details={datetime.datetime.now(pytz.utc).isoformat()}))
