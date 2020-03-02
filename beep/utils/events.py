# Copyright 2019 Toyota Research Institute. All rights reserved.
""" Logging and Event utils"""

import logging
import sys
import os
import datetime
import json
import base64

import watchtower
import numpy as np
import boto3
import pytz
from beep import LOG_DIR


class Logger:
    """
    Logger utility. Provides a thin wrapper over the builtin
    python logging utility.

    Attributes:
        log_level (str): logging level.
        service (str): service name.
        _terminal_logger: logger instance that prints to console.
    """

    def __init__(self, log_level='INFO', log_file=None, log_cloudwatch=False, service='Testing'):
        self.service = service
        self._terminal_logger = setup_logger(log_level,
                                             log_file=log_file,
                                             log_cloudwatch=log_cloudwatch,
                                             service=service)

    def info(self, *args):
        self._terminal_logger.info(args)

    def error(self, *args):
        self._terminal_logger.error(args)

    def warning(self, *args):
        self._terminal_logger.warning(args)

    def critical(self, *args):
        self._terminal_logger.critical(args)


class KinesisEvents:
    """
    Attributes:
        service (str): default: Which service is instantiating the class so that
        we know who is sending events. Defaults to 'Testing'.
        mode (str): Mode for events, test will only put test message. Defaults to 'run'.
    """

    def __init__(self,
                 service='Testing',
                 mode='run'
                 ):

        self.service = service
        self.mode = mode

        if self.mode == 'run':
            self.stream = 'beep-events'
            self.kinesis = boto3.client('kinesis', region_name='us-west-2')

        if self.mode == 'test':
            self.stream = 'kinesis-test'
            self.kinesis = boto3.client('kinesis', region_name='us-west-2')

        if self.mode == 'events_off':
            self.logger = Logger(log_file=os.path.join(LOG_DIR, "Event_logger.log"))

    def get_file_size(self, file_list):
        """
        Basic function to get file size from a list.

        Args:
            file_list (list): List of full file names (paths).
        """
        file_sizes = []
        for file in file_list:
            file_sizes.append(os.path.getsize(file))
        return file_sizes

    def put_basic_event(self, module_name, record):
        """
        Basic function to put events into the Kinesis stream.
        No filtering or parsing, not for general use

        Args:
            module_name (str): Name of the module or script producing the event.
            record (blob): Data blob to be written to the Kinesis stream. Under 1MB.
        """

        if self.mode == 'test':
            response = self.kinesis.put_record(StreamName=self.stream,
                                               Data=record,
                                               PartitionKey=str(hash('test'))
                                               )
        elif self.mode == 'events_off':
            self.logger.warning(dict(level="WARNING",
                                     details={
                                         "mode": self.mode,
                                         "datetime": datetime.datetime.now(pytz.utc).isoformat(),
                                         "Data": record,
                                         "PartitionKey": str(hash('test'))
                                     }
                                     ))
            response = None

        else:
            response = self.kinesis.put_record(StreamName=self.stream,
                                               Data=record,
                                               PartitionKey=str(hash(module_name))
                                               )
        return response

    def put_service_event(self, action, status, data):
        """
        Function to put service events into the Kinesis stream. For each
        event the timestamp of the event and the project_id of the event class
        is added prepended to the json string

        To decode the data
        data = json.loads(base64.b64decode(data_base64.encode('utf-8')).decode('utf-8'))

        For data such as np.arrays please create a key
        and convert the value with .tolist() i.e. {"my_array": my_array.tolist()}
        Use np.array() to convert back to np.array

        Args:
            action (str): Action being taken by the service. For example, the
                Data Syncer might 'sync out'.
            status (str): Status of the action being taken by the service. For
                example, during 'sync out' the status might be 'writing to DB'.

            data (dict): Data payload for the event. Supported variable types
                in the dict string, int, float, list, dict.
        """
        # Attempt to encode the data into a base64 format and return the type error
        # if the data type is not supported
        try:
            data_base64 = base64.b64encode(json.dumps(data).encode('utf-8')).decode('utf-8')
        except TypeError as error:
            print(error)
            return error

        # Create the dict that contains the standard format for all services in beep
        record = {
            "timestamp": datetime.datetime.now(pytz.utc).isoformat(),  # Example: '2019-02-27T17:37:40.626564+00:00'
            "service": self.service,  # Example 'DataSyncer'
            "action": action,  # Example 'sync out'
            "status": status,  # Example 'writing to database'
            "data": json.dumps(data)
        }

        if self.mode == 'test':
            response = self.kinesis.put_record(StreamName=self.stream,
                                               Data=json.dumps(record),
                                               PartitionKey=str(hash('test'))
                                               )

        elif self.mode == 'events_off':
            self.logger.warning(dict(level="WARNING",
                                     details={
                                         "mode": self.mode,
                                         "datetime": datetime.datetime.now(pytz.utc).isoformat(),
                                         "Data": record,
                                         "PartitionKey": str(hash('test'))
                                     }
                                     ))
            response = None

        else:
            response = self.kinesis.put_record(StreamName=self.stream,
                                               Data=json.dumps(record),
                                               PartitionKey=str(hash(self.service))
                                               )
        return response

    def put_upload_retrigger_event(self, upload_status, retrigger_data):
        """
        Function to put upload events into the Kinesis stream. Function should be called
        after each file is validated. Validation data should include result of validation,
        file name and run_id.

        Args:
            upload_status (str): status for uploading event.
            retrigger_data  (dict): data payload for the event.
        """

        response = self.put_service_event('upload', upload_status, retrigger_data)

        return response

    def put_validation_event(self, output_data, validator_status):
        """
        Function to put validation events into the Kinesis stream. Function should be called
        after each file is validated. Validation data should include result of validation,
        file name and run_id.

        Args:
            output_data (dict): data payload for the event.
            validator_status (str): starting|complete|error.
        """
        files = []
        size_list = self.get_file_size(output_data['file_list'])

        for indx, file in enumerate(output_data['file_list']):
            obj_json = {
                "filename": output_data['file_list'][indx],
                "size": size_list[indx],
                "run_id": output_data['run_list'][indx],
                "result": output_data['validity'][indx],
                "message": output_data['message_list'][indx]
            }
            files.append(obj_json)

        data = {
            "files": files
        }

        response = self.put_service_event('validating', validator_status, data)

        return response

    def put_structuring_event(self, output_data, structure_status):
        """
        Function to put structuring events into the Kinesis stream. Function should be called
        after each file is structured. Data should include result of structuring,
        file name and run_id.

        Args:
            output_data (dict): data payload for the event.
            structure_status (str): starting|complete|error
        """
        files = []
        size_list = self.get_file_size(output_data['file_list'])

        for indx, file in enumerate(output_data['file_list']):
            obj_json = {
                "filename": output_data['file_list'][indx],
                "size": size_list[indx],
                "run_id": output_data['run_list'][indx],
                "result": output_data['result_list'][indx],
                "message": output_data['message_list'][indx]
            }
            files.append(obj_json)

        data = {
            "files": files
        }

        response = self.put_service_event('structuring', structure_status, data)

        return response

    def put_analyzing_event(self, output_data, analyzer_action, analyzer_status):
        """
        Function to put analyzing events into the Kinesis stream. Function should be called
        after each file is featurized and fitted. Data field should include result of featurizing/fitting,
        file name and run_id.

        Args:
            output_data (dict): data payload for the event.
            analyzer_action (str): featurizing|predicting|fitting.
            analyzer_status (str): starting|complete|error.
        """
        files = []
        size_list = self.get_file_size(output_data['file_list'])

        if analyzer_action == 'fitting':
            for indx, file in enumerate(output_data['file_list']):
                obj_json = {
                    "filename": output_data['file_list'][indx],
                    "size": size_list[indx],
                    "run_id": output_data['run_list'][indx],
                    "message": output_data['message_list'][indx]
                }
                files.append(obj_json)
            data = {
                "files": files,
                "model": output_data['model'],
                "result": output_data['result'],
                "message": output_data['model_message']
            }

        else:
            for indx, file in enumerate(output_data['file_list']):
                obj_json = {
                    "filename": output_data['file_list'][indx],
                    "size": size_list[indx],
                    "run_id": output_data['run_list'][indx],
                    "result": output_data['result_list'][indx],
                    "message": output_data['message_list'][indx]
                }
                files.append(obj_json)

            data = {
                "files": files
            }

        response = self.put_service_event(analyzer_action, analyzer_status, data)

        return response

    def put_generate_event(self, output_data, generate_status):
        """
        Function to put structuring events into the Kinesis stream. Function should be called
        after each file is structured. Data should include result of structuring,
        file name and run_id.

        Args:
            output_data (dict): data payload for the event.
            generate_status (str): starting|complete|error.
        """

        data = {
            "files": output_data['file_list'],
            "result": output_data['result'],
            "message": output_data['message']
        }

        response = self.put_service_event('generating_protocol', generate_status, data)

        return response


def setup_logger(log_level='INFO', log_file=None, log_cloudwatch=False,
                 service='Testing', np_precision=3):
    """
    Creates and configures a logger object.

    Args:
        log_level (str): the logging level. Defaults to 'INFO'.
        log_file (str): optional local log_file. Defaults to None.
        log_cloudwatch (bool): Will log to CloudWatch topic if set to True.
            Defaults to False.
        service (str): service name .
        np_precision (int): numpy precision.

    Returns:
        logging.Logger: python Logger object.
    """

    # numpy float precisions when printing
    np.set_printoptions(precision=np_precision)

    # Python logger config
    logger = logging.getLogger(__name__ + '.' + service)  # '' or None is the root logger

    # Remove all previous filters and handlers
    logger.handlers = []
    logger.filters = []

    # Get handler on log
    if log_file is not None:
        hdlr = logging.FileHandler(log_file, 'a')
    else:
        hdlr = logging.StreamHandler(sys.stdout)

    logger.addHandler(hdlr)

    # Add cloudwatch logging if requested
    if log_cloudwatch:
        hdlr = watchtower.CloudWatchLogHandler()

    logger.addHandler(hdlr)

    fmt_str = '# {cols[y]}%(asctime)s{cols[reset]}'
    fmt_str += " %(levelname)-8s"
    fmt_str += " {cols[c]}%(funcName)-3s{cols[reset]} %(message)s"

    logger.setLevel(1 if log_level == 'ALL' else getattr(logging, log_level))
    logger.propagate = False
    return logger
