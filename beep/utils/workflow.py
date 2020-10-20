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
""" Logging and Event utils"""

import logging
import sys
import os
import json
import tempfile
import watchtower
import numpy as np
from pathlib import Path


class Logger:
    """
    Logger utility. Provides a thin wrapper over the builtin
    python logging utility.

    Attributes:
        log_level (str): logging level.
        service (str): service name.
        _terminal_logger: logger instance that prints to console.
    """

    def __init__(
        self, log_level="INFO", log_file=None, log_cloudwatch=False, service="Testing"
    ):
        self.service = service
        self._terminal_logger = setup_logger(
            log_level, log_file=log_file, log_cloudwatch=log_cloudwatch, service=service
        )

    def info(self, *args):
        self._terminal_logger.info(args)

    def error(self, *args):
        self._terminal_logger.error(args)

    def warning(self, *args):
        self._terminal_logger.warning(args)

    def critical(self, *args):
        self._terminal_logger.critical(args)


class WorkflowOutputs:
    """
    Supports writing outputs to local file system
    """

    def get_local_file_size(self, filename):
        """
        Basic function get the local file size. Returns -1 for non-existent files

        Parameters
        ----------
        filename: str
            Full name of the file (path).
        """
        try:
            file_size = os.path.getsize(filename)
        except FileNotFoundError:
            file_size = -1
        return file_size

    def split_workflow_outputs(self, path_str, result):
        """
        Function to split workflow outputs into individual files on local file system.

        Args:
            path (str): outputs base file path
            result (dict): single processing result data
        """

        path = Path(path_str)

        filename_path = path / "filename.txt"
        size_path = path / "size.txt"
        run_id_path = path / "run_id.txt"
        action_path = path / "action.txt"
        status_path = path / "status.txt"

        filename_path.write_text(result["filename"])
        size_path.write_text(str(result["size"]))
        run_id_path.write_text(str(result["run_id"]))
        action_path.write_text(result["action"])
        status_path.write_text(result["status"])

    def put_generate_outputs_list(self, output_data, generate_status):
        """
        Function create generate outputs list json file on local file system.

        Args:
            output_data (dict): data payload for output.
            generate_status (str): starting|complete|error.
        """

        tmp_dir = Path(tempfile.gettempdir())
        output_file_path = tmp_dir / "results.json"
        results = []

        file_list = output_data["file_list"]

        for index, filename in enumerate(file_list):
            size = self.get_local_file_size(filename)
            if size < 0:
                raise FileNotFoundError()

            result = {
                "filename": filename,
                "size": size,
                "result": output_data["result"],
                "status": generate_status,
            }

            results.append(result)

        output_file_path.write_text(json.dumps(results))

    def put_workflow_outputs(self, output_data, action):
        """
        Function to create workflow outputs json file on local file system.

        Args:
            output_data (dict): single processing result data
            action (str): workflow action
        """

        tmp_dir = Path(tempfile.gettempdir())
        output_file_path = tmp_dir / "results.json"

        # Most operating systems should have a temp directory
        if not tmp_dir.exists:
            try:
                tmp_dir.mkdir()
            except OSError:
                print("creation of temp directory failed")

        size = self.get_local_file_size(output_data["filename"])
        if size < 0:
            raise FileNotFoundError()

        result = {
            "filename": output_data["filename"],
            "size": size,
            "run_id": output_data["run_id"],
            "action": action,
            "status": output_data["result"],
        }

        # Argo limitations require outputting each value separately
        self.split_workflow_outputs(str(tmp_dir), result)

        output_file_path.write_text(json.dumps(result))

    def put_workflow_outputs_list(self, output_data, action):
        """
        Function to create workflow outputs list json file on local file system.

        Args:
            output_data (list[dict]): processing result data
            action (str): workflow action
        """

        tmp_dir = Path(tempfile.gettempdir())
        output_file_path = tmp_dir / "results.json"
        results = []

        # Most operating systems should have a temp directory
        if not tmp_dir.exists:
            try:
                tmp_dir.mkdir()
            except OSError:
                print("creation of temp directory failed")

        file_list = output_data["file_list"]

        for index, filename in enumerate(file_list):
            size = self.get_local_file_size(filename)
            if size < 0:
                raise FileNotFoundError()

            result = {
                "filename": filename,
                "size": size,
                "run_id": output_data["run_list"][index],
                "action": action,
                "status": output_data["result_list"][index],
            }

            results.append(result)

        output_file_path.write_text(json.dumps(results))


def setup_logger(
    log_level="INFO",
    log_file=None,
    log_cloudwatch=False,
    service="Testing",
    np_precision=3,
):
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
    logger = logging.getLogger(
        __name__ + "." + service
    )  # '' or None is the root logger

    # Remove all previous filters and handlers
    logger.handlers = []
    logger.filters = []

    # Get handler on log
    if log_file is not None:
        hdlr = logging.FileHandler(log_file, "a")
    else:
        hdlr = logging.StreamHandler(sys.stdout)

    logger.addHandler(hdlr)

    # Add cloudwatch logging if requested
    if log_cloudwatch:
        hdlr = watchtower.CloudWatchLogHandler()

    logger.addHandler(hdlr)

    fmt_str = "# {cols[y]}%(asctime)s{cols[reset]}"
    fmt_str += " %(levelname)-8s"
    fmt_str += " {cols[c]}%(funcName)-3s{cols[reset]} %(message)s"

    logger.setLevel(1 if log_level == "ALL" else getattr(logging, log_level))
    logger.propagate = False
    return logger
