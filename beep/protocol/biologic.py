# Copyright 2019 Toyota Research Institute. All rights reserved.
"""
Module for generating maccor procedure files from
input parameters and procedure templates
"""

import os
import warnings
import json
import time
import datetime
import csv
import re
from collections import OrderedDict
from copy import deepcopy

from beep import logger, __version__
from beep.protocol import PROCEDURE_TEMPLATE_DIR

from beep.utils import KinesisEvents, DashOrderedDict
s = {'service': 'ProtocolGenerator'}


class Settings(DashOrderedDict):
    """
    Settings file object. Provides factory methods
    to read a Biologic-type settings file and invoke
    from templates for specific experimental
     parameters

    """
    @classmethod
    def from_file(cls, filename, encoding='ISO-8859-1', column_width=20):
        """
        Settings file ingestion. Invokes Settings object
        from standard Biologic *.mps file.

        Args:
            filename (str): settings file.
            encoding (str): file encoding to use when reading the file
            column_width (int): number of characters per step column

        Returns:
            (Settings): Ordered dictionary with keys corresponding to options or
                control variables. Section headers are nested dicts or lists
                within the dict.
        """
        tq_offset = 2
        tq_length = 62
        obj = cls()
        with open(filename, 'rb') as f:
            text = f.read()
            text = text.decode(encoding)
        split_text = re.split(r'\r\n', text)
        number_of_columns = max([len(l) for l in split_text]) // column_width

        section = 'Technique'
        technique_lines = [indx for indx, val in enumerate(split_text) if 'Technique' in val]
        for technique_start_line in technique_lines:
            technique = split_text[technique_start_line].split(':')[-1].strip()
            start = technique_start_line + tq_offset
            end = start + tq_length
            technique_steps = []
            for line in split_text[start:end]:
                steps_values = []
                for col in range(number_of_columns):
                    steps_values.append(line[col*column_width:(col+1)*column_width].strip())
                technique_steps.append(steps_values)
            step_matrix = list(zip(*technique_steps))
            step_headers = step_matrix[0]

            for step_number in range(1, number_of_columns):
                step = OrderedDict(zip(step_headers, step_matrix[step_number]))
                obj.set('{}.{}.{}'.format(section, technique, 'Step' + str(step_number)), step)

        return obj
