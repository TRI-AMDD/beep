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
"""
Module and script for validating battery cycler flat files using dictionary
key or DataFrame based validation of typing, min/max, and non-allowed values

Usage:
    validate [INPUT_JSON]

Options:
    -h --help       Show this screen
    --version       Show version

The validation script, `validate`, runs the validation procedure contained
in beep.validate on renamed files according to the output of `collate`.
It also updates a general json validation record in `/data-share/validation/validation.json`.

The input json must contain the following fields

* `file_list` - the list of filenames to be validated

The output json will have the following fields:

* `validity` - a list of boolean validation results, e. g. `[True, True, False]`
* `file_list` - a list of full path filenames which have been processed

Example:
$ validate '{"fid": [0, 1, 2], "strname": ["2017-05-09_test-TC-contact", "2017-08-14_8C-5per_3_47C",
    "2017-12-04_4_65C-69per_6C"], "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
    "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv",
    "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"],
    "protocol": [null, "8C(5%)-3.47C", "4.65C(69%)-6C"], "date": ["2017-05-09", "2017-08-14", "2017-12-04"],
    "channel_no": ["CH33", "CH44", "CH29"],
    "filename": ["/data-share/raw_cycler_files/2017-05-09_test-TC-contact_CH33.csv",
        "/data-share/raw_cycler_files/2017-08-14_8C-5per_3_47C_CH44.csv",
        "/data-share/raw_cycler_files/2017-12-04_4_65C-69per_6C_CH29.csv"]}'
{"validity": [false, false, true], "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
    "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv",
    "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"]}
"""
import os

import numpy as np
from monty.serialization import loadfn

from beep import VALIDATION_SCHEMA_DIR

DEFAULT_ARBIN_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-arbin-lfp.yaml")
DEFAULT_MACCOR_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-lfp.yaml")
DEFAULT_EIS_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-eis.yaml")
PROJECT_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-projects.yaml")
DEFAULT_VALIDATION_RECORDS = os.path.join(
    VALIDATION_SCHEMA_DIR, "validation_records.json"
)


class SimpleValidator:
    """
    Lightweight class that does Dataframe-based, as
    opposed to dictionary based validation

    Note that schemas here are made to be identical to cerberus
    schemas and should support similar syntax, e. g.

    {COLUMN_NAME:
        {schema:
            type: TYPE_IN_COLUMN, [int, float, str, object]
            max: MAX_VALUE_IN_COLUMN,
            min: MIN_VALUE_IN_COLUMN
         type: list
        }
    }

    Note that the COLUMN_NAME.type key above is ignored, but
    COLUMN_NAME.schema.type is used.

    The only schema keys that are supported at this time are
    max, min, and type.

    Typing is compared using the key-mapping by rule defined
    by the ALLOWED_TYPES_BY_RULE attribute defined below.
    Supported type rules include "integer", "float", "numeric",
    and "string".  Note that type-checking for this class is
    not equivalent to checking types, and may involve custom
    logic which is defined in the check_type method below.
    """

    def __init__(self, schema_filename=DEFAULT_ARBIN_SCHEMA):
        """
        Args:
            schema_filename (str): filename corresponding to
                the schema
        """
        self.schema = loadfn(schema_filename)
        self.validation_records = None

    @staticmethod
    def check_type(df, type_rule):
        """
        Method to check type of input dataframe.

        Args:
            df (pandas.Dataframe): DataFrame.
            type_rule (str): string corresponding to type_rule
                to check, supported type rules are:
                integer: checks for numeric values which are
                    equal to their rounded values

        Returns:
            bool: valid
            str: verbose description of reason
        """
        if type_rule not in ["integer", "float", "numeric", "string"]:
            raise ValueError(
                "type_rule {} not supported, please choose one "
                "of integer, float, numeric, or string"
            )
        # Integer: Check residual from rounding
        if type_rule == "integer":
            nonint_indices = np.arange(len(df))[(df != np.round(df))]
            if nonint_indices.size > 0:
                value = df.iloc[nonint_indices[0]]
                return (
                    False,
                    "integer type check failed at index {} with value {}".format(
                        nonint_indices[0], value
                    ),
                )
        # Float: just check numpy dtyping
        elif type_rule == "float":
            if not np.issubdtype(df.dtype, np.floating):
                return False, "float type check failed, type is {}".format(df.dtype)

        # Numeric: check numpy number dtyping
        elif type_rule == "numeric":
            if not np.issubdtype(df.dtype, np.number):
                return False, "number type check failed, type is {}".format(df.dtype)

        # String: check string/unicode subdtype
        elif type_rule == "string":
            if not (
                np.issubdtype(df.dtype, np.object_)
                or np.issubdtype(df.dtype, np.unicode_)
            ):
                return False, "string type check failed, type is {}".format(df.dtype)
        return True, ""

    def validate(self, dataframe):
        """
        Method to run the validation on everything, and report
        the results, i. e. which columns are inconsistent with
        the schema.

        Args:
            dataframe (pandas.DataFrame): dataframe to be validated.

        Returns:
            dict: report corresponding to each validation
            str: reason for report validation failure, empty string on report
                validation success
        """
        dataframe = dataframe.rename(str.lower, axis="columns")
        for column_name, value in self.schema.items():
            column_schema = value["schema"]
            max_at_least_rule = column_schema.get("max_at_least")
            min_is_below_rule = column_schema.get("min_is_below")
            max_rule = column_schema.get("max")
            min_rule = column_schema.get("min")
            type_rule = column_schema.get("type")
            monotonic_rule = column_schema.get("monotonic")

            # Check type
            if type_rule is not None:
                validity, reason = self.check_type(
                    dataframe[column_name], type_rule=type_rule
                )
                if not validity:
                    reason = "Column {}: {}".format(column_name, reason)
                    return validity, reason

            # Check max
            if max_rule is not None:
                comp = np.where(dataframe[column_name] > max_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = (
                        "{} is higher than allowed max {} at index {}: "
                        "value={}".format(column_name, max_rule, index, value)
                    )
                    return False, reason

            # Check min
            if min_rule is not None:
                comp = np.where(dataframe[column_name] < min_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = (
                        "{} is lower than allowed min {} at index {}:"
                        "value={}".format(column_name, min_rule, index, value)
                    )
                    return False, reason

            # Check a maximum value is at least above a threshold
            if max_at_least_rule is not None:
                comp = np.where(dataframe[column_name].max() < max_at_least_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = (
                        "{} needs to reach at least {} for processing, instead found:"
                        "value={}".format(column_name, max_at_least_rule, value)
                    )
                    return False, reason

            # Check a minimum value is below above a threshold
            if min_is_below_rule is not None:
                comp = np.where(dataframe[column_name].min() > min_is_below_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = (
                        "{} needs to reach under {} for processing, instead found:"
                        "value={}".format(column_name, max_at_least_rule, value)
                    )
                    return False, reason

            if monotonic_rule == 'increasing':
                diff_series = dataframe[column_name].diff().dropna()
                if len(diff_series[diff_series < 0]) > 0:
                    reason = (
                        "{} needs to be monotonically increasing for processing".format(column_name)
                    )
                    return False, reason

        return True, ""


class BeepValidationError(Exception):
    """Custom error to raise when validation fails"""
