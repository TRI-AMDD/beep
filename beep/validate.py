# Copyright 2019 Toyota Research Institute. All rights reserved.
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

import json
import os
import warnings
import re
from datetime import datetime

import numpy as np
import pandas as pd
from docopt import docopt
from cerberus import Validator
from beep import tqdm
from monty.serialization import loadfn, dumpfn

from beep import VALIDATION_SCHEMA_DIR
from beep.conversion_schemas import ARBIN_CONFIG, MACCOR_CONFIG
from beep.utils import KinesisEvents
from beep import logger, __version__

DEFAULT_ARBIN_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-arbin-lfp.yaml")
DEFAULT_MACCOR_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-lfp.yaml")
DEFAULT_EIS_SCHEMA = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-eis.yaml")
DEFAULT_VALIDATION_RECORDS = os.path.join(VALIDATION_SCHEMA_DIR, "validation_records.json")
s = {'service': 'DataValidator'}


class ValidatorBeep(Validator):
    """
    Data validation for battery cycling.
    Currently supports Arbin and Maccor cyclers.

    """

    def validate_arbin_dataframe(self, df, schema=DEFAULT_ARBIN_SCHEMA):
        """
        Validator for large, cyclic dataframes coming from Arbin.
        Requires a valid Cycle_Index column of type int.
        Designed for performance - will stop at the first encounter of issues.

        Args:
            df (pandas.DataFrame): Arbin output as DataFrame.
            schema (str): Path to the validation schema. Defaults to arbin for now.
        Returns:
            bool: True if validated with out errors. If validation fails, errors
                are listed at ValidatorBeep.errors.
        """

        try:
            schema = loadfn(schema)
            self.arbin_schema = schema
        except Exception as e:
            warnings.warn('Arbin schema could not be found: {}'.format(e))

        df = df.rename(str.lower, axis='columns')

        # Validation cycle index data and cast to int
        if not self._prevalidate_nonnull_column(df, 'cycle_index'):
            return False
        df.cycle_index = df.cycle_index.astype(int, copy=False)

        # Validation starts here
        self.schema = self.arbin_schema

        for cycle_index, cycle_df in tqdm(df.groupby("cycle_index")):
            cycle_dict = cycle_df.replace({np.nan, 'None'}).to_dict(orient='list')
            result = self.validate(cycle_dict)
            if not result:
                return False
        return True

    def validate_maccor_dataframe(self, df, schema=DEFAULT_MACCOR_SCHEMA):
        """
        Validator for large, cyclic dataframes coming from Maccor.
        Requires a valid Cyc# column of type int.
        Designed for performance - will stop at the first encounter of issues.

        Args:
            df (pandas.DataFrame): Maccor output as DataFrame.
            schema (str): Path to the validation schema. Defaults to maccor for now.
        Returns:
            bool: True if validated with out errors. If validation fails, errors
            are listed at ValidatorBeep.errors.
        """

        try:
            schema = loadfn(schema)
            self.maccor_schema = schema
        except Exception as e:
            warnings.warn('Maccor schema could not be found: {}'.format(e))

        df = df.rename(str.lower, axis='columns')

        # Validation cycle index data and cast to int
        if not self._prevalidate_nonnull_column(df, 'cyc#'):
            return False
        df['cyc#'] = df['cyc#'].astype(int, copy=False)

        # Validation starts here
        self.schema = self.maccor_schema

        for cycle_index, cycle_df in tqdm(df.groupby("cyc#")):
            cycle_dict = cycle_df.replace({np.nan, 'None'}).to_dict(orient='list')
            result = self.validate(cycle_dict)
            if not result:
                return False
        return True

    def validate_eis_dataframe(self, df, schema=DEFAULT_EIS_SCHEMA):
        """
        Validator for Maccor EIS.
        Args:
            df (pandas.DataFrame): Maccor EIS output as DataFrame.
            schema (str): Path to the validation schema. Defaults to maccor for now.
        Returns:
            bool: True if validated with out errors. If validation fails, errors
                are listed at ValidatorBeep.errors.
        """

        try:
            schema = loadfn(schema)
            self.eis_schema = schema
        except Exception as e:
            warnings.warn('Maccor EIS schema could not be found: {}'.format(e))

        df = df.rename(str.lower, axis='columns')
        self.schema = self.eis_schema

        return self.validate(df.replace(np.nan, 'None').to_dict(orient='list'), )

    def validate_from_paths(self, paths, record_results=False, skip_existing=False,
                            record_path=DEFAULT_VALIDATION_RECORDS):
        """
        This method streamlines validation of multiple Arbin csv files given a list of paths.

        It can also do bookkeeping of validations by dumping results in a json file,
        locally until a more centralized method is implemented.

        Args:
            paths (list): a list of paths to csv files.
            record_results (bool): Whether to record the validation results locally or not (defaults to False).
            skip_existing (bool): Whether to skip already validated files. This is done by checking if the file is in
                the validation_records. skip_existing only matters if record_results is True. Defaults to False.
            record_path (str): path to the json file storing the past validation results.
        Returns:
            dict: Results of the validation in the form of a key,value pairs where each key corresponds to the filename
                validated. For each file, the results contain a field "validated", True if validation was successful or
                False if not. "errors", "method" and "time" are simply the errors encountered during validation, method
                used for validation, and time of validation, respectively.

        """
        self.allow_unknown = True

        if record_results:
            if os.path.isfile(record_path):
                self.validation_records = loadfn(record_path)
                if skip_existing:
                    paths = [path for path in paths if os.path.basename(path)
                             not in self.validation_records]
            else:
                self.validation_records = {}

        results = {}
        for path in paths:
            name = os.path.basename(path)
            results[name] = {}
            if re.match(ARBIN_CONFIG['file_pattern'], path):
                df = pd.read_csv(path, index_col=0)
                results[name]['validated'] = self.validate_arbin_dataframe(df)
                results[name]['method'] = self.validate_arbin_dataframe.__name__
            elif re.match(MACCOR_CONFIG['file_pattern'], path):
                df = pd.read_csv(path, delimiter='\t', skiprows=1)
                results[name]['validated'] = self.validate_maccor_dataframe(df)
                results[name]['method'] = self.validate_maccor_dataframe.__name__
            else:
                results[name]['validated'] = False
                results[name]['method'] = None
                self.errors = ["File type not recognized"]

            results[name]['time'] = json.dumps(datetime.now(), indent=4, sort_keys=True, default=str)
            results[name]['errors'] = self.errors

        if record_results:
            self.validation_records.update(results)
            dumpfn(self.validation_records, record_path)

        return results

    def _prevalidate_nonnull_column(self, df, column_name):
        """
        Scheme for prevalidation of non-numeric column,
        primarily used to pre-validate non-null cycle index,
        This is induces an error on validation of a dataframe
        with a null value in cycle_index, using a dummy doc
        and a custom schema for non-numeric cycle values.

        Args:
            df (pandas.DataFrame): dataframe.
            column_name: column identifier for pandas.

        Returns:
            bool: whether or not column contains a null value according to
                pandas.isnull().

        """
        null_index_mask = df[column_name].isnull()
        if null_index_mask.any():

            non_numeric_schema = {column_name : {'type': 'list',
                                                 'schema': {'type': 'number'}}}
            dummy_df = df[null_index_mask].iloc[0:1].replace({np.nan: 'None'})
            dummy_doc = dummy_df.to_dict(orient='list')
            self.validate(dummy_doc, schema=non_numeric_schema)
            return False
        else:
            return True


class SimpleValidator(object):
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
            raise ValueError("type_rule {} not supported, please choose one "
                             "of integer, float, numeric, or string")
        # Integer: Check residual from rounding
        if type_rule == "integer":
            nonint_indices = np.arange(len(df))[(df != np.round(df))]
            if nonint_indices.size > 0:
                value = df.iloc[nonint_indices[0]]
                return False, "integer type check failed at index {} with value {}".format(
                    nonint_indices[0], value
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
            if not (np.issubdtype(df.dtype, np.object_) or np.issubdtype(df.dtype, np.unicode_)):
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
        dataframe = dataframe.rename(str.lower, axis='columns')
        for column_name, value in self.schema.items():
            column_schema = value['schema']
            max_at_least_rule = column_schema.get('max_at_least')
            min_is_below_rule = column_schema.get('min_is_below')
            max_rule = column_schema.get('max')
            min_rule = column_schema.get('min')
            type_rule = column_schema.get('type')

            # Check type
            if type_rule is not None:
                validity, reason = self.check_type(dataframe[column_name], type_rule=type_rule)
                if not validity:
                    reason = "Column {}: {}".format(column_name, reason)
                    return validity, reason

            # Check max
            if max_rule is not None:
                comp = np.where(dataframe[column_name] > max_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = "{} is higher than allowed max {} at index {}: " \
                             "value={}".format(
                        column_name, max_rule, index, value
                    )
                    return False, reason

            # Check min
            if min_rule is not None:
                comp = np.where(dataframe[column_name] < min_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = "{} is lower than allowed min {} at index {}:" \
                             "value={}".format(
                        column_name, min_rule, index, value
                    )
                    return False, reason

            # Check a maximum value is at least above a threshold
            if max_at_least_rule is not None:
                comp = np.where(dataframe[column_name].max() < max_at_least_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = "{} needs to reach at least {} for processing, instead found:" \
                             "value={}".format(
                        column_name, max_at_least_rule, value
                    )
                    return False, reason

            # Check a minimum value is below above a threshold
            if min_is_below_rule is not None:
                comp = np.where(dataframe[column_name].min() > min_is_below_rule)
                if comp[0].size > 0:
                    index = comp[0][0]
                    value = dataframe[column_name].iloc[index]
                    reason = "{} needs to reach under {} for processing, instead found:" \
                             "value={}".format(
                        column_name, max_at_least_rule, value
                    )
                    return False, reason

        return True, ''

    def validate_from_paths(self, paths, record_results=False, skip_existing=False,
                            record_path=DEFAULT_VALIDATION_RECORDS):
        """
        This method streamlines validation of multiple Arbin csv files given a list of paths.

        It can also do bookkeeping of validations by dumping results in a json file,
        locally until a more centralized method is implemented.

        Args:
            paths (list): a list of paths to csv files
            record_results (bool): Whether to record the validation results locally or not (defaults to False)
            skip_existing (bool): Whether to skip already validated files. This is done by checking if the
                                    file is in the validation_records. skip_existing only matters if record_results
                                    is True. (defaults to False)
            record_path (str): path to the json file storing the past validation results.
        Returns:
            dict: Results of the validation in the form of a key,value pairs where each key corresponds to the filename
                validated. For each file, the results contain a field "validated", True if validation was successful or
                False if not. "errors", "method" and "time" are simply the errors encountered during validation, method
                used for validation, and time of validation, respectively.

        """
        if record_results:
            if os.path.isfile(record_path):
                self.validation_records = loadfn(record_path)
                if skip_existing:
                    paths = [path for path in paths if os.path.basename(path)
                             not in self.validation_records]
            else:
                self.validation_records = {}

        results = {}
        for path in tqdm(paths):
            name = os.path.basename(path)
            results[name] = {}
            if re.match(ARBIN_CONFIG['file_pattern'], path):
                schema_filename = os.path.join(VALIDATION_SCHEMA_DIR, "schema-arbin-lfp.yaml")
                self.schema = loadfn(schema_filename)
                df = pd.read_csv(path, index_col=0)
                validated, reason = self.validate(df)
                method = "simple_arbin"
            elif re.match(MACCOR_CONFIG['file_pattern'], path):
                schema_filename = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml")
                self.schema = loadfn(schema_filename)
                self.allow_unknown = True
                df = pd.read_csv(path, delimiter='\t', skiprows=1)

                # Columns need to be retyped and renamed for validation,
                # conversion will happen during structuring
                df['State'] = df['State'].astype(str)
                df['current'] = df['Amps']

                validated, reason = self.validate(df)
                method = "simple_maccor"
            else:
                validated, reason = False, "File type not recognized"
                method = None
            results[name].update({"validated": validated,
                                  "method": method,
                                  "errors": reason,
                                  "time": json.dumps(datetime.now(), indent=4, sort_keys=True, default=str)})

            if validated:
                logger.info("%s method=%s errors=%s", name, method, reason, extra=s)
            else:
                logger.warning("%s method=%s errors=%s", name, method, reason, extra=s)

        if record_results:
            self.validation_records.update(results)
            dumpfn(self.validation_records, record_path)

        return results


class BeepValidationError(Exception):
    """Custom error to raise when validation fails"""


def validate_file_list_from_json(file_list_json, record_results=False,
                                 skip_existing=False, validator_class=SimpleValidator):
    """
    Validates a list of files from json input

    Args:
        file_list_json (str): input for validation files, should be a json string
            with attribute "file_list" or a filename (e. g. something.json)
            corresponding to a json object with a similar attribute.
        record_results (bool): Whether to record the validation results locally
            or not (defaults to False).
        skip_existing (bool): Whether to skip already validated files. This
            is done by checking if the file is in the validation_records.
            skip_existing only matters if record_results is True. (defaults to False)
        validator_class (ValidatorBeep or SimpleValidator): validator class
            to use in validation.

    Returns:
        str: json dump of the validator results.

    """
    # Process input json
    if file_list_json.endswith(".json"):
        file_list_data = loadfn(file_list_json)
    else:
        file_list_data = json.loads(file_list_json)

    # Setup Events
    events = KinesisEvents(service='DataValidator', mode=file_list_data['mode'])

    file_list = file_list_data['file_list']

    validator = validator_class()
    all_results = validator.validate_from_paths(
        file_list, record_results=record_results, skip_existing=skip_existing,
    )

    # Get validities and recast to strings (valid/invalid) based on result
    validity = [all_results[os.path.split(file)[-1]]['validated']
                for file in file_list]

    validity = list(map(lambda x: 'valid' if x else 'invalid', validity))

    # Get errors
    errors = [all_results[os.path.split(file)[-1]]['errors']
              for file in file_list]
    messages = [{'comment': '',
                 'error': error} for error in errors]
    output_json = {'file_list': file_list, 'run_list': file_list_data['run_list'],
                   'validity': validity, 'message_list': messages}

    events.put_validation_event(output_json, 'complete')

    return json.dumps(output_json)


def main():
    logger.info('starting', extra=s)
    logger.info('Running version=%s', __version__, extra=s)
    try:
        args = docopt(__doc__)
        input_json = args['INPUT_JSON']
        print(validate_file_list_from_json(input_json), end="")
    except Exception as e:
        logger.error(str(e), extra=s)
        raise e
    logger.info('finish', extra=s)
    return None


if __name__ == "__main__":
    main()
