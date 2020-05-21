# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to batch validation"""

import json
import os
import unittest
import warnings

import pandas as pd
import numpy as np
import boto3

from monty.tempfile import ScratchDir
from beep.validate import ValidatorBeep, validate_file_list_from_json, \
    SimpleValidator
from beep import S3_CACHE, VALIDATION_SCHEMA_DIR

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


@unittest.skip
class ValidationArbinTest(unittest.TestCase):
    def setUp(self):
        # Setup events for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = "test"
        except Exception as e:
            warnings.warn("Cloud resources not configured")
            self.events_mode = "events_off"

    def test_validation_arbin_bad_index(self):
        path = "2017-05-09_test-TC-contact_CH33.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = ValidatorBeep()
        v.allow_unknown = True

        df = pd.read_csv(path, index_col=0)

        self.assertFalse(v.validate_arbin_dataframe(df))
        self.assertEqual(v.errors['cycle_index'][0][0][0], 'must be of number type')

        # Test bigger file
        path = "2017-08-14_8C-5per_3_47C_CH44.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = ValidatorBeep()
        v.allow_unknown = True

        df = pd.read_csv(path, index_col=0)
        self.assertFalse(v.validate_arbin_dataframe(df))
        self.assertEqual(v.errors['cycle_index'][0][0][0], 'must be of number type')

    def test_validation_arbin_bad_data(self):
        path = "2017-12-04_4_65C-69per_6C_CH29.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = ValidatorBeep()
        v.allow_unknown = True

        df = pd.read_csv(path, index_col=0)

        self.assertTrue(v.validate_arbin_dataframe(df))

        # Alter the schema on-the-fly to induce error
        v.schema['discharge_capacity']['schema']['max'] = 1.8
        self.assertFalse(v.validate_arbin_dataframe(df, schema=v.schema))
        self.assertEqual(v.errors['discharge_capacity'][0][11264][0], 'max value is 1.8')

        # Alter the schema on-the-fly to move on to the next errors
        v.schema['discharge_capacity']['schema']['max'] = 2.1
        v.schema['step_time'] = {"schema": {"min": 0.0, "type": "float"},
                                 "type": "list"}
        self.assertFalse(v.validate_arbin_dataframe(df, schema=None))
        self.assertEqual(v.errors['step_time'][0][206][0], 'min value is 0.0')

        # Alter schema once more to recover validation
        del v.schema['step_time']['schema']['min']
        self.assertTrue(v.validate_arbin_dataframe(df, schema=None))

    def test_validation_many_from_paths(self):
        paths = ["2017-05-09_test-TC-contact_CH33.csv",
                 "2017-12-04_4_65C-69per_6C_CH29.csv"]
        paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
        v = ValidatorBeep()

        temp_records = os.path.join(TEST_FILE_DIR, 'temp_records.json')
        with open(temp_records, 'w') as f:
            f.write("{}")

        results = v.validate_from_paths(paths, record_results=False)
        self.assertFalse(results["2017-05-09_test-TC-contact_CH33.csv"]["validated"])
        errmsg = results["2017-05-09_test-TC-contact_CH33.csv"]["errors"]['cycle_index'][0][0][0]
        self.assertEqual(errmsg, 'must be of number type')
        self.assertTrue(results["2017-12-04_4_65C-69per_6C_CH29.csv"]["validated"])

        v.validate_from_paths(paths, record_results=True, record_path=temp_records)
        with open(temp_records, 'r') as f:
            results_form_rec = json.load(f)

        self.assertFalse(results_form_rec["2017-05-09_test-TC-contact_CH33.csv"]["validated"])

        results = v.validate_from_paths(paths, record_results=True, skip_existing=True,
                                        record_path=temp_records)
        self.assertEqual(results, {})

    @unittest.skip
    def test_bad_file(self):
        paths = ["2017-08-14_8C-5per_3_47C_CH44.csv"]
        paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
        v = ValidatorBeep()
        results = v.validate_from_paths(paths, record_results=False)

    def test_validation_from_json(self):
        with ScratchDir('.'):
            os.environ['BEEP_PROCESSING_DIR'] = os.getcwd()
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "validation"))
            paths = ["2017-05-09_test-TC-contact_CH33.csv",
                     "2017-12-04_4_65C-69per_6C_CH29.csv"]
            paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
            # Create dummy json obj
            json_obj = {
                        "mode": self.events_mode,
                        "file_list": paths,
                        'run_list': list(range(len(paths)))
                        }
            json_string = json.dumps(json_obj)
            json_output = validate_file_list_from_json(json_string)
            loaded = json.loads(json_output)
        self.assertEqual(loaded['validity'][0], 'invalid')
        self.assertEqual(loaded['validity'][1], 'valid')


class ValidationMaccorTest(unittest.TestCase):
    # To further develop as Maccor data / schema becomes available
    def setUp(self):
        # Setup events for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = "test"
        except Exception as e:
            warnings.warn("Cloud resources not configured")
            self.events_mode = "events_off"

    def test_validation_maccor(self):
        path = "xTESLADIAG_000019_CH70.070"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator(schema_filename=os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml"))
        v.allow_unknown = True
        header = pd.read_csv(path, delimiter='\t', nrows=0)
        df = pd.read_csv(path, delimiter='\t', skiprows=1)
        df['State'] = df['State'].astype(str)
        df['current'] = df['Amps']
        validity, reason = v.validate(df)
        self.assertTrue(validity)

    def test_validate_from_paths_maccor(self):
        paths = [os.path.join(TEST_FILE_DIR, "xTESLADIAG_000019_CH70.070")]

        # Run validation on everything
        v = SimpleValidator()
        validate_record = v.validate_from_paths(paths, record_results=True,
                                                skip_existing=False)
        df = pd.DataFrame(v.validation_records)
        df = df.transpose()
        self.assertEqual(df.loc["xTESLADIAG_000019_CH70.070", "method"], "simple_maccor")
        self.assertEqual(df.loc["xTESLADIAG_000019_CH70.070", "validated"], True)


class ValidationEisTest(unittest.TestCase):
    # To further develop
    def setUp(self):
        pass

    def test_validation_maccor(self):
        path = "maccor_test_file_4267-66-6519.EDA0001.041"
        path = os.path.join(TEST_FILE_DIR, path)

        v = ValidatorBeep()
        v.allow_unknown = True

        df = pd.read_csv(path, delimiter='\t', skip_blank_lines=True, skiprows=10)

        self.assertTrue(v.validate_eis_dataframe(df))


class SimpleValidatorTest(unittest.TestCase):
    def setUp(self):
        # Setup events for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = "test"
        except Exception as e:
            warnings.warn("Cloud resources not configured")
            self.events_mode = "events_off"

    def test_file_incomplete(self):
        path = "FastCharge_000025_CH8.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)

        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(
            reason, "cycle_index needs to reach at least 1 "
                    "for processing, instead found:value=0.0")

    def test_basic(self):
        path = "2017-05-09_test-TC-contact_CH33.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)

        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(
            reason, "Column cycle_index: integer type check failed "
                    "at index 0 with value nan")

        # Test bigger file, with float/numeric type
        path = "2017-08-14_8C-5per_3_47C_CH44.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)
        v.schema['cycle_index']['schema']['type'] = 'float'
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, '')

        v.schema['cycle_index']['schema']['type'] = 'numeric'
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, '')

        # Test good file
        path = "2017-12-04_4_65C-69per_6C_CH29.csv"
        path = os.path.join(TEST_FILE_DIR, path)

        v = SimpleValidator()
        df = pd.read_csv(path, index_col=0)
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, '')

        # Alter the schema on-the-fly to induce error
        v.schema['discharge_capacity']['schema']['max'] = 1.8
        validity, reason = v.validate(df)
        self.assertFalse(validity)
        # Cut off reasons to prevent floating format discrepancies
        self.assertEqual(
            reason[:80], "discharge_capacity is higher than allowed max 1.8 "
                         "at index 11154: value=1.801418")

        # Alter the schema on-the-fly to move on to the next errors
        v.schema['discharge_capacity']['schema']['max'] = 2.1
        v.schema['step_time'] = {"schema": {"min": 0.0, "type": "float"}}
        validity, reason = v.validate(df)
        self.assertFalse(validity)
        self.assertEqual(reason[:70], 'step_time is lower than allowed min 0.0 '
                                 'at index 104416:value=-450.945')

        # Alter schema once more to recover validation
        del v.schema['step_time']['schema']['min']
        validity, reason = v.validate(df)
        self.assertTrue(validity)
        self.assertEqual(reason, '')

    def test_validation_from_json(self):
        with ScratchDir('.'):
            os.environ['BEEP_PROCESSING_DIR'] = os.getcwd()
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "validation"))
            paths = ["2017-05-09_test-TC-contact_CH33.csv",
                     "2017-12-04_4_65C-69per_6C_CH29.csv"]
            paths = [os.path.join(TEST_FILE_DIR, path) for path in paths]
            # Create dummy json obj
            json_obj = {
                        "mode": self.events_mode,
                        "file_list": paths,
                        'run_list': list(range(len(paths)))
                        }
            json_string = json.dumps(json_obj)
            json_output = validate_file_list_from_json(json_string)
            loaded = json.loads(json_output)
        self.assertEqual(loaded['validity'][0], 'invalid')
        self.assertEqual(loaded['validity'][1], 'valid')

    @unittest.skipUnless(False, "toggle this test")
    def test_heavy(self):
        # Sync all S3 objects
        from beep.utils.memprof import cache_all_kitware_data
        cache_all_kitware_data()
        paths = os.listdir(os.path.join(S3_CACHE, "D3Batt_Data_publication"))
        paths = [path for path in paths if not "Metadata" in path]
        paths = [os.path.join(S3_CACHE, "D3Batt_Data_publication", path)
                 for path in paths]

        # Run validation on everything
        v = SimpleValidator()
        validate_record = v.validate_from_paths(paths, record_results=True,
                                                skip_existing=True)
        df = pd.DataFrame(v.validation_records)
        df = df.transpose()
        print(df)
        print("{} valid, {} invalid".format(
            len([x for x in df.validated if x]),
            len([x for x in df.validated if not x])))
        invalid_runs = df[np.logical_not(df.validated)]
        print(invalid_runs)
        self.assertEqual(len(invalid_runs), 3)


if __name__ == "__main__":
    unittest.main()
