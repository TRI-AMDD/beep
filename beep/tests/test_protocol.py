# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Generating protocol files"""

import os
import unittest
import json
import boto3
import numpy as np
import datetime
import pandas as pd
from beep import PROCEDURE_TEMPLATE_DIR
from beep.generate_protocol import ProcedureFile, \
    generate_protocol_files_from_csv, convert_velocity_to_power_waveform, generate_maccor_waveform_file
from monty.tempfile import ScratchDir
from monty.serialization import dumpfn, loadfn
from monty.os import makedirs_p
from botocore.exceptions import NoRegionError, NoCredentialsError
from beep.utils import os_format
import difflib
from sklearn.metrics import mean_absolute_error

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class GenerateProcedureTest(unittest.TestCase):
    def setUp(self):
        # Determine events mode for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = 'test'
        except NoRegionError or NoCredentialsError as e:
            self.events_mode = 'events_off'

    def test_dict_to_file_1(self):
        sdu = ProcedureFile(version='0.1')
        templates = TEST_FILE_DIR
        test_file = 'xTESLADIAG_000003_CH68.000'
        json_file = 'xTESLADIAG_000003_CH68.json'
        test_out = 'test1.000'
        test_dict, sp = sdu.to_dict(
            os.path.join(templates, test_file),
            os.path.join(templates, json_file)
        )
        test_dict = sdu.maccor_format_dict(test_dict)
        sdu.dict_to_xml(
            test_dict, os.path.join(templates, test_out), sp)
        hash1 = sdu.hash_file(os.path.join(templates, test_file))
        hash2 = sdu.hash_file(os.path.join(templates, test_out))
        if hash1 != hash2:
            original = open(os.path.join(templates, test_file)).readlines()
            parsed = open(os.path.join(templates, test_out)).readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                self.assertIsNotNone(line)
        os.remove(os.path.join(templates, test_out))

    def test_dict_to_file_2(self):
        sdu = ProcedureFile(version='0.1')
        templates = TEST_FILE_DIR
        test_file = 'xTESLADIAG_000004_CH69.000'
        json_file = 'xTESLADIAG_000004_CH69.json'
        test_out = 'test2.000'
        test_dict, sp = sdu.to_dict(
            os.path.join(templates, test_file),
            os.path.join(templates, json_file)
        )
        test_dict = sdu.maccor_format_dict(test_dict)
        sdu.dict_to_xml(
            test_dict, os.path.join(templates, test_out), sp)
        hash1 = sdu.hash_file(os.path.join(templates, test_file))
        hash2 = sdu.hash_file(os.path.join(templates, test_out))
        if hash1 != hash2:
            original = open(os.path.join(templates, test_file)).readlines()
            parsed = open(os.path.join(templates, test_out)).readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                self.assertIsNotNone(line)

        os.remove(os.path.join(templates, test_out))

    def test_generate_proc_exp(self):
        sdu = ProcedureFile(version='0.1')
        templates = TEST_FILE_DIR
        test_file = 'EXP.000'
        json_file = 'EXP.json'
        test_out = 'test_EXP.000'
        test_parameters = ["EXP", "4.2", "2.0C", "2.0C"]
        test_dict, sp = sdu.to_dict(
            os.path.join(templates, test_file),
            os.path.join(templates, json_file)
        )
        test_dict = sdu.generate_procedure_exp(test_dict, *test_parameters[1:])
        test_dict = sdu.maccor_format_dict(test_dict)
        sdu.dict_to_xml(
            test_dict, os.path.join(templates, test_out), sp)
        hash1 = sdu.hash_file(os.path.join(templates, test_file))
        hash2 = sdu.hash_file(os.path.join(templates, test_out))
        if hash1 != hash2:
            original = open(os.path.join(templates, test_file)).readlines()
            parsed = open(os.path.join(templates, test_out)).readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                self.assertIsNotNone(line)
        os.remove(os.path.join(templates, test_out))

    def test_missing(self):
        sdu = ProcedureFile(version='0.1')
        templates = TEST_FILE_DIR
        test_file = 'EXP_missing.000'
        json_file = 'EXP_missing.json'
        test_parameters = ["EXP", "4.2", "2.0C", "2.0C"]
        test_dict, sp = sdu.to_dict(
            os.path.join(templates, test_file),
            os.path.join(templates, json_file)
        )
        test_dict = sdu.maccor_format_dict(test_dict)
        self.assertRaises(UnboundLocalError,
                          sdu.generate_procedure_exp, test_dict,
                          *test_parameters[1:])

    def test_from_csv(self):
        csv_file = os.path.join(TEST_FILE_DIR, "parameter_test.csv")

        # Test basic functionality
        with ScratchDir('.') as scratch_dir:
            makedirs_p(os.path.join(scratch_dir, "procedures"))
            makedirs_p(os.path.join(scratch_dir, "names"))
            generate_protocol_files_from_csv(csv_file, scratch_dir)
            self.assertEqual(len(os.listdir(os.path.join(scratch_dir, "procedures"))), 3)

        # Test avoid overwriting file functionality
        with ScratchDir('.') as scratch_dir:
            makedirs_p(os.path.join(scratch_dir, "procedures"))
            makedirs_p(os.path.join(scratch_dir, "names"))
            dumpfn({"hello": "world"}, os.path.join("procedures", "name_000007.000"))
            generate_protocol_files_from_csv(csv_file, scratch_dir)
            post_file = loadfn(os.path.join("procedures", "name_000007.000"))
            self.assertEqual(post_file, {"hello": "world"})
            self.assertEqual(len(os.listdir(os.path.join(scratch_dir, "procedures"))), 3)

    def test_from_csv_2(self):
        csv_file = os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_parameters.csv")

        # Test basic functionality
        with ScratchDir('.') as scratch_dir:
            makedirs_p(os.path.join(scratch_dir, "procedures"))
            makedirs_p(os.path.join(scratch_dir, "names"))
            generate_protocol_files_from_csv(csv_file, scratch_dir)
            self.assertEqual(len(os.listdir(os.path.join(scratch_dir, "procedures"))), 2)

            original = open(os.path.join(PROCEDURE_TEMPLATE_DIR, "diagnosticV2.000")).readlines()
            parsed = open(os.path.join(os.path.join(scratch_dir, "procedures"),
                                       "PredictionDiagnostics_000000.000")).readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                self.assertIsNotNone(line)

            original = open(os.path.join(PROCEDURE_TEMPLATE_DIR, "diagnosticV3.000")).readlines()
            parsed = open(os.path.join(os.path.join(scratch_dir, "procedures"),
                                       "PredictionDiagnostics_000196.000")).readlines()
            diff = list(difflib.unified_diff(original, parsed))
            diff_expected = ['--- \n', '+++ \n', '@@ -27,7 +27,7 @@\n', '           <SpecialType> </SpecialType>\n',
             '           <Oper> = </Oper>\n', '           <Step>002</Step>\n', '-          <Value>03:00:00</Value>\n',
             '+          <Value>03:12:00</Value>\n', '         </EndEntry>\n', '         <EndEntry>\n',
             '           <EndType>Voltage </EndType>\n']
            self.assertEqual(diff, diff_expected)
            for line in difflib.unified_diff(original, parsed):
                self.assertIsNotNone(line)

            _, namefile = os.path.split(csv_file)
            namefile = namefile.split('_')[0] + '_names_'
            namefile = namefile + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
            names_test = open(os.path.join(scratch_dir, "names", namefile)).readlines()
            self.assertEqual(names_test, ['PredictionDiagnostics_000000_\n', 'PredictionDiagnostics_000196_\n'])

    @unittest.skip
    def test_from_csv_3(self):

        csv_file_list = os.path.join(TEST_FILE_DIR, "PreDiag_parameters - GP.csv")
        makedirs_p(os.path.join(TEST_FILE_DIR, "procedures"))
        makedirs_p(os.path.join(TEST_FILE_DIR, "names"))
        generate_protocol_files_from_csv(csv_file_list, TEST_FILE_DIR)
        if os.path.isfile(os.path.join(TEST_FILE_DIR, "procedures", ".DS_Store")):
            os.remove(os.path.join(TEST_FILE_DIR, "procedures", ".DS_Store"))
        self.assertEqual(len(os.listdir(os.path.join(TEST_FILE_DIR, "procedures"))), 265)

    def test_console_script(self):
        csv_file = os.path.join(TEST_FILE_DIR, "parameter_test.csv")

        # Test script functionality
        with ScratchDir('.') as scratch_dir:
            # Set BEEP_ROOT directory to scratch_dir
            os.environ['BEEP_ROOT'] = os.getcwd()
            procedures_path = os.path.join("data-share", "protocols", "procedures")
            names_path = os.path.join("data-share", "protocols", "names")
            makedirs_p(procedures_path)
            makedirs_p(names_path)
            # Test the script
            json_input = json.dumps(
                {"file_list": [csv_file],
                 "mode": self.events_mode})
            os.system("generate_protocol {}".format(os_format(json_input)))
            self.assertEqual(len(os.listdir(procedures_path)), 3)

    def test_convert_velocity_to_power_waveform(self):
        velocity_waveform_file = os.path.join(TEST_FILE_DIR, "LA4_velocity_waveform.txt")

        df_velocity = pd.read_csv(velocity_waveform_file, sep="\t", header=0)
        df_power = convert_velocity_to_power_waveform(velocity_waveform_file, 'mph')
        #Check input and output sizes
        self.assertEqual(len(df_velocity), len(df_power))
        self.assertTrue(any(df_power['power']<0))

    def test_generate_maccor_waveform_file_default(self):
        velocity_waveform_file = os.path.join(TEST_FILE_DIR, "LA4_velocity_waveform.txt")

        df_power = convert_velocity_to_power_waveform(velocity_waveform_file, 'mph')
        df_MWF = pd.read_csv(generate_maccor_waveform_file(df_power, "test_LA4_waveform", TEST_FILE_DIR), sep='\t', header=None)

        #Reference mwf file generated by the cycler for the same power waveform.
        df_MWF_ref = pd.read_csv(os.path.join(TEST_FILE_DIR, "LA4_reference_default_settings.mwf"), sep="\t", header=None)

        self.assertEqual(df_MWF.shape, df_MWF_ref.shape)

        #Check that the fourth column for charge/discharge limit is empty (default setting)
        self.assertTrue(df_MWF.iloc[:,3].isnull().all())

        #Check that sum of durations equals length of the power timeseries
        self.assertEqual(df_MWF.iloc[:,5].sum(), len(df_power))

        #Check that charge/discharge steps are identical
        self.assertTrue((df_MWF.iloc[:,0] == df_MWF_ref.iloc[:,0]).all())

        #Check that power values are close to each other (col 2)
        relative_differences = np.abs((df_MWF.iloc[:,2] - df_MWF_ref.iloc[:,2]) / df_MWF_ref.iloc[:,2])
        self.assertLessEqual(np.mean(relative_differences)*100, 0.01) #mean percentage error < 0.01%

    def test_generate_maccor_waveform_file_custom(self):
        velocity_waveform_file = os.path.join(TEST_FILE_DIR, "US06_velocity_waveform.txt")
        mwf_config = {'control_mode': 'I',
                      'value_scale': 1,
                      'charge_limit_mode': 'R',
                      'charge_limit_value': 2,
                      'discharge_limit_mode': 'P',
                      'discharge_limit_value': 3,
                      'charge_end_mode': 'V',
                      'charge_end_operation': '>=',
                      'charge_end_mode_value': 4.2,
                      'discharge_end_mode': 'V',
                      'discharge_end_operation': '<=',
                      'discharge_end_mode_value': 3,
                      'report_mode': 'T',
                      'report_value': 10,
                      'range': 'A',
                      }
        df_power = convert_velocity_to_power_waveform(velocity_waveform_file, 'mph')
        df_MWF = pd.read_csv(generate_maccor_waveform_file(df_power, "test_US06_waveform", TEST_FILE_DIR,
                                                           mwf_config=mwf_config), sep='\t', header=None)
        df_MWF_ref = pd.read_csv(os.path.join(TEST_FILE_DIR, "US06_reference_custom_settings.mwf"), sep="\t", header=None)

        #Check dimensions with the reference mwf file
        self.assertEqual(df_MWF.shape, df_MWF_ref.shape)

        #Check that control_mode, charge/discharge state, limit and limit_value columns are identical.
        self.assertTrue((df_MWF.iloc[:, [0,1,3,4]] == df_MWF_ref.iloc[:, [0,1,3,4]]).all().all())

        #Check that power values are close to each other (col 2)
        relative_differences = np.abs((df_MWF.iloc[:,2] - df_MWF_ref.iloc[:,2]) / df_MWF_ref.iloc[:,2])
        self.assertLessEqual(np.mean(relative_differences)*100, 0.01) #mean percentage error < 0.01%