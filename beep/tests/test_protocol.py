# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Generating protocol files"""

import os
import unittest
import json
import boto3
import datetime
from beep.protocol import PROCEDURE_TEMPLATE_DIR, SCHEDULE_TEMPLATE_DIR
from beep.generate_protocol import Procedure, \
    generate_protocol_files_from_csv
from beep.protocol.arbin import Schedule
from beep.protocol.maccor_to_arbin import ProcedureToSchedule
from beep.tests.test_maccor_to_arbin import TEST_FILE_DIR
from monty.tempfile import ScratchDir
from monty.serialization import dumpfn, loadfn
from monty.os import makedirs_p
from botocore.exceptions import NoRegionError, NoCredentialsError
from beep.utils import os_format, hash_file
import difflib

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class ProcedureTest(unittest.TestCase):
    def setUp(self):
        # Determine events mode for testing
        try:
            kinesis = boto3.client('kinesis')
            response = kinesis.list_streams()
            self.events_mode = 'test'
        except NoRegionError or NoCredentialsError as e:
            self.events_mode = 'events_off'

    def test_io(self):
        test_file = os.path.join(TEST_FILE_DIR, 'xTESLADIAG_000003_CH68.000')
        json_file = os.path.join(TEST_FILE_DIR, 'xTESLADIAG_000003_CH68.json')
        test_out = 'test1.000'

        procedure = Procedure.from_file(os.path.join(TEST_FILE_DIR, test_file))
        with ScratchDir('.'):
            dumpfn(procedure, json_file)
            procedure.to_file(test_out)
            hash1 = hash_file(test_file)
            hash2 = hash_file(test_out)
            if hash1 != hash2:
                original = open(test_file).readlines()
                parsed = open(test_out).readlines()
                self.assertFalse(list(difflib.unified_diff(original, parsed)))
                for line in difflib.unified_diff(original, parsed):
                    self.assertIsNotNone(line)

        test_file = os.path.join(TEST_FILE_DIR, 'xTESLADIAG_000004_CH69.000')
        json_file = os.path.join(TEST_FILE_DIR, 'xTESLADIAG_000004_CH69.json')
        test_out = 'test2.000'

        procedure = Procedure.from_file(os.path.join(TEST_FILE_DIR, test_file))
        with ScratchDir('.'):
            dumpfn(procedure, json_file)
            procedure.to_file(test_out)
            hash1 = hash_file(test_file)
            hash2 = hash_file(test_out)
            if hash1 != hash2:
                original = open(test_file).readlines()
                parsed = open(test_out).readlines()
                self.assertFalse(list(difflib.unified_diff(original, parsed)))
                for line in difflib.unified_diff(original, parsed):
                    self.assertIsNotNone(line)

    def test_generate_proc_exp(self):
        test_file = os.path.join(TEST_FILE_DIR, 'EXP.000')
        json_file = os.path.join(TEST_FILE_DIR, 'EXP.json')
        test_out = 'test_EXP.000'
        test_parameters = ["4.2", "2.0C", "2.0C"]
        procedure = Procedure.from_exp(*test_parameters)
        with ScratchDir('.'):
            dumpfn(procedure, json_file)
            procedure.to_file(test_out)
            hash1 = hash_file(test_file)
            hash2 = hash_file(test_out)
            if hash1 != hash2:
                original = open(test_file).readlines()
                parsed = open(test_out).readlines()
                self.assertFalse(list(difflib.unified_diff(original, parsed)))
                for line in difflib.unified_diff(original, parsed):
                    self.assertIsNotNone(line)

    def test_missing(self):
        test_parameters = ["EXP", "4.2", "2.0C", "2.0C"]
        template = os.path.join(TEST_FILE_DIR, "EXP_missing.000")
        self.assertRaises(UnboundLocalError, Procedure.from_exp,
                          *test_parameters[1:]+[template])

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


class ProcedureToScheduleTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_single_step_conversion(self):
        procedure = Procedure()

        templates = PROCEDURE_TEMPLATE_DIR

        test_file = 'diagnosticV3.000'
        json_file = 'test.json'

        proc_dict, sp = procedure.to_dict(os.path.join(templates, test_file),
                                          os.path.join(templates, json_file)
                                          )
        proc_dict = procedure.maccor_format_dict(proc_dict)
        test_step_dict = proc_dict['MaccorTestProcedure']['ProcSteps']['TestStep']

        converter = ProcedureToSchedule(test_step_dict)
        step_index = 5
        step_name_list, step_flow_ctrl = converter.create_metadata()

        self.assertEqual(step_flow_ctrl[7], '5-reset cycle C/20')
        self.assertEqual(step_flow_ctrl[68], '38-reset cycle')

        step_arbin = converter.compile_to_arbin(test_step_dict[step_index], step_index, step_name_list, step_flow_ctrl)
        self.assertEqual(step_arbin['m_szLabel'], '6-None')
        self.assertEqual(step_arbin['[Schedule_Step5_Limit0]']['m_szGotoStep'], 'Next Step')
        self.assertEqual(step_arbin['[Schedule_Step5_Limit0]']['Equation0_szLeft'], 'PV_CHAN_Voltage')
        self.assertEqual(step_arbin['[Schedule_Step5_Limit2]']['m_szGotoStep'], '70-These are the 2 reset cycles')

        step_index = 8
        step_arbin = converter.compile_to_arbin(test_step_dict[step_index], step_index, step_name_list, step_flow_ctrl)
        print(step_index, test_step_dict[step_index])
        print(step_arbin)
        self.assertEqual(step_arbin['[Schedule_Step8_Limit0]']['Equation0_szLeft'], 'PV_CHAN_CV_Stage_Current')
        self.assertEqual(step_arbin['[Schedule_Step8_Limit0]']['Equation0_szRight'],
                         test_step_dict[step_index]['Ends']['EndEntry'][0]['Value'])
        os.remove(os.path.join(templates, json_file))

    def test_serial_conversion(self):
        procedure = Procedure()

        templates = PROCEDURE_TEMPLATE_DIR

        test_file = 'diagnosticV3.000'
        json_file = 'test.json'

        proc_dict, sp = procedure.to_dict(os.path.join(templates, test_file),
                                          os.path.join(templates, json_file)
                                          )
        proc_dict = procedure.maccor_format_dict(proc_dict)
        test_step_dict = proc_dict['MaccorTestProcedure']['ProcSteps']['TestStep']

        converter = ProcedureToSchedule(test_step_dict)
        step_name_list, step_flow_ctrl = converter.create_metadata()

        for step_index, step in enumerate(test_step_dict):
            if 'Loop' in step['StepType']:
                print(step_index, step)
            step_arbin = converter.compile_to_arbin(test_step_dict[step_index], step_index,
                                                    step_name_list, step_flow_ctrl)
            if 'Loop' in step['StepType']:
                self.assertEqual(step_arbin['m_szStepCtrlType'], 'Set Variable(s)')
                self.assertEqual(step_arbin['m_uLimitNum'], '2')
            if step_index == 15:
                self.assertEqual(step_arbin['[Schedule_Step15_Limit0]']['m_szGotoStep'], '11-None')
                self.assertEqual(step_arbin['[Schedule_Step15_Limit1]']['m_szGotoStep'], 'Next Step')
        os.remove(os.path.join(templates, json_file))

    def test_schedule_creation(self):
        procedure = Procedure()

        templates = PROCEDURE_TEMPLATE_DIR

        test_file = 'diagnosticV3.000'
        json_file = 'test.json'
        sdu_test_input = os.path.join(SCHEDULE_TEMPLATE_DIR, '20170630-3_6C_9per_5C.sdu')
        sdu_test_output = os.path.join(TEST_FILE_DIR, 'schedule_test_output.sdu')

        proc_dict, sp = procedure.to_dict(os.path.join(templates, test_file),
                                          os.path.join(templates, json_file)
                                          )
        proc_dict = procedure.maccor_format_dict(proc_dict)
        test_step_dict = proc_dict['MaccorTestProcedure']['ProcSteps']['TestStep']

        converter = ProcedureToSchedule(test_step_dict)
        converter.create_sdu(sdu_test_input, sdu_test_output)
        os.remove(os.path.join(templates, json_file))
        os.remove(sdu_test_output)


class ArbinScheduleTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_dict_to_file(self):
        filename = '20170630-3_6C_9per_5C.sdu'
        schedule = Schedule.from_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        testname = 'test1.sdu'
        with ScratchDir('.'):
            dumpfn(schedule, "schedule_test.json")
            schedule.to_file(testname)
            hash1 = hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
            hash2 = hash_file(testname)
            if hash1 != hash2:
                original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').read()
                parsed = open(testname, encoding='latin-1').read()
                self.assertFalse(list(difflib.unified_diff(original, parsed)))
                for line in difflib.unified_diff(original, parsed):
                    print(line)

    def test_fastcharge(self):
        filename = '20170630-3_6C_9per_5C.sdu'
        test_file = 'test.sdu'
        sdu = Schedule.from_fast_charge(1.1 * 3.6, 0.086, 1.1 * 5, os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        with ScratchDir('.'):
            sdu.to_file(test_file)
            hash1 = hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
            hash2 = hash_file(test_file)
            if hash1 != hash2:
                original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').readlines()
                parsed = open(test_file, encoding='latin-1').readlines()
                udiff = list(difflib.unified_diff(original, parsed))
                for line in udiff:
                    print(line)
                self.assertFalse(udiff)