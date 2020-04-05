# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Generating Arbin Schedule files"""

import os
import unittest
from beep import PROCEDURE_TEMPLATE_DIR, SCHEDULE_TEMPLATE_DIR
from beep.generate_protocol import ProcedureFile
from beep.protocol_tools.maccor_to_arbin import ProcedureToSchedule

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class ProcedureToScheduleTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_single_step_conversion(self):
        procedure = ProcedureFile()

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
        procedure = ProcedureFile()

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
        procedure = ProcedureFile()

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
