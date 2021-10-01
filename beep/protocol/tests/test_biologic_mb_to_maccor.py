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

import unittest
import os
import json
from beep.protocol.biologic_mb_to_maccor import BiologicMbToMaccorProcedure
from beep.tests.constants import TEST_FILE_DIR

SAMPLE_MB_FILE_NAME = "BCS - 171.64.160.115_Ta19_ourprotocol_gdocSEP2019_CC7.mps"
CONVERTED_OUTPUT_FILE_NAME = "test_biologic_mb_to_maccor_output_diagnostic"


class BiologicMbToMaccorTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        with open(os.path.join(TEST_FILE_DIR, 'biologic_mb_test_sample_mb_text.json'), 'r') as jfile:
            temp_json = json.load(jfile)
        self.sample_mb_text = temp_json["sample_mb_text"]
        self.expected_xml = temp_json["expected_xml"]

    def test_convert_resistance(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "MOhm", "lim3_unit", 1
            ),
            "1.32E6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "kOhm", "lim3_unit", 1
            ),
            "1.32E3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "Ohm", "lim3_unit", 1
            ),
            "1.32",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "mOhm", "lim3_unit", 1
            ),
            "1.32E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "\N{Greek Small Letter Mu}Ohm", "lim3_unit", 1
            ),
            "1.32E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_resistance(
                "1.32", "\N{Micro Sign}Ohm", "lim3_unit", 1
            ),
            "1.32E-6",
        )
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_resistance,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def test_convert_voltage(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_voltage("4.57", "V", "rec2_unit", 2),
            "4.57",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_voltage("4.57", "mV", "rec2_unit", 2),
            "4.57E-3",
        )
        # wrong unit
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_voltage,
            "4.57",
            "mA",
            "rec2_unit",
            2,
        )

    def test_convert_current(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "A", "lim3_unit", 1),
            "2.11",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "mA", "lim3_unit", 1),
            "2.11E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current(
                "2.11", "\N{Greek Small Letter Mu}A", "lim3_unit", 1
            ),
            "2.11E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current(
                "2.11", "\N{Micro Sign}A", "lim3_unit", 1
            ),
            "2.11E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "nA", "lim3_unit", 1),
            "2.11E-9",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_current("2.11", "pA", "lim3_unit", 1),
            "2.11E-12",
        )
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_current,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def test_convert_power(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power("0.560", "W", "rec1_unit", 1),
            "0.560",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power("0.560", "mW", "rec1_unit", 1),
            "0.560E-3",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power(
                "0.560", "\N{Greek Small Letter Mu}W", "rec1_unit", 1
            ),
            "0.560E-6",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_power(
                "0.560", "\N{Micro Sign}W", "rec1_unit", 1
            ),
            "0.560E-6",
        )

    def test_convert_time(self):
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("1", "h", "lim3_unit", 1),
            "1:00:00",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("56", "mn", "lim3_unit", 1),
            "00:56:00",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("1.5", "s", "lim3_unit", 1),
            "00:00:1.5",
        )
        self.assertEqual(
            BiologicMbToMaccorProcedure._convert_time("10", "ms", "lim3_unit", 1),
            "00:00:0.01",
        )

        # too much specificity for maccor to handle
        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_time,
            "1",
            "ms",
            "lim3_unit",
            1,
        )

        self.assertRaises(
            Exception,
            BiologicMbToMaccorProcedure._convert_time,
            "4.57",
            "mV",
            "rec2_unit",
            1,
        )

    def _test_step_helper(self, expected_step, actual_step):
        self.assertEqual(expected_step["StepMode"], actual_step["StepMode"])
        self.assertEqual(expected_step["StepValue"], actual_step["StepValue"])
        self.assertEqual(expected_step["Range"], actual_step["Range"])
        self.assertEqual(expected_step["Option1"], actual_step["Option1"])
        self.assertEqual(expected_step["Option2"], actual_step["Option2"])
        self.assertEqual(expected_step["Option3"], actual_step["Option3"])
        self.assertEqual(expected_step["StepNote"], actual_step["StepNote"])
        # self.assertEqual(actual_step["Reports"]["ReportEntry"][0], "a")
        if type(expected_step["Reports"]) == str:
            self.assertEqual(expected_step["Reports"], actual_step["Reports"])
        else:
            for report_num, report in enumerate(
                expected_step["Reports"]["ReportEntry"]
            ):
                actual_entries = actual_step["Reports"]["ReportEntry"]
                for key, value in report.items():
                    self.assertEqual(
                        value,
                        actual_entries[report_num][key],
                        msg="bad ReportEntry Field: <{}>, Value:{}".format(key, value),
                    )

        if type(expected_step["Ends"]) == str:
            self.assertEqual(expected_step["Ends"], actual_step["Ends"])
        else:
            for end_num, end in enumerate(expected_step["Ends"]["EndEntry"]):
                actual_end_entries = actual_step["Ends"]["EndEntry"]
                for key, value in end.items():
                    self.assertEqual(
                        value,
                        actual_end_entries[end_num][key],
                        msg="bad ReportEntry Field: <{}>, Value:{}".format(key, value),
                    )

    def test_convert_step_rest(self):
        seq = {
            "ctrl_type": "Rest",
            "ctrl1_val": "",
            "ctrl1_val_unit": "",
            "ctrl1_val_vs": "",
            "Ns": "1",
            "Apply I/C": "I",
            "charge/discharge": "Discharge",
            "lim_nb": "1",
            "lim1_type": "Ecell",
            "lim1_comp": ">",
            "lim1_value": "4.4",
            "lim1_value_unit": "V",
            "lim1_action": "Goto sequence",
            "lim1_seq": "3",
            "rec_nb": "1",
            "rec1_type": "I",
            "rec1_value": "2.2",
            "rec1_value_unit": "A",
        }
        seq_num_by_step_num = {1: 1, 3: 5}
        seq_num_is_non_empty_loop_start = set()
        end_step_num = 5

        rest_step = BiologicMbToMaccorProcedure._create_step(
            seq, seq_num_by_step_num, seq_num_is_non_empty_loop_start, end_step_num
        )
        expected_rest_step = {
            "StepType": "  Rest  ",
            "StepMode": "        ",
            "StepValue": "",
            "Limits": "",
            "Ends": {
                "EndEntry": [
                    {
                        "EndType": "Voltage ",
                        "SpecialType": " ",
                        "Oper": ">= ",
                        "Step": "005",
                        "Value": "4.4",
                    }
                ],
            },
            "Reports": {"ReportEntry": [{"ReportType": "Current ", "Value": "2.2"}]},
            "Range": "4",
            "Option1": "N",
            "Option2": "N",
            "Option3": "N",
            "StepNote": "",
        }

        self._test_step_helper(expected_rest_step, rest_step)

    def test_convert_constant_current_step(self):
        seq = {
            "ctrl_type": "CC",
            "ctrl1_val": "100.00",
            "ctrl1_val_unit": "µA",
            "ctrl1_val_vs": "",
            "Ns": "1",
            "Apply I/C": "I",
            "charge/discharge": "Discharge",
            "lim_nb": "1",
            "lim1_type": "Ecell",
            "lim1_comp": ">",
            "lim1_value": "4.4",
            "lim1_value_unit": "V",
            "lim1_action": "End",
            "lim1_seq": "3",
            "rec_nb": "1",
            "rec1_type": "I",
            "rec1_value": "2.2",
            "rec1_value_unit": "A",
        }
        seq_num_by_step_num = {1: 1, 3: 5}
        seq_num_is_non_empty_loop_start = set()
        end_step_num = 5

        constant_current_step = BiologicMbToMaccorProcedure._create_step(
            seq, seq_num_by_step_num, seq_num_is_non_empty_loop_start, end_step_num
        )
        expected_constant_current_step = {
            "StepType": "Dischrge",
            "StepMode": "Current ",
            "StepValue": "100.00E-6",
            "Limits": "",
            "Ends": {
                "EndEntry": [
                    {
                        "EndType": "Voltage ",
                        "SpecialType": " ",
                        "Oper": ">= ",
                        "Step": "005",
                        "Value": "4.4",
                    }
                ],
            },
            "Reports": {"ReportEntry": [{"ReportType": "Current ", "Value": "2.2"}]},
            "Range": "4",
            "Option1": "N",
            "Option2": "N",
            "Option3": "N",
            "StepNote": "",
        }

        self._test_step_helper(expected_constant_current_step, constant_current_step)

    def test_biologic_mb_text_to_maccor_xml(self):
        xml = BiologicMbToMaccorProcedure.biologic_mb_text_to_maccor_xml(self.sample_mb_text)
        xml_lines = xml.splitlines()
        expected_xml_lines = self.expected_xml.splitlines()
        for line_num in range(len(expected_xml_lines)):
            assert line_num < len(xml_lines)
            self.assertEqual(expected_xml_lines[line_num], xml_lines[line_num])

    def test_convert(self):
        # TODO
        #
        # Assert equivalence to a verified file
        source = os.path.join(TEST_FILE_DIR, SAMPLE_MB_FILE_NAME)
        target = os.path.join(TEST_FILE_DIR, CONVERTED_OUTPUT_FILE_NAME)
        BiologicMbToMaccorProcedure.convert(source, target)
        os.remove(target)
