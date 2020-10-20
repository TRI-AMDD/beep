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
"""Unit tests for maccor protcol files to biologic modulo bat protcol files"""

import os
import unittest
import xmltodict
from beep.protocol.maccor_to_biologic_mb import MaccorToBiologicMb


class ConversionTest(unittest.TestCase):
    def proc_step_to_seq_test(self, test_step_xml, diff_dict):
        """
        test utility for testing proc_step_to_seq
         """
        proc = xmltodict.parse(test_step_xml)
        test_step = proc["MaccorTestProcedure"]["ProcSteps"]["TestStep"]
        converter = MaccorToBiologicMb()

        expected = {}
        expected.update(converter.blank_seq)
        expected.update(diff_dict)

        result = converter._proc_step_to_seq(test_step, "0",)
        for key, value in expected.items():
            self.assertEqual(
                value,
                result[key],
                msg="Expected {0}: {1} got {0}: {2}".format(key, value, result[key]),
            )

    def test_rest_step_conversion(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<MaccorTestProcedure>"
            "  <ProcSteps>"
            "    <TestStep>"
            "      <StepType>  Rest  </StepType>"
            "      <StepMode>        </StepMode>"
            "      <StepValue></StepValue>"
            "      <Limits/>"
            "      <Ends>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&gt;= </Oper>"
            "          <Step>070</Step>"
            "          <Value>4.4</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&lt;= </Oper>"
            "          <Step>070</Step>"
            "          <Value>2.5</Value>"
            "        </EndEntry>"
            "      </Ends>"
            "      <Reports>"
            "        <ReportEntry>"
            "          <ReportType>Voltage</ReportType>"
            "          <Value>2.2</Value>"
            "        </ReportEntry>"
            "      </Reports>"
            "      <Range>A</Range>"
            "      <Option1>N</Option1>"
            "      <Option2>N</Option2>"
            "      <Option3>N</Option3>"
            "      <StepNote></StepNote>"
            "    </TestStep>"
            "  </ProcSteps>"
            "</MaccorTestProcedure>"
        )
        diff_dict = {
            "ctrl_type": "Rest",
            "Apply I/C": "I",
            "N": "1.00",
            "charge/discharge": "Charge",
            "lim_nb": "2",
            "lim1_type": "Voltage",
            "lim1_comp": ">",
            "lim1_value": "4.4",
            "lim1_value_unit": "V",
            "lim2_type": "Voltage",
            "lim2_comp": "<",
            "lim2_value": "2.5",
            "lim2_value_unit": "V",
            "rec_nb": "1",
            "rec1_type": "Voltage",
            "rec1_value": "2.2",
            "rec1_value_unit": "V",
        }

        self.proc_step_to_seq_test(xml, diff_dict)
        pass

    def test_discharge_current_step_conversion(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<MaccorTestProcedure>"
            "  <ProcSteps>"
            "    <TestStep>"
            #      mispelling taken directly from sample file
            "      <StepType>Dischrge</StepType>"
            "      <StepMode>Current </StepMode>"
            "      <StepValue>1.0</StepValue>"
            "      <Limits/>"
            "      <Ends>"
            "        <EndEntry>"
            "          <EndType>StepTime</EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper> = </Oper>"
            "          <Step>013</Step>"
            "          <Value>00:00:30</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&lt;= </Oper>"
            "          <Step>017</Step>"
            "          <Value>2.7</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&gt;= </Oper>"
            "          <Step>070</Step>"
            "          <Value>4.4</Value>"
            "        </EndEntry>"
            "      </Ends>"
            "      <Reports>"
            "        <ReportEntry>"
            "          <ReportType>Voltage </ReportType>"
            "          <Value>0.001</Value>"
            "        </ReportEntry>"
            "        <ReportEntry>"
            "          <ReportType>StepTime</ReportType>"
            #          10ms
            "          <Value>::.01</Value>"
            "        </ReportEntry>"
            "      </Reports>"
            "      <Range>A</Range>"
            "      <Option1>N</Option1>"
            "      <Option2>N</Option2>"
            "      <Option3>N</Option3>"
            "      <StepNote></StepNote>"
            "    </TestStep>"
            "  </ProcSteps>"
            "</MaccorTestProcedure>"
        )
        diff_dict = {
            "ctrl_type": "CC",
            "Apply I/C": "C / N",
            "ctrl1_val": "1.0",
            "ctrl1_val_unit": "A",
            "ctrl1_val_vs": "<None>",
            "N": "15.00",
            "charge/discharge": "Discharge",
            "lim_nb": "3",
            "lim1_type": "Time",
            "lim1_comp": ">",
            "lim1_value": "30000",
            "lim1_value_unit": "ms",
            "lim2_type": "Voltage",
            "lim2_comp": "<",
            "lim2_value": "2.7",
            "lim2_value_unit": "V",
            "lim3_type": "Voltage",
            "lim3_comp": ">",
            "lim3_value": "4.4",
            "lim3_value_unit": "V",
            "rec_nb": "2",
            "rec1_type": "Voltage",
            "rec1_value": "0.001",
            "rec1_value_unit": "V",
            "rec2_type": "Time",
            "rec2_value": "10",
            "rec2_value_unit": "ms",
        }

        self.proc_step_to_seq_test(xml, diff_dict)
        pass
