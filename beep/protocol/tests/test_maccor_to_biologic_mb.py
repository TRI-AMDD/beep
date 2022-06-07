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
from typing import OrderedDict
import unittest
import xmltodict
import copy
import pandas as pd
from monty.tempfile import ScratchDir
from pydash import get
from beep.protocol import (
    PROTOCOL_SCHEMA_DIR,
    BIOLOGIC_TEMPLATE_DIR,
    PROCEDURE_TEMPLATE_DIR,
)
from beep.protocol.maccor import Procedure
from beep.protocol.maccor_to_biologic_mb import (
    MaccorToBiologicMb,
    CycleAdvancementRules,
    CycleAdvancementRulesSerializer
)
from beep.tests.constants import TEST_FILE_DIR


class ConversionTest(unittest.TestCase):
    maxDiff = None

    def maccor_values_to_biologic_value_and_unit_test(self, func, tests):
        for value_str, expected_value_str, expected_unit in tests:
            actual_value, actual_unit = func(value_str)            
            self.assertEqual(actual_value, expected_value_str)
            self.assertEqual(actual_unit, expected_unit)

    def test_convert_volts(self):
        converter = MaccorToBiologicMb()
        tests = [
            ("0.1429", "142.900", "mV"),
            ("0.1429e3", "142.900", "V"),
            ("159.3624", "159362.400", "mV"),
            ("152.9", "152.900",  "V")
        ]
        self.maccor_values_to_biologic_value_and_unit_test(
            converter._convert_volts,
            tests,
        )
    
    def test_convert_amps(self):
        converter = MaccorToBiologicMb()
        tests = [
            ("0.1429", "142.900", "mA"),
            ("1.23", "1.230", "A"),
            ("152.9", "152.900",  "A"),
            ("1.2e-4", "120.000", "\N{Micro Sign}A")
        ]
        self.maccor_values_to_biologic_value_and_unit_test(
            converter._convert_amps,
            tests,
        )

    def test_convert_watts(self):
        converter = MaccorToBiologicMb()
        tests = [
            ("0.1429", "142.900", "mW"),
            ("1.23", "1.230", "W"),
            ("152.9", "152.900",  "W"),
            ("1.2e-5", "12.000", "\N{Micro Sign}W")
        ]
        self.maccor_values_to_biologic_value_and_unit_test(
            converter._convert_watts,
            tests,
        )

    def test_convert_ohms(self):
        converter = MaccorToBiologicMb()
        tests = [
            ("0.1429", "142.900", "mOhms"),
            ("1.459e4", "14.590", "kOhms"),
            ("152.9", "152.900",  "Ohms"),
            ("1.2e-4", "120.000", "\N{Micro Sign}Ohms")
        ]
        self.maccor_values_to_biologic_value_and_unit_test(
            converter._convert_ohms,
            tests,
        )
    
    def test_convert_time(self):
        converter = MaccorToBiologicMb()
        tests = [
            ("::.01", "10.000", "ms"),
            ("03::", "3.000", "h"),
            ("03:30:", "210.000",  "mn"),
            ("00:00:50", "50.000", "s")
        ]
        self.maccor_values_to_biologic_value_and_unit_test(
            converter._convert_time,
            tests,
        )

    def single_step_to_single_seq_test(self, test_step_xml, diff_dict):
        """
        test utility for testing proc_step_to_seq
         """
        proc = xmltodict.parse(test_step_xml)
        test_step = proc["MaccorTestProcedure"]["ProcSteps"]["TestStep"]
        converter = MaccorToBiologicMb()

        expected = converter._blank_seq.copy()
        expected["Ns"] = 0
        expected["lim1_seq"] = 1
        expected["lim2_seq"] = 1
        expected["lim3_seq"] = 1
        expected.update(diff_dict)

        step_num = 1
        seq_nums_by_step_num = {
            step_num: [0],
            step_num + 1: [1],
        }

        result = converter._convert_step_parts(
            step_parts=[test_step],
            step_num=step_num,
            seq_nums_by_step_num=seq_nums_by_step_num,
            goto_lowerbound=0,
            goto_upperbound=3,
            end_step_num=4,
        )[0]

        for key, value in expected.items():
            self.assertEqual(
                value,
                result[key],
                msg="Expected {0}: {1} got {0}: {2}".format(key, value, result[key]),
            )

    def test_partition_steps_into_techniques(self):
        converter = MaccorToBiologicMb()
        ast = converter.load_maccor_ast(
            os.path.join(PROCEDURE_TEMPLATE_DIR, "diagnosticV5.000")
        )
        steps = get(ast, "MaccorTestProcedure.ProcSteps.TestStep")
        self.assertEqual(True, len(steps) > 71)
        
        # existence of looped tech 2
        nested_loop_open_idx = 36
        nested_loop_open_type = get(steps[nested_loop_open_idx], 'StepType')
        self.assertEqual(nested_loop_open_type, "Do 1")
        nested_loop_close_idx = 68
        nested_loop_close_type = get(steps[nested_loop_close_idx], 'StepType')
        self.assertEqual(nested_loop_close_type, "Loop 1")


        technique_partitions = converter._partition_steps_into_techniques(steps)
        self.assertEqual(3, len(technique_partitions))
        partition1, partition2, partition3 =  technique_partitions
        
        self.assertEqual(partition1.technique_num, 1)
        self.assertEqual(partition2.technique_num, 2)
        self.assertEqual(partition3.technique_num, 4)

        self.assertEqual(partition1.tech_does_loop, False)
        self.assertEqual(partition2.tech_does_loop, True)
        self.assertEqual(partition3.tech_does_loop, False)

        self.assertEqual(partition2.num_loops, 1000)

        self.assertEqual(partition1.step_num_offset, 0)
        self.assertEqual(partition2.step_num_offset, nested_loop_open_idx + 1)
        self.assertEqual(partition3.step_num_offset, nested_loop_close_idx + 1)

        self.assertEqual(len(partition1.steps), 36)
        # trim opening/closing loops
        self.assertEqual(len(partition2.steps), nested_loop_close_idx - nested_loop_open_idx - 1)
        self.assertEqual(len(partition3.steps), 27)

    def test_apply_step_mappings_global_noop(self):
        xml = (step_with_bounds_template).format(
            voltage_v_lowerbound = 2.2,
            voltage_v_upperbound = 4.2,
            current_a_lowerbound = 0.1,
            current_a_upperbound = 1.0,
        )
        step = get(
            xmltodict.parse(xml, process_namespaces=False, strip_whitespace=True),
            'TestStep',
        )
        converter = MaccorToBiologicMb()
        
        # no limits no mappings
        unmapped_step = converter._apply_step_mappings([step])[0]
        self.assertEqual(step, unmapped_step)

        # limits outside of bounds, don't
        converter.max_voltage_v = 10.0
        converter.min_voltage_v = -10.0
        converter.max_current_a = 10.0
        converter.min_current_a = -10.0
        unmapped_step = converter._apply_step_mappings([step])[0]
        self.assertEqual(step, unmapped_step)
    
    def test_apply_step_mappings_global_voltage(self):
        xml = (step_with_bounds_template).format(
            voltage_v_lowerbound = 2.2,
            voltage_v_upperbound = 4.2,
            current_a_lowerbound = 0.1,
            current_a_upperbound = 1.0,
        )
        step = get(
            xmltodict.parse(xml, process_namespaces=False, strip_whitespace=True),
            'TestStep',
        )
        converter = MaccorToBiologicMb()

        converter.max_voltage_v = 3.9
        converter.min_voltage_v = 3.1
        step_without_voltage_end_entries = converter._apply_step_mappings([step])[0]

        end_entries = get(step_without_voltage_end_entries, "Ends.EndEntry")
        self.assertEqual(2, len(end_entries))
        self.assertEqual("Current", get(end_entries[0], "EndType"))
        self.assertEqual("Current", get(end_entries[1], "EndType"))

        # check there was not mutation
        original_end_entries = get(step, 'Ends.EndEntry')
        self.assertEqual(4, len(original_end_entries))
    
    def test_apply_step_mappings_all_global_limits(self):
        xml = (step_with_bounds_template).format(
            voltage_v_lowerbound = 2.2,
            voltage_v_upperbound = 4.2,
            current_a_lowerbound = 0.1,
            current_a_upperbound = 1.0,
        )
        step = get(
            xmltodict.parse(xml, process_namespaces=False, strip_whitespace=True),
            'TestStep',
        )
        converter = MaccorToBiologicMb()


        converter.max_voltage_v = 3.9
        converter.min_voltage_v = 3.1
        converter.max_current_a = 0.7
        converter.min_current_a = 0.3

        step_with_no_end_entries = converter._apply_step_mappings([step])[0]
        self.assertEqual(None, get(step_with_no_end_entries, "Ends.EndEntry"))

        # check there was not mutation
        original_end_entries = get(step, 'Ends.EndEntry')
        self.assertEqual(4, len(original_end_entries))

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
            "          <Step>002</Step>"
            "          <Value>4.4</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&lt;= </Oper>"
            "          <Step>002</Step>"
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
            "lim_nb": 2,
            "lim1_type": "Ecell",
            "lim1_comp": ">",
            "lim1_value": "4.400",
            "lim1_value_unit": "V",
            "lim2_type": "Ecell",
            "lim2_comp": "<",
            "lim2_value": "2.500",
            "lim2_value_unit": "V",
            "rec_nb": 1,
            "rec1_type": "Ecell",
            "rec1_value": "2.200",
            "rec1_value_unit": "V",
        }

        self.single_step_to_single_seq_test(xml, diff_dict)
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
            "          <Step>002</Step>"
            "          <Value>00:00:30</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&lt;= </Oper>"
            "          <Step>002</Step>"
            "          <Value>2.7</Value>"
            "        </EndEntry>"
            "        <EndEntry>"
            "          <EndType>Voltage </EndType>"
            "          <SpecialType> </SpecialType>"
            "          <Oper>&gt;= </Oper>"
            "          <Step>002</Step>"
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
            "Apply I/C": "I",
            "ctrl1_val": "1.000",
            "ctrl1_val_unit": "A",
            "ctrl1_val_vs": "<None>",
            "N": "15.00",
            "charge/discharge": "Discharge",
            "lim_nb": 3,
            "lim1_type": "Time",
            "lim1_comp": ">",
            "lim1_value": "30.000",
            "lim1_value_unit": "s",
            "lim2_type": "Ecell",
            "lim2_comp": "<",
            "lim2_value": "2.700",
            "lim2_value_unit": "V",
            "lim3_type": "Ecell",
            "lim3_comp": ">",
            "lim3_value": "4.400",
            "lim3_value_unit": "V",
            "rec_nb": 2,
            "rec1_type": "Ecell",
            "rec1_value": "1.000",
            "rec1_value_unit": "mV",
            "rec2_type": "Time",
            "rec2_value": "10.000",
            "rec2_value_unit": "ms",
        }

        self.single_step_to_single_seq_test(xml, diff_dict)
        pass

    def test_conversion_with_updated(self):
        converter = MaccorToBiologicMb()

        with ScratchDir(".") as scratch_dir:
            # Generate a protocol that can be used with the existing cells for testing purposes
            reg_params = {
                'project_name': {0: 'FormDegrade'},
                'seq_num': {0: 0},
                'template': {0: 'diagnosticV5.000'},
                'charge_constant_current_1': {0: 2.0},
                'charge_percent_limit_1': {0: 30},
                'charge_constant_current_2': {0: 2.0},
                'charge_cutoff_voltage': {0: 4.4},
                'charge_constant_voltage_time': {0: 60},
                'charge_rest_time': {0: 5},
                'discharge_constant_current': {0: 1.0},
                'discharge_cutoff_voltage': {0: 3.0},
                'discharge_rest_time': {0: 15},
                'cell_temperature_nominal': {0: 25},
                'cell_type': {0: 'LiFun240'},
                'capacity_nominal': {0: 0.240},
                'diagnostic_type': {0: 'HPPC+RPT'},
                'diagnostic_parameter_set': {0: 'LiFunForm'},
                'diagnostic_start_cycle': {0: 30},
                'diagnostic_interval': {0: 100}
                          }

            protocol_params_df = pd.DataFrame.from_dict(reg_params)
            index = 0
            protocol_params = protocol_params_df.iloc[index]
            diag_params_df = pd.read_csv(
                os.path.join(PROCEDURE_TEMPLATE_DIR, "PreDiag_parameters - DP.csv")
            )
            diagnostic_params = diag_params_df[
                diag_params_df["diagnostic_parameter_set"]
                == protocol_params["diagnostic_parameter_set"]
                ].squeeze()

            procedure = Procedure.generate_procedure_regcyclev3(index, protocol_params)
            procedure.generate_procedure_diagcyclev3(
                protocol_params["capacity_nominal"], diagnostic_params
            )
            procedure.set_skip_to_end_diagnostic(4.5, 2.0, step_key="070", new_step_key="095")
            procedure.to_file(os.path.join(scratch_dir, "BioTest_000001.000"))

            # Setup the converter and run it
            def set_i_range(tech_num, seq, idx):
                seq_copy = copy.deepcopy(seq)
                seq_copy["I Range"] = "1 A"
                return seq_copy
            converter.seq_mappers.append(set_i_range)
            converter.min_voltage_v = 2.0
            converter.max_voltage_v = 4.5

            converter.convert(os.path.join(scratch_dir, "BioTest_000001.000"),
                              TEST_FILE_DIR, "BioTest_000001")
            f = open(os.path.join(TEST_FILE_DIR, "BioTest_000001.mps"), encoding="ISO-8859-1")
            file = f.readlines()
            control_list = [
                'ctrl_type', 'Rest', 'CC', 'Rest', 'CC', 'CV', 'CC', 'Loop', 'CC', 'CV', 'Rest', 'CC',
                'Rest', 'CC', 'CC', 'Loop', 'CV', 'CC', 'CC', 'CV', 'CC', 'CC', 'CV', 'CC', 'CC', 'CV',
                'CC', 'CC', 'CC', 'CV', 'Rest', 'CC', 'Rest', 'Loop'
            ]
            self.assertListEqual(control_list, file[35].split())
            value_list = [
                'ctrl1_val', '240.000', '34.300', '4.400', '34.300', '100.000', '80.000', '4.400', '240.000',
                '180.000', '80.000', '100.000', '3.000', '80.000', '48.000', '4.400', '48.000', '48.000', '4.400',
                '240.000', '48.000', '4.400', '480.000', '480.000', '480.000', '4.400', '240.000', '100.000'
            ]
            self.assertListEqual(value_list, file[37].split())

            voltage_min = '\tEcell min = 2.00 V\n'
            self.assertEqual(voltage_min, file[9])
            voltage_max = '\tEcell max = 4.50 V\n'
            self.assertEqual(voltage_max, file[10])

    def test_cycle_transition_serialization(self):
        cycle_transition_rules = CycleAdvancementRules(
            tech_num=2,
            tech_does_loop=True,
            adv_cycle_on_start=1,
            adv_cycle_on_tech_loop=1,
            adv_cycle_seq_transitions={(2, 5): 1, (14, 17): 1},
            debug_adv_cycle_on_step_transitions={(72, 71): 1, (72, 75): 1},
        )

        serializer = CycleAdvancementRulesSerializer()
        json_str = serializer.json(cycle_transition_rules)
        parsed_cycle_transition_rules = serializer.parse_json(json_str)

        self.assertEqual(
            cycle_transition_rules.__repr__(),
            parsed_cycle_transition_rules.__repr__(),
        )

    def test_convert_protocol_with_goto_end_step(self):
        path = os.path.join(TEST_FILE_DIR, "goto_end_example.000")
        converter = MaccorToBiologicMb()
        ast = converter.load_maccor_ast(path)
        test_steps = get(ast, "MaccorTestProcedure.ProcSteps.TestStep")
        assert isinstance(test_steps, list)

        first_step = test_steps[0]
        first_step_goto = int(get(first_step, "Ends.EndEntry.Step"))
        assert first_step_goto == len(test_steps)
        assert first_step_goto not in [1, 2]

        converted_fp, rule_fps = converter.convert(path, TEST_FILE_DIR, "to_delete")
        os.remove(converted_fp)
        for rule_fp in rule_fps:
            os.remove(rule_fp)



step_with_bounds_template = (
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<TestStep>\n"
    #  mispelling taken directly from sample file
    "  <StepType>Dischrge</StepType>\n"
    "  <StepMode>Current </StepMode>\n"
    "  <StepValue>1.0</StepValue>\n"
    "  <Limits/>\n"
    "  <Ends>\n"
    "    <EndEntry>\n"
    "      <EndType>Voltage </EndType>\n"
    "      <SpecialType> </SpecialType>\n"
    "      <Oper>&lt;= </Oper>\n"
    "      <Step>002</Step>\n"
    "      <Value>{voltage_v_lowerbound}</Value>\n"
    "    </EndEntry>\n"
    "    <EndEntry>\n"
    "      <EndType>Voltage </EndType>\n"
    "      <SpecialType> </SpecialType>\n"
    "      <Oper>&gt;= </Oper>\n"
    "      <Step>002</Step>\n"
    "      <Value>{voltage_v_upperbound}</Value>\n"
    "    </EndEntry>\n"
    "    <EndEntry>\n"
    "      <EndType>Current </EndType>\n"
    "      <SpecialType> </SpecialType>\n"
    "      <Oper>&lt;= </Oper>\n"
    "      <Step>002</Step>\n"
    "      <Value>{current_a_lowerbound}</Value>\n"
    "    </EndEntry>\n"
    "    <EndEntry>\n"
    "      <EndType>Current </EndType>\n"
    "      <SpecialType> </SpecialType>\n"
    "      <Oper>&gt;= </Oper>\n"
    "      <Step>002</Step>\n"
    "      <Value>{current_a_upperbound}</Value>\n"
    "    </EndEntry>\n"
    "  </Ends>\n"
    "  <Reports></Reports>\n"
    "  <Range>A</Range>\n"
    "  <Option1>N</Option1>\n"
    "  <Option2>N</Option2>\n"
    "  <Option3>N</Option3>\n"
    "  <StepNote></StepNote>\n"
    "</TestStep>\n"
)