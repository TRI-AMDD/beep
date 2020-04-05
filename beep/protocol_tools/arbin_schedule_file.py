# Copyright 2019 Toyota Research Institute. All rights reserved.
""" Schedule file parsing and parameter insertion"""


import os
import hashlib
import re
from datetime import datetime
import numbers
from beep_oed import SCHEMA_FILE_DIR
from collections import OrderedDict, defaultdict
from monty.serialization import loadfn


# TODO: Should this be called a schedule file generator?
#       It doesn't seem to be represenative of a schedule
#       file.  Alternatively, we could have a "schedule"
#       object with appropriate serialization methods
#       which could mimic the design patterns of BEEP-EP
#       more closely.
class ScheduleFile:
    """
    Schedule file utility. Provides the ability to read an Arbin type schedule file

    Args:
        version (str): Default version.
        section_regex (raw str): regex string to return all section headers from
            the schedule file
        step_regex (raw str): regex string to return all step headers from the
            schedule file
        limit_regex (raw str): regex string to return all limit headers from
            the schedule file

    """
    def __init__(self,
                 version=None,
                 section_regex=r'(?<=\[).*',
                 step_regex=r'.Schedule_Step[0-9]*',
                 limit_regex=r'.Schedule_Step[0-9]*_Limit[0-9]*'):
        self.service = version
        self.section = section_regex
        self.step = step_regex
        self.limit = limit_regex
        self.schedule_file_encoding = 'latin-1'

    def hash_file(self, inputfile):
        with open(inputfile, 'rb') as f:
            chunk = f.read()
        return hashlib.md5(chunk).digest()

    def to_dict(self, inputfile):
        """Schedule file ingestion. Converts a schedule file with section headers
        to an ordered dict with section headers as nested dicts. One line in the
        schedule file is not parsable by utf-8. This line is stored and returned
        separately with the line number that it came from

        Args:
            inputfile (str): Schedule file (tested with FastCharge schedule file)

        Returns:
            sdu_dict (dict): Ordered dictionary with keys corresponding to options
            or control variables. Section headers are nested dicts within the dict

        """
        sdu_dict = defaultdict(dict)
        f = open(inputfile, 'rb')
        lines = f.readlines()
        keys = []
        for line_num, line in enumerate(lines):
            try:
                line_plain = line.decode(self.schedule_file_encoding)
            except UnicodeDecodeError:
                print('Wrong encoding for schedule file at line: ' + str(line_num))
            if re.search(self.section, line_plain) is not None:
                if re.search(self.step, line_plain) is not None:
                    if re.search(self.limit, line_plain) is not None:
                        if len(keys) >= 3:
                            keys = keys[:3]
                            keys[2] = line_plain.strip('\r\n')
                            sdu_dict[keys[0]][keys[1]][keys[2]] = OrderedDict({})
                        else:
                            keys.append(line_plain.strip('\r\n'))
                            sdu_dict[keys[0]][keys[1]][keys[2]] = OrderedDict({})
                    else:
                        if len(keys) >= 2:
                            keys = keys[:2]
                            keys[1] = line_plain.strip('\r\n')
                            sdu_dict[keys[0]][keys[1]] = OrderedDict({})
                        else:
                            keys.append(line_plain.strip('\r\n'))
                            sdu_dict[keys[0]][keys[1]] = OrderedDict({})
                else:
                    if len(keys) >= 1:
                        keys = keys[:1]
                        keys[0] = line_plain.strip('\r\n')
                    else:
                        keys.append(line_plain.strip('\r\n'))
            else:
                key, value = line_plain.split('=', 1)
                # print(sdu_dict)
                # if len(keys) == 0:
                #     sdu_dict[keys[0]] = OrderedDict()
                if len(keys) == 1:
                    sdu_dict[keys[0]][key] = value.strip('\r\n')
                if len(keys) == 2:
                    sdu_dict[keys[0]][keys[1]][key] = value.strip('\r\n')
                if len(keys) == 3:
                    sdu_dict[keys[0]][keys[1]][keys[2]][key] = value.strip('\r\n')

        sdu_dict = OrderedDict(sdu_dict)
        return sdu_dict

    def dict_to_file(self, dict_obj, outputfile):
        """
        Schedule file output. Converts an dictionary to a schedule file with
        the appropriate section headers. The one line in the schedule file that is
        not parsable is reinserted at the correct line number. This function
        DOES NOT check the flow control or limits set in the steps. The dictionary
        must represent a valid schedule before it is passed to this function.

        Args:
            dict_obj (dict): Ordered dictionary containing all of the schedule file sections with keys
                and values. Nested dicts correspond to sections
            outputfile (str): File string corresponding to the file to output the schedule to
        """
        f = open(outputfile, 'wb')
        dict_obj.keys()
        for key_line in dict_obj.keys():
            if re.search(self.section, key_line) is not None:
                line = key_line + '\r\n'
                f.write(line.encode(self.schedule_file_encoding))
                key_lines_1 = dict_obj[key_line].keys()
                for key_line_1 in key_lines_1:

                    if re.search(self.step, key_line_1) is not None:
                        line = key_line_1 + '\r\n'
                        f.write(line.encode(self.schedule_file_encoding))
                        key_lines_2 = dict_obj[key_line][key_line_1].keys()
                        for key_line_2 in key_lines_2:
                            if re.search(self.limit, key_line_2) is not None:
                                line = key_line_2 + '\r\n'
                                f.write(line.encode(self.schedule_file_encoding))
                                for key_line_3 in dict_obj[key_line][key_line_1][key_line_2].keys():
                                    line = key_line_3 + '=' + \
                                           dict_obj[key_line][key_line_1][key_line_2][key_line_3] + \
                                           '\r\n'
                                    f.write(line.encode(self.schedule_file_encoding))
                            else:
                                line = key_line_2 + '=' + \
                                       dict_obj[key_line][key_line_1][key_line_2] + \
                                       '\r\n'
                                f.write(line.encode(self.schedule_file_encoding))
                    else:
                        line = key_line_1 + '=' + \
                               dict_obj[key_line][key_line_1] + \
                               '\r\n'
                        f.write(line.encode(self.schedule_file_encoding))
        f.close()

    def fast_charge_file(self, CC1, CC1_capacity, CC2, inputname, outputname):
        """
        Function takes parameters for the FastCharge Project
        and creates the schedule files necessary to run each of
        these parameter combinations. Assumes that control type
        is CCCV.

        Args:
            CC1 (float): Constant current value for charge section 1
            CC1_capacity (float): Capacity to charge to for section 1
            CC2 (float): Constant current value for charge section 2
            inputname (str): File path to pull the template schedule
                file from
            outputname (str): File path to save the parameterized
                schedule file to

        """

        templates = os.path.join(os.path.dirname(__file__), 'sdu_templates')
        schedules = os.path.join(os.path.dirname(__file__), 'schedules')

        # TODO replace with less brittle file integrity check
        # hash_test = self.hash_file(os.path.join(templates, inputname))
        # print(hash_test)
        # hash_fast_charge_struct1 = b'\xe0\xcb;\x9d\x87JS\n\xd9\xbf\xb4\x08\x1d\xb0\x9av'
        # assert hash_fast_charge_struct1 == hash_test, "Input file is different than expected"

        sdu_dict = self.to_dict(os.path.join(templates, inputname))
        sdu_dict = self.step_values(sdu_dict, 'CC1', 'm_szCtrlValue',
                                    step_value='{0:.3f}'.format(CC1).rstrip('0'))
        sdu_dict = self.step_limit_values(sdu_dict,
                                          'CC1',
                                          'PV_CHAN_Charge_Capacity',
                                          {'compare': '>',
                                           'value': '{0:.3f}'.format(CC1_capacity).rstrip('0')}
                                          )
        sdu_dict = self.step_values(sdu_dict, 'CC2', 'm_szCtrlValue',
                                    step_value='{0:.3f}'.format(CC2).rstrip('0'))
        self.dict_to_file(sdu_dict, os.path.join(schedules, outputname))

    def step_values(self, sdu_dict, step_label, step_key, step_value=None):
        """
        Insert values for steps in the schedule section

        Args:
            sdu_dict (dict): Ordered dictionary containing all of the schedule file
            step_label (str): The user determined step label for the step. If there are multiple
                identical labels this will operate on the first one it encounters
            step_key (int): Key in the step to set, e.g. ('m_szStepCtrlType')
            step_value (str): Value to set for the key

        Returns:
            dict: Altered ordered dictionary with keys corresponding to options or control
                variables.
        """
        values = []
        s = '[Schedule]'
        for sch_keys in sdu_dict[s].keys():
            if re.search(self.step, sch_keys) and sdu_dict[s][sch_keys]['m_szLabel'] == step_label:
                if step_value is not None:
                    sdu_dict[s][sch_keys][step_key] = step_value
                values.append(sdu_dict[s][sch_keys][step_key])
        return sdu_dict

    def step_limit_values(self, sdu_dict, step_label, limit_var, limit_set=None):
        """Insert values for the limits in the steps in the schedule section

        Args:
            sdu_dict (dict): Ordered dictionary containing all of the schedule file
            step_label (str): The user determined step label for the step. If there
                are multiple identical labels this will operate on the first one it
                encounters
            limit_var (str): Variable being used for this particular limit in the step
            limit_set (dict): Value comparison to trip the limit (evaluating to True
                triggers the limit) {'compare': '>', 'value': '0.086'}

        Returns:
            dict: Altered ordered dictionary with keys corresponding to options or control
                variables.
        """
        values = []
        s = '[Schedule]'
        equ = 'Equation0_sz'
        for sch_keys in sdu_dict[s].keys():
            if re.search(self.step, sch_keys) and sdu_dict[s][sch_keys]['m_szLabel'] == step_label:
                for step_keys in sdu_dict[s][sch_keys].keys():
                    if re.search(self.limit, step_keys) and \
                            sdu_dict[s][sch_keys][step_keys][equ + 'Left'] == limit_var and \
                            sdu_dict[s][sch_keys][step_keys]['m_bStepLimit'] == '1':
                        if limit_set is not None:
                            sdu_dict[s][sch_keys][step_keys][equ + 'CompareSign'] = limit_set['compare']
                            sdu_dict[s][sch_keys][step_keys][equ + 'Right'] = limit_set['value']
                        values.append(sdu_dict[s][sch_keys][step_keys][equ + 'Left'] +
                                      sdu_dict[s][sch_keys][step_keys][equ + 'CompareSign'] +
                                      sdu_dict[s][sch_keys][step_keys][equ + 'Right'])
                    elif re.search(self.limit, step_keys) and \
                            sdu_dict[s][sch_keys][step_keys]['m_bStepLimit'] == '1':
                        print('Warning, additional step limit: ' +
                              sdu_dict[s][sch_keys][step_keys][equ + 'Left'] +
                              sdu_dict[s][sch_keys][step_keys][equ + 'CompareSign'] +
                              sdu_dict[s][sch_keys][step_keys][equ + 'Right'])
        return sdu_dict


def compile_to_arbin(step_abs, step_index, step_name_list, step_flow_ctrl, range='Parallel-High'):
    ARBIN_SCHEMA = loadfn(os.path.join(SCHEMA_FILE_DIR, "arbin_schedule_schema.yaml"))
    blank_step = OrderedDict(ARBIN_SCHEMA['step_blank_body'])

    blank_step['m_szLabel'] = str(step_index + 1) + '-' + str(step_abs['StepNote'])
    blank_step['m_szCurrentRange'] = range

    # Current control mode with currents measured in Amps
    if step_abs['StepMode'] == 'Current ' and 'C' not in step_abs['StepValue']:
        if step_abs['Limits'] is not None:
            blank_step['m_szStepCtrlType'] = "CCCV"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['StepValue']
                blank_step['m_szExtCtrlValue1'] = step_abs['Limits']['Voltage']
                blank_step['m_szExtCtrlValue2'] = "0"
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = '-' + step_abs['StepValue']
                blank_step['m_szExtCtrlValue1'] = step_abs['Limits']['Voltage']
                blank_step['m_szExtCtrlValue2'] = "0"
        elif step_abs['Limits'] is None:
            blank_step['m_szStepCtrlType'] = "Current(A)"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['StepValue']
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = '-' + step_abs['StepValue']
        else:
            raise ValueError("Unable to set m_szStepCtrlType for current")

    # Current control mode currents measured in C-rate
    elif step_abs['StepMode'] == 'Current ' and 'C' in step_abs['StepValue']:
        if step_abs['Limits'] is not None:
            blank_step['m_szStepCtrlType'] = "CCCV"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['StepValue'].replace('C', '')
                blank_step['m_szExtCtrlValue1'] = step_abs['Limits']['Voltage']
                blank_step['m_szExtCtrlValue2'] = "0"
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = '-' + step_abs['StepValue'].replace('C', '')
        elif step_abs['Limits'] is None:
            blank_step['m_szStepCtrlType'] = "C-Rate"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['StepValue'].replace('C', '')
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = '-' + step_abs['StepValue'].replace('C', '')
        else:
            raise ValueError("Unable to set m_szStepCtrlType for current")

    # Voltage control mode and current limit measured in Amps
    elif step_abs['StepMode'] == 'Voltage ':
        if step_abs['Limits'] is not None and 'C' not in step_abs['Limits']['Current']:
            blank_step['m_szStepCtrlType'] = "CCCV"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['Limits']['Current']
                blank_step['m_szExtCtrlValue1'] = step_abs['StepValue']
                blank_step['m_szExtCtrlValue2'] = "0"
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = '-' + step_abs['Limits']['Current']
                blank_step['m_szExtCtrlValue1'] = step_abs['StepValue']
                blank_step['m_szExtCtrlValue2'] = "0"
        elif step_abs['Limits'] is None:
            blank_step['m_szStepCtrlType'] = "Voltage(V)"
            if step_abs['StepType'] == ' Charge ':
                blank_step['m_szCtrlValue'] = step_abs['StepValue']
            elif step_abs['StepType'] == 'Dischrge':
                blank_step['m_szCtrlValue'] = step_abs['StepValue']
        else:
            raise ValueError("Unable to set m_szStepCtrlType for voltage")

    # Rest control mode
    elif step_abs['StepMode'] == '        ' and step_abs['StepType'] == '  Rest  ':
        if step_abs['Limits'] is None:
            blank_step['m_szStepCtrlType'] = "Rest"
        else:
            raise ValueError("Unable to set m_szStepCtrlType for voltage")

    # Flow control steps
    elif step_abs['StepMode'] == '        ' and step_abs['StepType'] in [' Loop 1 ', '  Do 1  ',
                                                                         ' Loop 2 ', '  Do 2  ',
                                                                         'AdvCycle', '  End   ']:
        if step_abs['StepType'] == 'AdvCycle':
            blank_step['m_szStepCtrlType'] = "Set Variable(s)"
            blank_step['m_szCtrlValue1'] = '0'
            blank_step['m_szExtCtrlValue1'] = '1'
            blank_step['m_szExtCtrlValue1'] = '0'
        elif 'Loop' in step_abs['StepType']:
            loop_counter = int(re.search(r'\d+', step_abs['StepType']).group())
            blank_step['m_szStepCtrlType'] = "Set Variable(s)"
            blank_step['m_szCtrlValue1'] = '0'
            blank_step['m_szExtCtrlValue1'] = str(2 ** loop_counter)
            blank_step['m_szExtCtrlValue1'] = '0'
            assert isinstance(step_abs['Ends']['EndEntry'], OrderedDict)
            loop_addendum = OrderedDict([('EndType', 'Loop Addendum'), ('Oper', '< '),
                                        ('Step', step_flow_ctrl[step_index].split('-')[0]),
                                         ('Value', step_abs['Ends']['EndEntry']['Value'])])
            step_abs['Ends']['EndEntry'] = [loop_addendum, step_abs['Ends']['EndEntry']]

        elif 'Do' in step_abs['StepType']:
            loop_counter = int(re.search(r'\d+', step_abs['StepType']).group())
            blank_step['m_szStepCtrlType'] = "Set Variable(s)"
            blank_step['m_szCtrlValue1'] = str(2 ** (loop_counter + 15))
            blank_step['m_szExtCtrlValue1'] = '0'
            blank_step['m_szExtCtrlValue1'] = '0'

        else:
            blank_step['m_szStepCtrlType'] = "Rest"
    else:
        raise ValueError("Unable to set StepMode for Flow control step")

    step_type = step_abs['StepType']

    # Ends
    if step_abs['Ends'] is not None:
        if isinstance(step_abs['Ends']['EndEntry'], OrderedDict):
            blank_step['m_uLimitNum'] = 1
            end = step_abs['Ends']['EndEntry']
            end_index = 0
            limit_key = "[Schedule_Step{}_Limit{}]".format(str(step_index), str(end_index))
            blank_step[limit_key] = OrderedDict(convert_end_to_limit(blank_step, end,
                                                                     step_index, step_name_list,
                                                                     step_type, step_flow_ctrl))
        elif isinstance(step_abs['Ends']['EndEntry'], list):
            blank_step['m_uLimitNum'] = len(step_abs['Ends']['EndEntry'])
            for end_index, end in enumerate(step_abs['Ends']['EndEntry']):
                limit_key = "[Schedule_Step{}_Limit{}]".format(str(step_index), str(end_index))
                blank_step[limit_key] = OrderedDict(convert_end_to_limit(blank_step, end,
                                                                         step_index, step_name_list,
                                                                         step_type, step_flow_ctrl))
    elif step_abs['Ends'] is None:
        blank_step['m_uLimitNum'] = 1
        end_index = 0
        limit_key = "[Schedule_Step{}_Limit{}]".format(str(step_index), str(end_index))
        blank_step[limit_key] = add_blank_limit()

    # Reports
    if step_abs['Reports'] is not None:
        if isinstance(step_abs['Reports']['ReportEntry'], OrderedDict):
            blank_step['m_uLimitNum'] = blank_step['m_uLimitNum'] + 1
            report = step_abs['Reports']['ReportEntry']
            report_index = 0
            limit_start = len(step_abs['Ends']['EndEntry'])
            limit_key = "[Schedule_Step{}_Limit{}]".format(str(step_index), str(report_index + limit_start))
            blank_step[limit_key] = OrderedDict(convert_report_to_limit(report))
        elif isinstance(step_abs['Ends']['EndEntry'], list):
            blank_step['m_uLimitNum'] = blank_step['m_uLimitNum'] + len(step_abs['Ends']['EndEntry'])
            for report_index, report in enumerate(step_abs['Reports']['ReportEntry']):
                limit_start = len(step_abs['Ends']['EndEntry'])
                limit_key = "[Schedule_Step{}_Limit{}]".format(str(step_index), str(report_index + limit_start))
                blank_step[limit_key] = OrderedDict(convert_report_to_limit(report))

    return blank_step


def add_blank_limit():
    ARBIN_SCHEMA = loadfn(os.path.join(SCHEMA_FILE_DIR, "arbin_schedule_schema.yaml"))
    limit = ARBIN_SCHEMA['step_blank_limit']
    limit['m_bStepLimit'] = "1"
    limit['m_bLogDataLimit'] = "0"
    limit['m_szGotoStep'] = "Next Step"
    limit['Equation0_szLeft'] = 'PV_CHAN_Step_Time'
    limit['Equation0_szCompareSign'] = '>'
    limit['Equation0_szRight'] = '0'
    return limit


def convert_end_to_limit(blank_step, end, step_index, step_name_list, step_type, step_flow_ctrl):
    ARBIN_SCHEMA = loadfn(os.path.join(SCHEMA_FILE_DIR, "arbin_schedule_schema.yaml"))
    limit = ARBIN_SCHEMA['step_blank_limit']
    limit['m_bStepLimit'] = "1"
    limit['m_bLogDataLimit'] = "1"

    if end['Step'] == str(int(step_index) + 2).zfill(3):
        limit['m_szGotoStep'] = 'Next Step'
    else:
        limit['m_szGotoStep'] = step_name_list[int(end['Step']) - 1]

    if end['EndType'] == 'Voltage ':
        limit['Equation0_szLeft'] = 'PV_CHAN_Voltage'
        limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '')
        limit['Equation0_szRight'] = end['Value']
    elif end['EndType'] == ' Current ' and blank_step['m_szStepCtrlType'] == "CCCV":
        limit['Equation0_szLeft'] = 'PV_CHAN_CV_Stage_Current'
        if step_type == ' Charge ':
            limit['Equation0_szRight'] = end['Value']
            limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '')
        elif step_type == 'Dischrge':
            limit['Equation0_szRight'] = '-' + end['Value']
            limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '').replace('<', '>')
    elif end['EndType'] == ' Current ' and blank_step['m_szStepCtrlType'] == "Voltage(V)":
        limit['Equation0_szLeft'] = 'PV_CHAN_Current'
        if step_type == ' Charge ':
            limit['Equation0_szRight'] = end['Value']
            limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '')
        elif step_type == 'Dischrge':
            limit['Equation0_szRight'] = '-' + end['Value']
            limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '').replace('<', '>')
    elif end['EndType'] == 'StepTime':
        limit['Equation0_szLeft'] = 'PV_CHAN_Step_Time'
        limit['Equation0_szCompareSign'] = '>'
        if '.' in end['Value']:
            nofrag, frag = end['Value'].split(".")
            frag = frag[:6]  # truncate to microseconds
            frag += (6 - len(frag)) * '0'  # add 0s
            elapsed = datetime.strptime(nofrag.replace('::', '00:00:0'), "%H:%M:%S").replace(microsecond=int(frag)) - \
                datetime.strptime("00:00:00", "%H:%M:%S")
        else:
            elapsed = datetime.strptime(end['Value'].replace('::', '00:00:0'), "%H:%M:%S") - \
                datetime.strptime("00:00:00", "%H:%M:%S")
        limit['Equation0_szRight'] = elapsed.total_seconds()
    elif end['EndType'] == 'Loop Cnt':
        loop_counter = int(re.search(r'\d+', step_type).group())
        limit['Equation0_szLeft'] = 'TC_Counter{}'.format(loop_counter)
        limit['Equation0_szCompareSign'] = end['Oper'].replace(' ', '')
        limit['Equation0_szRight'] = end['Value']

    elif end['EndType'] == 'Loop Addendum':
        loop_counter = int(re.search(r'\d+', step_type).group())
        limit['m_szGotoStep'] = step_flow_ctrl[step_index]
        limit['Equation0_szLeft'] = 'TC_Counter{}'.format(loop_counter)
        limit['Equation0_szCompareSign'] = '<'
        limit['Equation0_szRight'] = end['Value']

    else:
        ValueError("Unable to set end for type {}".format(end['EndType']))

    return limit


def convert_report_to_limit(report):
    ARBIN_SCHEMA = loadfn(os.path.join(SCHEMA_FILE_DIR, "arbin_schedule_schema.yaml"))
    limit = ARBIN_SCHEMA['step_blank_limit']
    limit['m_bStepLimit'] = "0"
    limit['m_bLogDataLimit'] = "1"
    limit['m_szGotoStep'] = 'Next Step'

    if report['ReportType'] == 'Voltage ':
        limit['Equation0_szLeft'] = 'DV_Voltage'
        limit['Equation0_szCompareSign'] = '>'
        limit['Equation0_szRight'] = report['Value']
    elif report['ReportType'] == ' Current ':
        limit['Equation0_szLeft'] = 'DV_Current'
        limit['Equation0_szRight'] = report['Value']
        limit['Equation0_szCompareSign'] = '>'
    elif report['ReportType'] == 'StepTime':
        limit['Equation0_szLeft'] = 'DV_Time'
        limit['Equation0_szCompareSign'] = '>'
        if '.' in report['Value']:
            nofrag, frag = report['Value'].split(".")
            frag = frag[:6]  # truncate to microseconds
            frag += (6 - len(frag)) * '0'  # add 0s
            elapsed = datetime.strptime(nofrag.replace('::', '00:00:0'), "%H:%M:%S").replace(microsecond=int(frag)) - \
                datetime.strptime("00:00:00", "%H:%M:%S")
        else:
            elapsed = datetime.strptime(report['Value'].replace('::', '00:00:0'), "%H:%M:%S") - \
                datetime.strptime("00:00:00", "%H:%M:%S")
        limit['Equation0_szRight'] = elapsed.total_seconds()

    return limit


def main():
    sdu = ScheduleFile(version='0.1')
    sdu.fast_charge_file(1.1*3.6, 0.086, 1.1*5, '20170630-3_6C_9per_5C.sdu', 'test.sdu')
    hash1 = sdu.hash_file('/Users/patrickherring/Downloads/20170630-3_6C_9per_5C.sdu')
    hash2 = sdu.hash_file('/Users/patrickherring/Documents/Test.sdu')
    assert hash1 == hash2


if __name__ == "__main__":
    main()
