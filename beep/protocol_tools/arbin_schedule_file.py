# Copyright 2019 Toyota Research Institute. All rights reserved.
""" Schedule file parsing and parameter insertion"""


import os
import hashlib
import re
from beep import SCHEDULE_TEMPLATE_DIR
from collections import OrderedDict, defaultdict


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
                 section_regex=r'(?<=\[).*',
                 step_regex=r'.Schedule_Step[0-9]*',
                 limit_regex=r'.Schedule_Step[0-9]*_Limit[0-9]*'):
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
        to an ordered dict with section headers as nested dicts.

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
            dict_obj (dict): Ordered dictionary containing all of the schedule file
                sections with keys and values. Nested dicts correspond to sections
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
            CC1 (float): Constant current value for charge section 1 in Amps
            CC1_capacity (float): Capacity to charge to for section 1 in Amp-hours
            CC2 (float): Constant current value for charge section 2 in Amps
            inputname (str): File path to pull the template schedule
                file from
            outputname (str): File path to save the parameterized
                schedule file to

        """

        templates = SCHEDULE_TEMPLATE_DIR

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
        self.dict_to_file(sdu_dict, os.path.join(templates, outputname))

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


def main():
    sdu = ScheduleFile()
    sdu.fast_charge_file(1.1 * 3.6, 0.086, 1.1 * 5, '20170630-3_6C_9per_5C.sdu', 'test.sdu')


if __name__ == "__main__":
    main()
