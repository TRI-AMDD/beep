# Copyright 2019 Toyota Research Institute. All rights reserved.
""" Schedule file parsing and parameter insertion"""


import os
import hashlib
import re
from copy import deepcopy
from beep import SCHEDULE_TEMPLATE_DIR
from collections import OrderedDict, defaultdict
from pydash import get, set_, unset


class Schedule(OrderedDict):
    """
    Schedule file utility. Provides the ability to read
    an Arbin type schedule file

    Args:
        # TODO: what are the facts?
        facts (str): Default version.

    """
    def __init__(self, facts):
        super.__init__()
        for fact in facts:
            self[fact[0]] = fact[1]

    @staticmethod
    def hash_file(inputfile):
        with open(inputfile, 'rb') as f:
            chunk = f.read()
        return hashlib.md5(chunk).digest()

    @classmethod
    def from_file(cls, filename, encoding='latin-1'):
        """
        Schedule file ingestion. Converts a schedule file with section headers
        to an ordered dict with section headers as nested dicts. One line in the
        schedule file is not parsable by utf-8. This line is stored and returned
        separately with the line number that it came from

        Args:
            filename (str): Schedule file name (tested with FastCharge schedule file)
            section_regex (raw str): regex string to return all section headers from
                the schedule file
            step_regex (raw str): regex string to return all step headers from the
                schedule file
            limit_regex (raw str): regex string to return all limit headers from
                the schedule file
            encoding (str): encoding of schedule file

        Returns:
            (Schedule): Ordered dictionary with keys corresponding to options
                or control variables. Section headers are nested dicts within
                the dict

        """
        sdu_dict = OrderedDict()
        with open(filename, 'rb') as f:
            # TODO: add error back?
            text = f.read()
            text = text.decode(encoding)

        split_text = re.split(r'\[(.+)\]', text)
        for heading, body in zip(split_text[0::2], split_text[1::2]):
            line_pairs = [line.split('=') for line in body.split()]
            body_dict = OrderedDict(line_pairs)
            # TODO: partition the ordinals as keys as well?
            heading = heading.replace('_', '.')
            set_(sdu_dict, heading, body_dict)

        return sdu_dict

    def to_file(self, filename, encoding="latin-1", linesep="\r\n"):
        """
        Schedule file output. Converts an dictionary to a schedule file with
        the appropriate section headers. The one line in the schedule file that is
        not parsable is reinserted at the correct line number. This function
        DOES NOT check the flow control or limits set in the steps. The dictionary
        must represent a valid schedule before it is passed to this function.

        Args:
            filename (str): string corresponding to the file to
                output the schedule to
            encoding (str): text encoding for the file
            linesep (str): line separator for the file,
                default Windows-compatible "\r\n"

        """
        # Flatten dict
        data = deepcopy(self)
        flat_keys = _get_headings(data, delimiter='.')
        flat_keys.reverse()  # Reverse ensures sub-dicts are removed first
        data_tuples = []
        for flat_key in flat_keys:
            data_tuple = (flat_key.replace('.', '_'), get(data, flat_key))
            data_tuples.append(data_tuple)
            unset(data, flat_key)
        data_tuples.reverse()

        # Construct text
        blocks = []
        for section_title, body_data in data_tuples:
            section_header = "[{}]".format(section_title)
            body = linesep.join(["=".join([key, value])
                                 for key, value in body_data.items()])
            blocks.append(linesep.join([section_header, body]))
        contents = linesep.join(blocks)

        # Write file
        with open(filename, 'wb') as f:
            f.write(contents.encode(encoding))

    @classmethod
    def from_fast_charge(cls, CC1, CC1_capacity, CC2, template_filename,
                         output_filename=None):
        """
        Function takes parameters for the FastCharge Project
        and creates the schedule files necessary to run each of
        these parameter combinations. Assumes that control type
        is CCCV.

        Args:
            CC1 (float): Constant current value for charge section 1
            CC1_capacity (float): Capacity to charge to for section 1
            CC2 (float): Constant current value for charge section 2
            template_filename (str): File path to pull the template schedule
                file from
            # TODO: do we want this here?
            output_filename (str): File path to save the parameterized
                schedule file to

        """
        obj = cls.from_file(template_filename)

        # TODO: give this more domain interpretability?
        obj.set_labelled_steps('CC1', 'm_szCtrlValue',
                               step_value='{0:.3f}'.format(CC1).rstrip('0'))
        obj.set_labelled_limits('CC1', 'PV_CHAN_Charge_Capacity',
                                {'compare': '>',
                                 'value': '{0:.3f}'.format(CC1_capacity).rstrip('0')}
                                )
        obj.set_labelled_steps('CC2', 'm_szCtrlValue',
                               step_value='{0:.3f}'.format(CC2).rstrip('0'))
        if output_filename is not None:
            obj.to_file(output_filename)
        return obj

    def get_labelled_steps(self, step_label):
        """
        Insert values for steps in the schedule section

        Args:
            step_label (str): The user determined step label for the step.
                If there are multiple identical labels this will operate
                on the first one it encounters

        Returns:
            (iterator): iterator for subkeys of schedule which match
                the label value
        """
        # Find all step labels
        labelled_steps = filter(
            lambda x: get(self, "Schedule.{}.m_szLabel".format(x)) == step_label,
            self['Schedule'].keys()
        )
        return labelled_steps

    def set_labelled_steps(self, step_label, step_key, step_value,
                           mode='first'):
        """
        Insert values for steps in the schedule section

        Args:
            step_label (str): The user determined step label for the step.
                If there are multiple identical labels this will operate
                on the first one it encounters
            step_key (str): Key in the step to set, e.g. ('m_szStepCtrlType')
            step_value (str): Value to set for the key
            mode (str): accepts 'first' or 'all',
                for 'first' updates only first step with matching label
                for 'all' updates all steps with matching labels

        Returns:
            dict: Altered ordered dictionary with keys corresponding to
                options or control variables.
        """
        # Find all step labels
        labelled_steps = self.get_labelled_steps(step_label)

        # TODO: should update happen in place or return new?
        for step in labelled_steps:
            set_(self, "Schedule.{}.{}".format(step, step_key), step_value)
            if mode == "first":
                break

        return self

    def step_limit_values(self, step_label, limit_var, limit_set=None):
        """
        Insert values for the limits in the steps in the schedule section

        Args:
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
        labelled_steps = self.get_labelled_steps(step_label)
        for step in labelled_steps:

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


def _get_headings(obj, delimiter='.'):
    """
    Utility function for getting all nested keys
    of a dictionary whose values are themselves
    a dict

    Args:
        obj (dict): nested dictionary to be searched
        delimiter (str): string delimiter for nested
            sub_headings, e. g. top_middle_low for
            'top', 'middle', and 'low' nested keys

    """
    headings = []
    for heading, body in obj.items():
        if isinstance(body, dict):
            headings.append(heading)
            sub_headings = _get_headings(body, delimiter=delimiter)
            headings.extend([delimiter.join([heading, sub_heading])
                             for sub_heading in sub_headings])
    return headings


def main():
    sdu = Schedule.from_fast_charge(
        1.1*3.6, 0.086, 1.1*5, '20170630-3_6C_9per_5C.sdu', 'test.sdu')


if __name__ == "__main__":
    main()
