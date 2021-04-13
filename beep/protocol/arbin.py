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
"""
Module for Arbin-compatible schedule file
parsing and parameter insertion
"""
import re
import warnings
from copy import deepcopy
from collections import OrderedDict
from beep.utils import DashOrderedDict


class Schedule(DashOrderedDict):
    """
    Schedule file utility. Provides the ability to read
    an Arbin type schedule file.  Note that __init__ works
    identically to that of an OrderedDict, i. e. with
    tuple or dictionary inputs and inherits from pydash
    getters and setters with Schedule.get, Schedule.set
    and Schedule.unset e.g.

    >>> schedule = Schedule.from_file("arbin_file_1.sdu")
    >>> schedule.set("Schedule.Step7.m_szLabel", "CC1")
    >>> print(schedule['Schedule']['Step7']['m_szLabel'])
    >>> "CC1"

    """

    @classmethod
    def from_file(cls, filename, encoding="latin-1"):
        """
        Schedule file ingestion. Converts a schedule file with section headers
        to an ordered dict with section headers as nested dicts.

        Args:
            filename (str): Schedule file name (tested with FastCharge schedule file)
            encoding (str): encoding of schedule file

        Returns:
            (Schedule): Ordered dictionary with keys corresponding to options
                or control variables. Section headers are nested dicts within
                the dict

        """
        obj = cls()
        with open(filename, "rb") as f:
            text = f.read()
            text = text.decode(encoding)

        split_text = re.split(r"\[(.+)\]", text)
        for heading, body in zip(split_text[1::2], split_text[2::2]):
            body_lines = re.split(r"[\r\n]+", body.strip())
            body_dict = OrderedDict([line.split("=", 1) for line in body_lines])
            heading = heading.replace("_", ".")
            obj.set(heading, body_dict)

        return obj

    def to_file(self, filename, encoding="latin-1", linesep="\r\n"):
        """
        Schedule file output. Converts an dictionary to a schedule file with
        the appropriate section headers. This function
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
        flat_keys = _get_headings(data, delimiter=".")
        flat_keys.reverse()  # Reverse ensures sub-dicts are removed first
        data_tuples = []
        for flat_key in flat_keys:
            data_tuple = (flat_key.replace(".", "_"), data.get_path(flat_key))
            data_tuples.append(data_tuple)
            data.unset(flat_key)
        data_tuples.reverse()

        # Construct text
        blocks = []
        for section_title, body_data in data_tuples:
            section_header = "[{}]".format(section_title)
            body = linesep.join(
                ["=".join([key, value]) for key, value in body_data.items()]
            )
            blocks.append(linesep.join([section_header, body]))
        contents = linesep.join(blocks) + linesep

        # Write file
        with open(filename, "wb") as f:
            f.write(contents.encode(encoding))

    @classmethod
    def from_fast_charge(cls, CC1, CC1_capacity, CC2, template_filename):
        """
        Function takes parameters for the FastCharge Project
        and creates the schedule files necessary to run each of
        these parameter combinations. Assumes that control type
        is CCCV.

        Args:
            CC1 (float): Constant current value for charge section 1 in Amps
            CC1_capacity (float): Capacity to charge to for section 1 in Amp-hours
            CC2 (float): Constant current value for charge section 2 in Amps
            template_filename (str): File path to pull the template schedule
                file from

        """
        obj = cls.from_file(template_filename)

        obj.set_labelled_steps(
            "CC1", "m_szCtrlValue", step_value="{0:.3f}".format(CC1).rstrip("0")
        )
        obj.set_labelled_limits(
            "CC1",
            "PV_CHAN_Charge_Capacity",
            comparator=">",
            value="{0:.3f}".format(CC1_capacity).rstrip("0"),
        )
        obj.set_labelled_steps(
            "CC2", "m_szCtrlValue", step_value="{0:.3f}".format(CC2).rstrip("0")
        )
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
            lambda x: self.get_path("Schedule.{}.m_szLabel".format(x)) == step_label,
            self["Schedule"].keys(),
        )
        return labelled_steps

    def set_labelled_steps(self, step_label, step_key, step_value, mode="first"):
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
            self.set("Schedule.{}.{}".format(step, step_key), step_value)
            if mode == "first":
                break

        return self

    def set_labelled_limits(self, step_label, limit_var, comparator, value):
        """
        Insert values for the limits in the steps in the schedule section

        Args:
            step_label (str): The user determined step label for the step. If there
                are multiple identical labels this will operate on the first one it
                encounters
            limit_var (str): Variable being used for this particular limit in the step
            value (int or str): threshold value to trip limit
            comparator (str): str-represented comparator to trip limit,
                e.g. '>' or '<'

        Returns:
            dict: Altered ordered dictionary with keys corresponding to options or control
                variables.
        """
        labelled_steps = self.get_labelled_steps(step_label)
        for step in labelled_steps:
            # Get all matching limit keys
            step_data = self.get_path("Schedule.{}".format(step))
            limits = [
                heading
                for heading in _get_headings(step_data)
                if heading.startswith("Limit")
            ]
            # Set limit of first limit step with matching code
            for limit in limits:
                limit_data = step_data[limit]
                if limit_data["m_bStepLimit"] == "1":  # Code corresponding to stop
                    if limit_data["Equation0_szLeft"] == limit_var:
                        limit_prefix = "Schedule.{}.{}".format(step, limit)
                        self.set(
                            "{}.Equation0_szCompareSign".format(limit_prefix),
                            comparator,
                        )
                        self.set("{}.Equation0_szRight".format(limit_prefix), value)
                    else:
                        warnings.warn(
                            "Additional step limit at {}.{}".format(step, limit)
                        )
        return self


def _get_headings(obj, delimiter="."):
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
            headings.extend(
                [delimiter.join([heading, sub_heading]) for sub_heading in sub_headings]
            )
    return headings
