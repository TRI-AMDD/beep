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
""" Parsing and conversion of maccor procedure files to arbin schedule files"""

import os
from beep.protocol import PROTOCOL_SCHEMA_DIR
from monty.serialization import loadfn
from collections import OrderedDict


class MaccorToBiologicMb:
    """
    Collection of methods to convert maccor protocol files to biologic modulo bat protocol files
    Differences in underlying hardware and software makes this impossible in some cases
    or forces workarounds that may fail. Certain work-arounds may be attempted and are
    documented under the conversion function.
    """

    def __init__(self):
        BIOLOGIC_SCHEMA = loadfn(
            os.path.join(PROTOCOL_SCHEMA_DIR, "biologic_mb_schema.yaml")
        )
        schema = OrderedDict(BIOLOGIC_SCHEMA)
        self.blank_seq = schema["blank_seq"]

    def _proc_step_to_seq(self, test_step, seq_num):
        """
        converts steps that are not related to control flow to sequence dicts
        (control flow steps are DO, LOOP, ADV CYCLE)
        """
        new_seq = {}
        new_seq.update(self.blank_seq)
        new_seq["Ns"] = seq_num
        new_seq["lim1_seq"] = seq_num
        new_seq["lim2_seq"] = seq_num
        new_seq["lim3_seq"] = seq_num

        # while biologic does not have >= or <= these are functionally
        # equivalent to > and < for bounds checks on floating points
        # most of the time
        operator_map = {
            ">=": ">",
            "<=": "<",
        }

        step_type = test_step["StepType"]
        assert type(step_type) == str

        step_mode = test_step["StepMode"]
        step_value = test_step["StepValue"]

        if step_type == "Rest":
            new_seq["ctrl_type"] = "Rest"
            new_seq["Apply I/C"] = "I"

            # magic number
            new_seq["N"] = "1.00"

            # should this depend on the previous step?
            # it seems like the value only matters if we were advancing cycle number
            # on charge discharge alternance. By default EC-lab seems to set this
            # to the previous steps charge/discharge
            new_seq["charge/discharge"] = "Charge"

        # Maccor intentionally misspells Discharge
        elif step_type not in ["Charge", "Dischrge"]:
            raise Exception("Unsupported Control StepType", step_type)
        elif step_mode == "Current":
            assert type(step_value) == str
            # does this need to be formatted? e.g. 1.0 from Maccor vs 1.000 for biologic
            new_seq["ctrl1_val"] = step_value

            new_seq["ctrl_type"] = "CC"
            new_seq["Apply I/C"] = "C / N"
            new_seq["ctrl1_val_unit"] = "A"
            new_seq["ctrl1_val_vs"] = "<None>"

            # magic number, unsure what this does
            new_seq["N"] = "15.00"
            new_seq["charge/discharge"] = (
                "Charge" if step_type == "Charge" else "Discharge"
            )
        elif step_mode == "Voltage":
            # does this need to be formatted? e.g. 1.0 from Maccor vs 1.000 for biologic
            assert type(step_value) == str
            new_seq["ctrl1_val"] = step_value
            new_seq["ctrl_type"] = "CV"
            new_seq["Apply I/C"] = "C / N"
            new_seq["ctrl1_val_unit"] = "V"
            new_seq["ctrl1_val_vs"] = "Ref"

            # magic number, unsure what this does
            new_seq["N"] = "15.00"
            new_seq["charge/discharge"] = (
                "Charge" if step_type == "Charge" else "Discharge"
            )
        else:
            raise Exception("Unsupported Charge/Discharge StepMode", step_mode)

        end_entries = test_step["Ends"]["EndEntry"]
        end_entries_list = (
            end_entries
            if isinstance(end_entries, list)
            else []
            if end_entries is None
            else [end_entries]
        )

        # maccor end entries are conceptually equivalent to biologic limits
        num_lims = len(end_entries_list)
        assert num_lims <= 3
        new_seq["lim_nb"] = "{0}".format(num_lims)

        for idx, end_entry in enumerate(end_entries_list):
            lim_num = idx + 1

            end_type = end_entry["EndType"]
            assert type(end_type) == str

            end_oper = end_entry["Oper"]
            assert type(end_oper) == str

            end_value = end_entry["Value"]
            assert type(end_value) == str

            if end_type == "StepTime":
                if end_oper != "=":
                    raise Exception(
                        "Unsupported StepTime operator in EndEntry", end_oper
                    )

                # Maccor time strings always contain two colons
                # at least one section must be a parseable float
                # "00:32:50" - 32 minutes and 50 seconds
                # "::.5" - 5 ms
                # "3600::" - 3600 hours
                #
                # are all possible values
                # the smallest possible value is
                # "::.01" - 10ms
                # longest value unknown
                hour_str, min_str, sec_str = end_value.split(":")
                hours = 0 if hour_str == "" else float(hour_str)
                mins = 0 if min_str == "" else float(min_str)
                secs = 0 if sec_str == "" else float(sec_str)

                ms = int(hours * 60 * 60 * 1000 + mins * 60 * 1000 + secs * 1000)

                new_seq["lim{0}_type".format(lim_num)] = "Time"
                new_seq["lim{0}_value_unit".format(lim_num)] = "ms"
                new_seq["lim{0}_value".format(lim_num)] = "{0}".format(ms)
                # even though maccor claims it checks for time equal to some threshold
                # it's actually looking for time greater than or equal to that threshold
                # biologic has no >=  so we use >
                new_seq["lim{0}_comp".format(lim_num)] = ">"
            elif end_type == "Voltage":
                if operator_map[end_oper] is None:
                    raise Exception(
                        "Unsupported Voltage operator in EndEntry", end_oper
                    )

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "Voltage"
                new_seq["lim{0}_value".format(lim_num)] = end_value
                new_seq["lim{0}_value_unit".format(lim_num)] = "V"
            elif end_type == "Current":
                if operator_map[end_oper] is None:
                    raise Exception(
                        "Unsupported Voltage operator in EndEntry", end_oper
                    )

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "Current"
                new_seq["lim{0}_value".format(lim_num)] = end_value
                new_seq["lim{0}_value_unit".format(lim_num)] = "A"
            else:
                raise Exception("Unsupported EndType", end_type)

        report_entries = test_step["Reports"]["ReportEntry"]
        report_entries_list = (
            report_entries
            if isinstance(report_entries, list)
            else []
            if report_entries is None
            else [report_entries]
        )

        num_reports = len(report_entries_list)
        assert num_reports <= 3
        new_seq["rec_nb"] = "{0}".format(num_reports)

        for idx, report in enumerate(report_entries_list):
            rec_num = idx + 1

            report_type = report["ReportType"]
            assert type(report_type) == str

            report_value = report["Value"]
            assert type(report_value) == str

            if report_type == "StepTime":
                hour_str, min_str, sec_str = report_value.split(":")
                hours = 0 if hour_str == "" else float(hour_str)
                mins = 0 if min_str == "" else float(min_str)
                secs = 0 if sec_str == "" else float(sec_str)

                ms = int(hours * 60 * 60 * 1000 + mins * 60 * 1000 + secs * 1000)

                new_seq["rec{0}_type".format(rec_num)] = "Time"
                new_seq["rec{0}_value".format(rec_num)] = "{}".format(ms)
                new_seq["rec{0}_value_unit".format(rec_num)] = "ms"
            elif report_type == "Voltage":
                new_seq["rec{0}_type".format(rec_num)] = "Voltage"
                new_seq["rec{0}_value".format(rec_num)] = report_value
                new_seq["rec{0}_value_unit".format(rec_num)] = "V"
            elif report_type == "Current":
                new_seq["rec{0}_type".format(rec_num)] = "Current"
                new_seq["rec{0}_value".format(rec_num)] = report_value
                new_seq["rec{0}_value_unit".format(rec_num)] = "A"
            else:
                raise Exception("Unsupported ReportType", report_type)

        return new_seq
