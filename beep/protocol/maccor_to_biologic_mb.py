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
import re
import copy
import xmltodict
from beep.protocol import (
    PROTOCOL_SCHEMA_DIR,
    BIOLOGIC_TEMPLATE_DIR,
    PROCEDURE_TEMPLATE_DIR,
)
from monty.serialization import loadfn
from collections import OrderedDict
from pydash import get, set_, find_index, clone_deep_with, clone_deep
import pandas as pd
import json

# magic number for biologic
END_SEQ_NUM = 9999


class MaccorToBiologicMb:
    """
    Collection of methods to convert maccor protocol files to biologic modulo bat protocol files
    Differences in underlying hardware and software makes this impossible in some cases
    or forces workarounds that may fail. Certain work-arounds may be attempted and are
    documented under the conversion function.
    """

    def __init__(self):
        BIOLOGIC_SCHEMA = loadfn(os.path.join(PROTOCOL_SCHEMA_DIR, "biologic_mb_schema.yaml"))
        schema = OrderedDict(BIOLOGIC_SCHEMA)
        self._blank_seq = OrderedDict(schema["blank_seq"])
        self._blank_end_entry = xmltodict.parse(
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<EndEntry>"
            "    <EndType></EndType>"
            "    <SpecialType></SpecialType>"
            "    <Oper></Oper>"
            "    <Step></Step>"
            "    <Value></Value>"
            "</EndEntry>",
            strip_whitespace=True,
        )
        self.step_mappers = []
        self.seq_mappers = []
        self.max_voltage_v = None
        self.min_voltage_v = None
        self.max_current_a = None
        self.min_current_a = None
        self._mps_header_template = (
            "BT-LAB SETTING FILE\r\n"
            "\r\n"
            "Number of linked techniques : {}\r\n"
            "\r\n"
            "Filename : C:\\Users\\Biologic Server\\Documents\\BT-Lab\\Data\\PK_loop_technique2.mps\r\n"
            "\r\n"
            "Device : BCS-815\r\n"
            "Ecell ctrl range : min = 0.00 V, max = 9.00 V\r\n"
            "Safety Limits :\r\n"
            "	Ecell min = 2.90 V\r\n"
            "	Ecell max = 4.3 V\r\n"
            "	for t > 100 ms\r\n"
            "Electrode material : \r\n"
            "Initial state : \r\n"
            "Electrolyte : \r\n"
            "Comments : \r\n"
            "Mass of active material : 0.001 mg\r\n"
            " at x = 0.000\r\n"
            "Molecular weight of active material (at x = 0) : 0.001 g/mol\r\n"
            "Atomic weight of intercalated ion : 0.001 g/mol\r\n"
            "Acquisition started at : xo = 0.000\r\n"
            "Number of e- transfered per intercalated ion : 1\r\n"
            "for DX = 1, DQ = 26.802 mA.h\r\n"
            "Battery capacity : 0.000 A.h\r\n"
            "Electrode surface area : 0.001 cm\N{superscript two}\r\n"
            "Characteristic mass : 0.001 g\r\n"
            "Text export\r\n"
            "   Mode : Standard\r\n"
            "   Time format : Absolute MMDDYYYY\r\n"
            "Cycle Definition : Charge/Discharge alternance\r\n"
            "Turn to OCV between techniques\r\n"
        )

    def _get_decimal_sig_figs(self, val_str):
        match_p10 = re.search("(e|E)([-+]?[0-9]+)", val_str)
        p10 = 0 if match_p10 is None else int(match_p10.groups()[1])

        match_sig_figs = re.search("\\.([0-9]*[1-9])", val_str)
        explicit_sig_figs = (
            0 if match_sig_figs is None else len(match_sig_figs.groups(1)[0])
        )

        return explicit_sig_figs - p10

    def _convert_volts(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)
        if num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mV"
        else:
            return "{:.3f}".format(num), "V"

    def _convert_amps(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)

        if num < 1e-9 or decimal_sig_figs > 12:
            return "{:.3f}".format(num * 1e12), "pA"
        if num < 1e-6 or decimal_sig_figs > 9:
            return "{:.3f}".format(num * 1e9), "nA"
        elif num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}A"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mA"
        else:
            return "{:.3f}".format(num), "A"

    def _convert_watts(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)

        if num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}W"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mW"
        else:
            return "{:.3f}".format(num), "W"

    def _convert_ohms(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)

        if num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}Ohms"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mOhms"
        elif num < 1e3 or decimal_sig_figs > 0:
            return "{:.3f}".format(num), "Ohms"
        elif num < 1e6 or decimal_sig_figs > -3:
            return "{:.3f}".format(num * 1e-3), "kOhms"
        else:
            return "{:.3f}".format(num * 1e-6), "MOhms"

    def _convert_time(self, time_str):
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
        hour_str, min_str, sec_str = time_str.split(":")
        hours = 0.0 if hour_str == "" else float(hour_str)
        mins = 0.0 if min_str == "" else float(min_str)
        secs = 0.0 if sec_str == "" else float(sec_str)

        if mins == 0.0 and secs == 0.0:
            return "{:.3f}".format(int(hours)), "h"

        if secs == 0.0:
            total_time_min = int(hours * 60 + mins)
            return "{:.3f}".format(total_time_min), "mn"

        if secs % 1.0 == 0.0:
            total_time_sec = int(hours * 60 * 60 + mins * 60 + secs)
            return "{:.3f}".format(total_time_sec), "s"

        total_time_ms = int(hours * 60 * 60 * 1000 + mins * 60 * 1000 + secs * 1000)

        biologic_max_time_num = 1e10  # taken from biologic ec-lab software
        if total_time_ms < biologic_max_time_num:
            return "{:.3f}".format(total_time_ms), "ms"
        else:
            # max hours in maccor is 3600, may exceed representable time in ms but not seconds
            total_time_sec = total_time_ms / 1000
            print(
                (
                    "Warning: lost precision converting time {} to {} {}, "
                    "Biologic does not have the precision to represent this number"
                ).format(time_str, total_time_sec, "s")
            )
            return "{:.3f}".format(total_time_sec), "s"

   """
    converts a maccor step, or the parts of a maccor step generated by splitting it
    into parts in _split_step into 1 or more seqs.

    accepts:
        - step_parts list(OrderedDict), len > 0
        - step_num: int,
        - seq_nums_by_step_num: dict(int, list(int))
        - goto_lowerbound: int
        - goto_upperbound: int
        - end_step_num: int

    return:
        - list(OrderedDict)
    """

    def _convert_step_parts(
        self,
        step_parts,
        step_num,
        seq_nums_by_step_num,
        goto_lowerbound,
        goto_upperbound,
        end_step_num,
    ):
        assert step_num in seq_nums_by_step_num
        assert len(step_parts) == len(seq_nums_by_step_num[step_num])
        seqs = []

        for i, step_part in enumerate(step_parts):
            seq_num = seq_nums_by_step_num[step_num][i]

        new_seq = self._blank_seq.copy()
        new_seq["Ns"] = seq_num
        new_seq["lim1_seq"] = seq_num + 1
        new_seq["lim2_seq"] = seq_num + 1
        new_seq["lim3_seq"] = seq_num + 1

        # while biologic does not have >= or <= these are functionally
        # equivalent to > and < for bounds checks on floating points
        # most of the time
        operator_map = {
            ">=": ">",
            "<=": "<",
        }

            step_type = step_part["StepType"]
        assert type(step_type) == str

            step_mode = step_part["StepMode"]
            step_value = step_part["StepValue"]

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
            ctrl1_val, ctrl1_val_unit = self._convert_amps(step_value)
            new_seq["ctrl1_val"] = ctrl1_val
            new_seq["ctrl1_val_unit"] = ctrl1_val_unit

            new_seq["ctrl_type"] = "CC"
            new_seq["Apply I/C"] = "I"
            new_seq["ctrl1_val_vs"] = "<None>"

            # magic number, unsure what this does
            new_seq["N"] = "15.00"

                new_seq["charge/discharge"] = "Charge" if step_type == "Charge" else "Discharge"

        elif step_mode == "Voltage":
            # does this need to be formatted? e.g. 1.0 from Maccor vs 1.000 for biologic
            assert type(step_value) == str

            ctrl1_val, ctrl1_val_unit = self._convert_volts(step_value)
            new_seq["ctrl1_val"] = ctrl1_val
            new_seq["ctrl1_val_unit"] = ctrl1_val_unit

            new_seq["ctrl_type"] = "CV"
            new_seq["Apply I/C"] = "I"
            new_seq["ctrl1_val_vs"] = "Ref"

            # magic number, unsure what this does
            new_seq["N"] = "15.00"
                new_seq["charge/discharge"] = "Charge" if step_type == "Charge" else "Discharge"

        else:
            raise Exception("Unsupported Charge/Discharge StepMode", step_mode)

            end_entries = get(step_part, "Ends.EndEntry")
        end_entries_list = (
                end_entries if isinstance(end_entries, list) else [] if end_entries is None else [end_entries]
        )

        # maccor end entries are conceptually equivalent to biologic limits
        num_end_entries = len(end_entries_list)
        if num_end_entries > 3:
            raise Exception(
                (
                    "Step {} has more than 3 EndEntries, the max allowed"
                    " by Biologic. Either remove some limits from the source"
                    " loaded diagnostic file or filter by number using the"
                    " remove_end_entries_by_pred method"
                ).format(step_num)
            )

        # number of limits for biologic to use
        new_seq["lim_nb"] = num_end_entries

        for idx, end_entry in enumerate(end_entries_list):
            lim_num = idx + 1

            end_type = end_entry["EndType"]
            assert isinstance(end_type, str)

            end_oper = end_entry["Oper"]
            assert isinstance(end_oper, str)

            end_value = end_entry["Value"]
            assert isinstance(end_value, str)

            goto_step_num_str = end_entry["Step"]
            assert isinstance(goto_step_num_str, str)
            goto_step_num = int(goto_step_num_str)

                if goto_step_num < goto_lowerbound or goto_step_num > goto_upperbound and goto_step_num != end_step_num:
                raise Exception(
                    "GOTO in step "
                    + str(step_num)
                        + " to location that could break loop.\nGOTO Lowerbound in technique: "
                        + str(goto_lowerbound)
                    + "\nGOTO target step num: "
                    + str(goto_step_num)
                        + "\nGOTO upperbound in technique: "
                        + str(goto_upperbound)
                        + "\nGOTO End:"
                    + str(end_step_num)
                )

                goto_seq = -1
                if goto_step_num == step_num + 1:
                goto_seq = seq_num + 1
                elif goto_seq == end_step_num:
                    goto_seq = END_SEQ_NUM
            else:
                    goto_seq = seq_nums_by_step_num[goto_step_num][0]

            new_seq["lim{}_seq".format(lim_num)] = goto_seq

            if goto_step_num != step_num + 1:
                new_seq["lim{}_action".format(lim_num)] = "Goto sequence"

            if end_type == "StepTime":
                if end_oper != "=":
                        raise Exception("Unsupported StepTime operator in EndEntry", end_oper)

                lim_value, lim_value_unit = self._convert_time(end_value)

                new_seq["lim{0}_type".format(lim_num)] = "Time"
                new_seq["lim{0}_value_unit".format(lim_num)] = lim_value_unit
                new_seq["lim{0}_value".format(lim_num)] = lim_value
                # even though maccor claims it checks for time equal to some threshold
                # it's actually looking for time greater than or equal to that threshold
                # biologic has no >=  so we use >1
                new_seq["lim{0}_comp".format(lim_num)] = ">"
            elif end_type == "Voltage":
                if operator_map[end_oper] is None:
                        raise Exception("Unsupported Voltage operator in EndEntry", end_oper)

                lim_value, lim_value_unit = self._convert_volts(end_value)

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "Ecell"
                new_seq["lim{0}_value".format(lim_num)] = lim_value
                new_seq["lim{0}_value_unit".format(lim_num)] = lim_value_unit
            elif end_type == "Current":
                if operator_map[end_oper] is None:
                        raise Exception("Unsupported Voltage operator in EndEntry", end_oper)

                lim_value, lim_value_unit = self._convert_amps(end_value)

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "|I|"
                new_seq["lim{0}_value".format(lim_num)] = lim_value
                new_seq["lim{0}_value_unit".format(lim_num)] = lim_value_unit
            else:
                raise Exception("Unsupported EndType", end_type)

            report_entries = get(step_part, "Reports.ReportEntry")
        report_entries_list = (
            report_entries
            if isinstance(report_entries, list)
            else []
            if report_entries is None
            else [report_entries]
        )

        num_reports = len(report_entries_list)
        if num_reports > 3:
            raise Exception(
                (
                    "Step {} has more than 3 ReportyEntries, the max allowed"
                    " by Biologic. Either remove them from the source file"
                    " or pre-process the loaded maccor_ast"
                ).format(step_num)
            )

        new_seq["rec_nb"] = num_reports

        for idx, report in enumerate(report_entries_list):
            rec_num = idx + 1

            report_type = report["ReportType"]
            assert type(report_type) == str

            report_value = report["Value"]
            assert type(report_value) == str

            if report_type == "StepTime":
                rec_value, rec_value_unit = self._convert_time(report_value)

                new_seq["rec{0}_type".format(rec_num)] = "Time"
                new_seq["rec{0}_value".format(rec_num)] = rec_value
                new_seq["rec{0}_value_unit".format(rec_num)] = rec_value_unit
            elif report_type == "Voltage":
                rec_value, rec_value_unit = self._convert_volts(report_value)

                new_seq["rec{0}_type".format(rec_num)] = "Ecell"
                new_seq["rec{0}_value".format(rec_num)] = rec_value
                new_seq["rec{0}_value_unit".format(rec_num)] = rec_value_unit
            elif report_type == "Current":
                rec_value, rec_value_unit = self._convert_amps(report_value)

                new_seq["rec{0}_type".format(rec_num)] = "I"
                new_seq["rec{0}_value".format(rec_num)] = rec_value
                new_seq["rec{0}_value_unit".format(rec_num)] = rec_value_unit
            else:
                raise Exception("Unsupported ReportType", report_type)

            seqs.append(new_seq)
        return seqs


    def _create_loop_seq(self, seq_num, seq_num_to_loop_to, num_loops):
        loop_seq = self._blank_seq.copy()
        loop_seq["Ns"] = seq_num
        loop_seq["ctrl_type"] = "Loop"
        loop_seq["ctrl_repeat"] = num_loops
        loop_seq["ctrl_seq"] = seq_num_to_loop_to
        # automatically added to loops, semantically useless
        loop_seq["lim1_seq"] = seq_num + 1
        loop_seq["lim2_seq"] = seq_num + 1
        loop_seq["lim3_seq"] = seq_num + 1
        loop_seq["Apply I/C"] = "I"
        loop_seq["ctrl1_val"] = "100.000"

        return loop_seq

    """
    returns the AST for a Maccor diagnostic file
    """

    def load_maccor_ast(self, maccorFilePath, encoding="UTF-8"):
        with open(maccorFilePath, "rb") as f:
            text = f.read().decode(encoding)

        return xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)

    def _seqs_to_str(self, seqs, col_width=20):
        seq_str = ""
        for key in OrderedDict.keys(self._blank_seq):
            if len(key) > col_width:
                raise Exception(
                    "seq key {} has length greater than col width {}".format(
                        key, col_width
                    )
                )

            field_row = key.ljust(col_width, " ")
            for seq_num, seq in enumerate(seqs):
                if key not in seq:
                    raise Exception(
                        "Could not find field {} in seq {}".format(key, seq_num)
                    )

                if len(str(seq[key])) > col_width:
                    raise Exception(
                        "{} in seq {} is greater than column width".format(
                            seq[key], seq_num
                        )
                    )
                field_row += str(seq[key]).ljust(col_width, " ")
            seq_str += field_row + "\r\n"

        return seq_str

    """
    partitions steps into multiple techniques

    note: for techniques that loop, the substeps will not include Do 1/Loop 1
    steps at each respective end. The offset is adjusted accordingly.

    accepts
        - a list of Maccor TestSteps parsed from the source XML

    returns
        - list(TechniquePartition)
    """

    def _partition_steps_into_techniques(self, steps):
        # any nested loops must be brokeN into two techniques,
        # the actual procedure, and an outer looping technique

        # locate partition points and mark technique
        tech_loops_by_start_step_num = {}
        curr_loop_1_start = -1
        curr_tech_start = 0
        in_loop_tech = False
        tech_ranges = []
        for i, step in enumerate(steps):
            step_type = get(step, "StepType")
            if step_type == "Do 1":
                curr_loop_1_start = i
            elif step_type == "Do 2":
                assert curr_loop_1_start != -1
                in_loop_tech = True
            elif step_type == "Loop 1" and in_loop_tech:
                # gather steps preceding start of looped tech
                if curr_tech_start != curr_loop_1_start:
                    tech_ranges.append((curr_tech_start, curr_loop_1_start))

                # gather data around looped tech
                curr_tech_start = i + 1
                tech_ranges.append((curr_loop_1_start, curr_tech_start))
                num_loops = int(get(step, "Ends.EndEntry.Value"))
                tech_loops_by_start_step_num[curr_loop_1_start] = num_loops

                in_loop_tech = False
                curr_loop_1_start = -1

        if curr_tech_start < len(steps):
            tech_ranges.append((curr_tech_start, len(steps)))

        # ranges must contiguously cover the steps with no overlap
        first_tech_start = tech_ranges[0][0]
        assert first_tech_start == 0
        trailing_edge = tech_ranges[-1][1]
        assert trailing_edge == len(steps)

        prev_end = 0
        technique_num = 1
        tech_partitions = []
        for start, end in tech_ranges:
            # ranges must contiguously cover the steps with no overlap
            assert prev_end == start
            assert end > start
            prev_end = end

            tech_does_loop = start in tech_loops_by_start_step_num
            # if tech loops, elide outer loop construct
            substeps = steps[start + 1 : end - 1] if tech_does_loop else steps[start:end]
            num_loops = tech_loops_by_start_step_num[start] if tech_does_loop else 0
            step_num_offset = start + 1 if tech_does_loop else start

            partition = TechniquePartition(
                technique_num=technique_num,
                steps=substeps,
                step_num_offset=step_num_offset,
                tech_does_loop=tech_does_loop,
                num_loops=num_loops,
            )
            tech_partitions.append(partition)
            if tech_does_loop:
                technique_num += 2
            else:
                technique_num += 1

        return tech_partitions

    """
    Utility for creating step mappers that filter out one or more unwanted EndEntry

    predicate should return True to keep an EndEntry, False to filter it

    accepts
        - pred: (end_entry: OrderedDict, idx: int, step_num: int) -> Boolean

    return
        - mapper: (step: OrderedDict, step_num: int) -> mapped_step: OrderedDict
    """

    def _create_step_end_entry_filter(self, pred):
        def mapper(step, step_num):
            end_entries = get(step, "Ends.EndEntry")
            if end_entries is None:
                return step

            end_entries = end_entries if type(end_entries) is not list else end_entries
            filtered_end_entries = []
            for idx, end_entry in enumerate(end_entries):
                if pred(end_entry, idx, step_num):
                    filtered_end_entries.append(end_entry)

            if len(filtered_end_entries) == len(end_entries):
                # noop
                return step
            else:
                filtered_end_entries = filtered_end_entries if len(filtered_end_entries) > 0 else None
                step_copy = clone_deep(step)
                step_copy = set_(step_copy, "Ends.EndEntry", filtered_end_entries)
                assert step_copy != step
                return step_copy

        return mapper

    def _filter_end_entry_by_max_voltage(self, step, step_num):
        def pred(end_entry, _idx, _step_num):
            if self.max_voltage_v is None:
                return True

            end_type = get(end_entry, "EndType")
            if end_type != "Voltage":
                return True

            val = float(get(end_entry, "Value"))
            oper = get(end_entry, "Oper")
            if oper == ">=" and val >= self.max_voltage_v:
                return False
            else:
                return True

        return self._create_step_end_entry_filter(pred)(step, step_num)

    def _filter_end_entry_by_min_voltage(self, step, step_num):
        def pred(end_entry, _idx, _step_num):
            if self.min_voltage_v is None:
                return True

            end_type = get(end_entry, "EndType")
            if end_type != "Voltage":
                return True

            val = float(get(end_entry, "Value"))
            oper = get(end_entry, "Oper")
            if oper == "<=" and val <= self.min_voltage_v:
                return False
            else:
                return True

        return self._create_step_end_entry_filter(pred)(step, step_num)

    def _filter_end_entry_by_max_current(self, step, step_num):
        def pred(end_entry, _idx, _step_num):
            if self.max_current_a is None:
                return True

            end_type = get(end_entry, "EndType")
            if end_type != "Current":
                return True

            val = float(get(end_entry, "Value"))
            oper = get(end_entry, "Oper")
            if end_type == "Current" and oper == ">=" and val > self.max_current_a:
                return False
            else:
                return True

        return self._create_step_end_entry_filter(pred)(step, step_num)

    def _filter_end_entry_by_min_current(self, step, step_num):
        def pred(end_entry, _idx, _step_num):
            if self.min_current_a is None:
                return True

            end_type = get(end_entry, "EndType")
            if end_type != "Current":
                return True

            val = float(get(end_entry, "Value"))
            oper = get(end_entry, "Oper")
            if end_type == "Current" and oper == "<=" and val < self.min_current_a:
                return False
            else:
                return True

        return self._create_step_end_entry_filter(pred)(step, step_num)

    def _apply_step_mappings(self, steps, extra_mappers=[]):
        mapped_steps = []

        for i, step in enumerate(steps):
            step_num = i + 1

            mapped_step = step
            mapped_step = self._filter_end_entry_by_max_current(mapped_step, step_num)
            mapped_step = self._filter_end_entry_by_min_current(mapped_step, step_num)
            mapped_step = self._filter_end_entry_by_max_voltage(mapped_step, step_num)
            mapped_step = self._filter_end_entry_by_min_voltage(mapped_step, step_num)

            for mapper in self.step_mappers + extra_mappers:
                mapped_step = mapper(mapped_step, step_num)

            mapped_steps.append(mapped_step)

        return mapped_steps

    def _apply_max_current_to_seq(self, technique_num, seq, i):
        if self.max_current_a is None:
            return seq

        mapped_seq = copy.deepcopy(seq)
        mapped_seq["I Range Max"] = "{:.3f}".format(self.max_current_a)
        return mapped_seq

    def _apply_min_current_to_seq(self, technique_num, seq, i):
        if self.min_current_a is None:
            return seq

        mapped_seq = copy.deepcopy(seq)
        mapped_seq["I Range Min"] = "{:.3f}".format(self.min_current_a)
        return mapped_seq

    def _apply_max_voltage_to_seq(self, technique_num, seq, i):
        if self.max_voltage_v is None:
            return seq

        mapped_seq = copy.deepcopy(seq)
        mapped_seq["E Range Max (V)"] = "{:.3f}".format(self.max_voltage_v)
        return mapped_seq

    def _apply_min_voltage_to_seq(self, technique_num, seq, i):
        if self.min_voltage_v is None:
            return seq

        mapped_seq = copy.deepcopy(seq)
        mapped_seq["E Range Min (V)"] = "{:.3f}".format(self.min_voltage_v)
        return mapped_seq

    """
    maps seqs based on global parameters such as min_voltage_v, applies
    user defined mappings, and any additional mappers passed to the function.

    maps have the signature (technique_num, seq, index_of_seq)

    accepts
        - technique_num: int
        - seqs: list(OrderedDict)
        - mappers: optional list((int, OrderedDict, int) -> OrderedDict))

    returns 
        - list(OrderedDict)
    """

    def _apply_seq_mappings(self, technique_num, seqs, extra_mappers=[]):
        mapped_seqs = []
        for i, seq in enumerate(seqs):
            mapped_seq = seq

            mapped_seq = self._apply_max_current_to_seq(technique_num, seq, i)
            mapped_seq = self._apply_min_current_to_seq(technique_num, seq, i)
            mapped_seq = self._apply_max_voltage_to_seq(technique_num, seq, i)
            mapped_seq = self._apply_min_voltage_to_seq(technique_num, seq, i)

            for mapper in self.seq_mappers + extra_mappers:
                mapped_seq = mapper(technique_nume, seq, i)

            mapped_seqs.append(mapped_seq)

        return mapped_seqs

    

        # based on sample biologic mps file

        # ordering from blank_seq template is _vital_ for this to work
        file_str = (
            "BT-LAB SETTING FILE\r\n"
            "\r\n"
            "Number of linked techniques : 1\r\n"
            "\r\n"
            "Filename : C:\\Users\\User\\Documents\\BT-Lab\\Data\\Grace\\BASF\\BCS - 171.64.160.115_Ja9_cOver70_CE3.mps\r\n\r\n"  # noqa
            "Device : BCS-805\r\n"
            "Ecell ctrl range : min = 0.00 V, max = 10.00 V\r\n"
            "Electrode material : \r\n"
            "Initial state : \r\n"
            "Electrolyte : \r\n"
            "Comments : \r\n"
            "Mass of active material : 0.001 mg\r\n"
            " at x = 0.000\r\n"  # leading space intentional
            "Molecular weight of active material (at x = 0) : 0.001 g/mol\r\n"
            "Atomic weight of intercalated ion : 0.001 g/mol\r\n"
            "Acquisition started at : xo = 0.000\r\n"
            "Number of e- transfered per intercalated ion : 1\r\n"
            "for DX = 1, DQ = 26.802 mA.h\r\n"
            "Battery capacity : 1.000 A.h\r\n"
            "Electrode surface area : 0.001 cm\N{superscript two}\r\n"
            "Characteristic mass : 8.624 mg\r\n"
            "Cycle Definition : Charge/Discharge alternance\r\n"
            "Do not turn to OCV between techniques\r\n"
            "\r\n"
            "Technique : 1\r\n"
            "Modulo Bat\r\n"
        )

        file_str += self._seqs_to_str(seqs, col_width)
        return file_str

    """
    converts maccor AST to biologic protocol
    resulting string assumes generated file will have
    LATIN-1 i.e. ISO-8859-1 encoding
    """

    def maccor_ast_to_protocol_str(self, maccor_ast, unroll=False, col_width=20):
        seqs = self.maccor_ast_to_biologic_seqs(maccor_ast, unroll=unroll)
        return self.biologic_seqs_to_protocol_str(seqs, col_width)

    """
    converted loaded biologic seqs to a protocol file
    """

    def biologic_seqs_to_protocol_file(self, seqs, fp, col_width=20):
        file_str = self.biologic_seqs_to_protocol_str(seqs, col_width)
        with open(fp, "wb") as f:
            f.write(file_str.encode("ISO-8859-1"))


    # REWRITE TIME - what are we doing?
    # 
    # Before we were doing loop unrolling, which ended up being wrong.
    # There's weird counting logic because GOTOs were gonna be real hard
    # Steps get split, we didn't account for this originally
    # 
    # what we need now:
    # split along technique lines
    # ensure all gotos are valid
    # 
    # counting logic is complex
    # set field processing mappings
    # 
    # 1. parse AST
    # 2. get steps
    # 3. apply filter rules, map rules
    # 4. split along technique lines
    # 5. convert physical operations
    # 6. convert gotos, assert  range invariants
    # 7. merge step/seq mappings
    # 8. apply mappings
    # 9. create transition rules 
    # 10. write all files 


class CycleAdvancementRules:
    def __init__(
        self,
        tech_num,
        tech_does_loop,
        adv_cycle_on_start,
        adv_cycle_on_tech_loop,
        adv_cycle_seq_transitions,
        debug_adv_cycle_on_step_transitions={},
    ):
        self.tech_num = tech_num
        self.tech_does_loop = tech_does_loop
        self.adv_cycle_on_start = adv_cycle_on_start
        self.adv_cycle_on_tech_loop = adv_cycle_on_tech_loop
        self.adv_cycle_seq_transitions = adv_cycle_seq_transitions
        self.debug_adv_cycle_on_step_transitions = debug_adv_cycle_on_step_transitions

    def __repr__(self):
        return (
            "{\n"
            + "  tech_num: {},\n".format(self.tech_num)
            + "  tech_does_loop: {},\n".format(self.tech_does_loop)
            + "  adv_cycle_on_start: {},\n".format(self.adv_cycle_on_start)
            + "  adv_cycle_on_tech_loop: {},\n".format(self.adv_cycle_on_tech_loop)
            + "  adv_cycle_seq_transitions: {},\n".format(self.adv_cycle_seq_transitions)
            + "  debug_adv_cycle_on_step_transitions: {},\n".format(self.debug_adv_cycle_on_step_transitions)
            + "}\n"
        )


class CycleAdvancementRulesSerializer:
    """
    accepts
        - cycle_advancement_rules: list(CycleAdvancementRules)
        - indent: option(int)
    """

    def json(self, cycle_advancement_rules, indent=2):
        parseable_adv_cycle_seq_transitions = []
        for (
            s,
            t,
        ), adv_cycle_count in cycle_advancement_rules.adv_cycle_seq_transitions.items():
            parseable_adv_cycle_seq_transitions.append(
                {
                    "source": s,
                    "target": t,
                    "adv_cycle_count": adv_cycle_count,
                }
            )

        parseable_debug_adv_cycle_on_step_transitions = []
        for (
            s,
            t,
        ), adv_cycle_count in cycle_advancement_rules.debug_adv_cycle_on_step_transitions.items():
            parseable_debug_adv_cycle_on_step_transitions.append(
                {
                    "source": s,
                    "target": t,
                    "adv_cycle_count": adv_cycle_count,
                }
            )

        obj = {
            "tech_num": cycle_advancement_rules.tech_num,
            "tech_does_loop": cycle_advancement_rules.tech_does_loop,
            "adv_cycle_on_start": cycle_advancement_rules.adv_cycle_on_start,
            "adv_cycle_on_tech_loop": cycle_advancement_rules.adv_cycle_on_tech_loop,
            # these are (int, int) -> int maps, tuples cannot be represented in json
            "adv_cycle_seq_transitions": parseable_adv_cycle_seq_transitions,
            "debug_adv_cycle_on_step_transitions": parseable_debug_adv_cycle_on_step_transitions,
        }

        return json.dumps(obj, indent=indent)

    def parse_json(self, serialized):
        data = json.loads(serialized)

        tech_num = data["tech_num"]
        tech_does_loop = data["tech_does_loop"]
        adv_cycle_on_start = data["adv_cycle_on_start"]
        adv_cycle_on_tech_loop = data["adv_cycle_on_tech_loop"]

        adv_cycle_seq_transitions = {}
        for d in data["adv_cycle_seq_transitions"]:
            transition = (d["source"], d["target"])
            adv_cycle_seq_transitions[transition] = d["adv_cycle_count"]

        debug_adv_cycle_on_step_transitions = {}
        for d in data["debug_adv_cycle_on_step_transitions"]:
            transition = (d["source"], d["target"])
            debug_adv_cycle_on_step_transitions[transition] = d["adv_cycle_count"]

        return CycleAdvancementRules(
            tech_num,
            tech_does_loop,
            adv_cycle_on_start,
            adv_cycle_on_tech_loop,
            adv_cycle_seq_transitions,
            debug_adv_cycle_on_step_transitions,
        )



"""
Processes CSV files generated from several biologic techniques
and creates a new set of CSVs with an additional "cycle_index" column.

accepts
  - technique_csv_file_paths: list of file paths to Biologic CSVs
  - technique_serialized_transition_rules_file_paths: list of file paths to serialized CycleAdvancementRules
  - technique_csv_out_file_paths: list of filepaths to write new data to

side-effects
   - writes a new CSV file for every entry in csv_and_transition_rules_file_paths

invariants
    - all arguments must be of the same length
    - the i-th entry form a logical tuple
    - technique files appear in the order in which they were created
      e.g. technique 1, then technique 2 etc.

example call:
add_cycle_nums_to_csvs(
    [
        os.path.join(MY_DIR, "protocol1_2a_technique_1.csv"),
        os.path.join(MY_DIR, "protocol1_2a_technique_2.csv"),
    ]
    [
        os.path.join(MY_DIR, "protocol1_technique_1_transiton_rules.json"),
        os.path.join(MY_DIR, "protocol1_technique_2_transiton_rules.json"),
    ]
    [
        os.path.join(MY_DIR, "protocol1_2a_technique_1_processed.csv"),
        os.path.join(MY_DIR, "protocol1_2a_technique_2_processed.csv"),
    ]
)
"""


def add_cycle_nums_to_csvs(
    technique_csv_file_paths,
    technique_serialized_transition_rules_file_paths,
    technique_csv_out_file_paths,
):
    assert len(technique_csv_file_paths) == len(technique_csv_out_file_paths)
    assert len(technique_csv_file_paths) == len(technique_serialized_transition_rules_file_paths)

    technique_conversion_filepaths = zip(
        technique_csv_file_paths,
        technique_serialized_transition_rules_file_paths,
        technique_csv_out_file_paths,
    )

    serializer = CycleAdvancementRulesSerializer()
    cycle_num = 1
    for csv_fp, serialized_transtion_fp, csv_out_fp in technique_conversion_filepaths:
        with open(serialized_transtion_fp, "r") as f:
            data = f.read()
            cycle_advancement_rules = serializer.parse_json(data)

        df = pd.read_csv(csv_fp, sep=";")

        cycle_num += cycle_advancement_rules.adv_cycle_on_start

        prev_seq_num = int(df.iloc[0]["Ns"])
        prev_loop_num = int(df.iloc[0]["Loop"])
        cycle_nums = []
        for _, row in df.iterrows():
            seq_num = int(row["Ns"])
            loop_num = int(row["Loop"])

            # a transition may occur because of a loop technique or a loop seq,
            # it is possible to double count cycle advances if we don't handle them separately
            if loop_num != prev_loop_num:
                cycle_num += cycle_advancement_rules.adv_cycle_on_tech_loop

            elif seq_num != prev_seq_num:
                transition = (prev_seq_num, seq_num)
                cycle_num += cycle_advancement_rules.adv_cycle_seq_transitions.get(transition, 0)

            prev_loop_num = loop_num
            prev_seq_num = seq_num

            cycle_nums.append(cycle_num)

        df["cycle_index"] = cycle_nums
        df.to_csv(csv_out_fp, sep=";")



