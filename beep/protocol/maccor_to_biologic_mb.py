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
from collections import OrderedDict, deque
from pydash import get, unset, set_, find_index, clone_deep_with
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
        BIOLOGIC_SCHEMA = loadfn(
            os.path.join(PROTOCOL_SCHEMA_DIR, "biologic_mb_schema.yaml")
        )
        schema = OrderedDict(BIOLOGIC_SCHEMA)
        self.blank_seq = OrderedDict(schema["blank_seq"])

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

    def _proc_step_to_seq(
        self,
        proc_step,
        step_num,
        seq_from_step_num,
        goto_lower_bound,
        end_step_num,
        current_range="10 A",
        split_step=None,
    ):
        """
        converts steps that are not related to control flow to sequence dicts
        (control flow steps are DO, LOOP, ADV CYCLE)
        """
        seq_num = seq_from_step_num[step_num]
        if split_step == "pt2":
            seq_num = seq_num + 1
        assert seq_num is not None

        new_seq = self.blank_seq.copy()
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

        step_type = proc_step["StepType"]
        assert type(step_type) == str

        step_mode = proc_step["StepMode"]
        step_value = proc_step["StepValue"]

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
            new_seq["charge/discharge"] = (
                "Charge" if step_type == "Charge" else "Discharge"
            )

            assert get(proc_step, "Limits.Voltage") is None
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
            new_seq["charge/discharge"] = (
                "Charge" if step_type == "Charge" else "Discharge"
            )

            assert get(proc_step, "Limits.Current") is None
        else:
            raise Exception("Unsupported Charge/Discharge StepMode", step_mode)

        end_entries = get(proc_step, "Ends.EndEntry")
        end_entries_list = (
            end_entries
            if isinstance(end_entries, list)
            else []
            if end_entries is None
            else [end_entries]
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

            if goto_step_num < goto_lower_bound or goto_step_num > end_step_num:
                raise Exception(
                    "GOTO in step "
                    + str(step_num)
                    + " to location that could break loop.\nGOTO Lowerbound: "
                    + str(goto_lower_bound)
                    + "\nGOTO target step num: "
                    + str(goto_step_num)
                    + "\nGOTO upperbound (end): "
                    + str(end_step_num)
                )

            assert goto_step_num in seq_from_step_num
            if split_step == "pt1":
                goto_seq = seq_num + 1
            else:
                goto_seq = seq_from_step_num[goto_step_num]
            new_seq["lim{}_seq".format(lim_num)] = goto_seq

            if goto_step_num != step_num + 1:
                new_seq["lim{}_action".format(lim_num)] = "Goto sequence"

            if end_type == "StepTime":
                if end_oper != "=":
                    raise Exception(
                        "Unsupported StepTime operator in EndEntry", end_oper
                    )

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
                    raise Exception(
                        "Unsupported Voltage operator in EndEntry", end_oper
                    )

                lim_value, lim_value_unit = self._convert_volts(end_value)

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "Ecell"
                new_seq["lim{0}_value".format(lim_num)] = lim_value
                new_seq["lim{0}_value_unit".format(lim_num)] = lim_value_unit
            elif end_type == "Current":
                if operator_map[end_oper] is None:
                    raise Exception(
                        "Unsupported Voltage operator in EndEntry", end_oper
                    )

                lim_value, lim_value_unit = self._convert_amps(end_value)

                new_seq["lim{0}_comp".format(lim_num)] = operator_map[end_oper]
                new_seq["lim{0}_type".format(lim_num)] = "|I|"
                new_seq["lim{0}_value".format(lim_num)] = lim_value
                new_seq["lim{0}_value_unit".format(lim_num)] = lim_value_unit
            else:
                raise Exception("Unsupported EndType", end_type)

        report_entries = get(proc_step, "Reports.ReportEntry")
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

        new_seq["I Range"] = current_range

        return new_seq

    def _split_combined_step(self, step, step_num):
        """
        converts steps that have a combined control mode (CCCV) into separated steps that are compatible with BT-Lab
        """
        step_part1 = clone_deep_with(step)
        step_part2 = clone_deep_with(step)

        if isinstance(get(step_part1, "Ends.EndEntry"), list):
            indx = find_index(
                get(step_part1, "Ends.EndEntry"), lambda x: x["EndType"] == "Current"
            )
            path = "Ends.EndEntry.{}".format(indx)
        else:
            path = "Ends.EndEntry"

        if (
            step_part1["StepMode"] == "Current"
            and "Voltage" in step_part1["Limits"].keys()
        ):
            if step_part1["StepType"] == "Charge":
                set_(step_part1, path + ".Oper", ">=")
            elif step_part1["StepType"] == "Dischrge":
                set_(step_part1, path + ".Oper", "<=")
            else:
                raise NotImplementedError
            set_(step_part1, path + ".EndType", "Voltage")
            set_(step_part1, path + ".Value", step["Limits"]["Voltage"])
            set_(step_part1, path + ".Step", str(step_num + 1).zfill(3))
            step_part1["Limits"] = None

            step_part2["StepMode"] = "Voltage"
            step_part2["StepValue"] = step["Limits"]["Voltage"]
            step_part2["Limits"] = None

        elif (
            step_part1["StepMode"] == "Voltage"
            and "Current" in step_part1["Limits"].keys()
        ):
            if step_part1["StepType"] == "Charge":
                set_(step_part1, path + ".Oper", ">=")
            elif step_part1["StepType"] == "Dischrge":
                set_(step_part1, path + ".Oper", "<=")
            else:
                raise NotImplementedError
            step_part1["StepMode"] = "Current"
            step_part1["StepValue"] = step["Limits"]["Current"]
            set_(step_part1, path + ".EndType", "Voltage")
            set_(step_part1, path + ".Value", step["StepValue"])
            set_(step_part1, path + ".Step", str(step_num + 1).zfill(3))
            step_part1["Limits"] = None

            step_part2["StepMode"] = "Voltage"
            step_part2["StepValue"] = step["StepValue"]
            step_part2["Limits"] = None

        return step_part1, step_part2

    def _create_loop_seq(self, seq_num, seq_num_to_loop_to, num_loops):
        loop_seq = self.blank_seq.copy()
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
    creates the seqs that will act as an advance cycle step
    when the Biologic cycle definition is set to loop
    """

    def _create_advance_cyle_seqs(self, seq_num):
        # Biologic Modulo Bat does not support real Advance Cycle sequences
        # however we can simulate them if the cycle definition is set to Loop
        # which means that for every successful loop the cycle num advances
        #
        # we create an  "empty loop" seq that has the number of loops set to 0, this is immediately
        # passed over whenever we reach it. Immediately following that empty loop, we create
        # another loop that goes back to the empty loop exaclty once, this advances the cycle number

        if seq_num == 0:
            # the ctrl_seq field, (the seq we loop to) must be smaller the the Ns field (the seq num)
            # but also a non-negative integer this is not posssible if the seq num is 0
            raise Exception(
                "Biologic does not support advancing cycle number at step 1/seq 0"
            )

        blank_loop_seq = self._create_loop_seq(seq_num, 0, 0)
        adv_cycle_loop_seq = self._create_loop_seq(seq_num + 1, seq_num, 1)

        return blank_loop_seq, adv_cycle_loop_seq

    def _unroll_loop(self, loop, num_loops, loop_start_seq_num, loop_post_seq_num):
        unrolled_loop = loop.copy()

        # if num loops is zero, should return original loop
        for i in range(1, num_loops + 1):
            for seq in loop:
                unrolled_seq = seq.copy()
                unrolled_seq["Ns"] += len(loop) * i

                for j in range(0, 3):
                    goto_field = "lim{}_seq".format(j + 1)
                    goto_seq_num = int(seq[goto_field])
                    # make gotos internal to the loop work during unrolled portions
                    # print("do we alter?", loop_start_seq_num <= goto_seq_num and goto_seq_num < loop_post_seq_num)
                    if (
                        loop_start_seq_num <= goto_seq_num
                        and goto_seq_num <= loop_post_seq_num
                    ):
                        unrolled_seq[goto_field] = goto_seq_num + (len(loop) * i)

                unrolled_loop.append(unrolled_seq)

        assert len(unrolled_loop) == len(loop) * (num_loops + 1)

        return unrolled_loop

    """
    Converts a Maccor AST to a list of equivlanet Biologic MB seqs assuming
    cycles advance only during a loop (Cycle Definition: Loop)

    Loops that are not representable in Biologic MB format will be unrolled

    returns
        - a list of biologic seqs
    """

    def maccor_ast_to_biologic_seqs(self, maccor_ast, unroll=False):
        steps = get(maccor_ast, "MaccorTestProcedure.ProcSteps.TestStep")
        if steps is None:
            print(
                'Failure: could not find path: "MaccorTestProcedure.ProcSteps.TestStep" to steps'
            )
            return

        seqs, _ = self._maccor_steps_to_biologic_seqs(steps, unroll=unroll)
        return seqs

    """
    Converts a list of Maccor AST to a list of equivlanet Biologic MB seqs assuming
    cycles advance only during a loop (Cycle Definition: Loop)

    Loops that are not representable in Biologic MB format will be unrolled

    returns
        - a list of biologic seqs
        - a dictionary containing the mapping from output seqs to input steps
          seqs are 0-indexed, steps are 1-indexed
    """

    def _maccor_steps_to_biologic_seqs(self, steps, unroll=False):

        # To build our seqs we need to be able to handle GOTOs
        # in order to do that we a mapping between Step Numbers
        # and seq numbers. This will require us to pre-compute
        # what loops will be unrolled and what terms will be
        # re-written and use that to create a mapping of of step numbers
        # to seq numbers. Actual loop unrolling is also simplified
        # by pre-computing whether each loop meets an unroll condition
        curr_seq_num = 0
        loop_seq_count_stack = [0]
        # first entry is the base sequence, the value is never used
        # but having it improves book keeping
        loop_will_unroll_stack = [False]

        num_steps = len(steps)
        seq_num_by_step_num = {num_steps: END_SEQ_NUM}
        loop_unroll_status_by_idx = {}
        adv_cycle_ignore_status_by_idx = {}

        # tracking
        loop_start_seq_num_stack = []
        step_num_by_seq_num = {}

        assert steps[-1]["StepType"] == "End"
        for idx, step in enumerate(steps[0:-1]):
            step_num = idx + 1
            seq_num_by_step_num[step_num] = curr_seq_num
            is_last_step_in_loop = steps[idx + 1]["StepType"][0:4] == "Loop"
            curr_loop_will_unroll = loop_will_unroll_stack[-1]
            step_type = step["StepType"]

            if step_type[0:2] == "Do":
                # no nested loops in Maccor
                # we don't care about marking base protocol as will_unroll
                loop_will_unroll_stack[-1] = True
                loop_will_unroll_stack.append(False)
                loop_start_seq_num_stack.append(curr_seq_num)

                loop_seq_count_stack.append(0)
            elif (
                step_type == "AdvCycle"
                and is_last_step_in_loop
                and not curr_loop_will_unroll
            ):
                # adv cycle will occur with loop
                adv_cycle_ignore_status_by_idx[idx] = {"ignore": True}
            elif step_type == "AdvCycle":
                loop_will_unroll_stack[-1] = True
                adv_cycle_ignore_status_by_idx[idx] = {"ignore": False}

                # creating adv cycle requires 2 seqs
                step_num_by_seq_num[curr_seq_num] = step_num
                loop_seq_count_stack[-1] += 1
                curr_seq_num += 1

                step_num_by_seq_num[curr_seq_num] = step_num
                loop_seq_count_stack[-1] += 1
                curr_seq_num += 1
            elif step_type[0:4] == "Loop":
                assert step["Ends"]["EndEntry"]["EndType"] == "Loop Cnt"
                num_loops = int(step["Ends"]["EndEntry"]["Value"])

                curr_loop_start_seq = loop_start_seq_num_stack.pop()
                curr_loop_seq_count = loop_seq_count_stack.pop()

                loop_unroll_status_by_idx[idx] = {"will_unroll": curr_loop_will_unroll}
                loop_will_unroll_stack.pop()

                if unroll and curr_loop_will_unroll:
                    # add unrolled loop seqs

                    # first add tracking data
                    for seq_num in range(curr_loop_start_seq, curr_seq_num):
                        step_num_for_seq = step_num_by_seq_num[seq_num]
                        for loop_num in range(1, num_loops):
                            looped_seq_num = seq_num + (loop_num * curr_loop_seq_count)
                            step_num_by_seq_num[looped_seq_num] = step_num_for_seq

                    curr_seq_num += curr_loop_seq_count * num_loops
                    curr_loop_seq_count *= num_loops + 1
                else:
                    # add loop seq
                    step_num_by_seq_num[curr_seq_num] = step_num
                    curr_seq_num += 1
                    curr_loop_seq_count += 1

                loop_seq_count_stack[-1] += curr_loop_seq_count
            else:
                # physical step

                # last step in loop does not advance cycle, must unroll
                if unroll and is_last_step_in_loop:
                    loop_will_unroll_stack[-1] = True

                if get(step, "Limits.Current") or get(step, "Limits.Voltage"):
                    step_num_by_seq_num[curr_seq_num] = step_num
                    step_num_by_seq_num[curr_seq_num + 1] = step_num
                    loop_seq_count_stack[-1] += 2
                    curr_seq_num += 2
                else:
                    step_num_by_seq_num[curr_seq_num] = step_num
                    loop_seq_count_stack[-1] += 1
                    curr_seq_num += 1

        assert len(loop_seq_count_stack) == 1
        assert len(loop_will_unroll_stack) == 1

        pre_computed_seq_count = loop_seq_count_stack.pop()
        assert pre_computed_seq_count == curr_seq_num

        # now that we've finished our pre-computations
        # we build the seqs
        seq_loop_stack = [[]]
        loop_start_seq_num_stack = []
        step_num_goto_lower_bound = 0
        end_step_num = len(steps)

        # ignore end step
        for idx, step in enumerate(steps[0:-1]):
            step_type = step["StepType"]
            step_num = idx + 1
            seq_num = seq_num_by_step_num[step_num]

            if step_type[0:2] == "Do":
                step_num_goto_lower_bound = step_num
                seq_loop_stack.append([])
                loop_start_seq_num_stack.append(seq_num)
            elif step_type[0:4] == "Loop" and step_num:
                step_num_goto_lower_bound = step_num
                loop_start_seq_num = loop_start_seq_num_stack.pop()

                assert step["Ends"]["EndEntry"]["EndType"] == "Loop Cnt"
                num_loops = int(step["Ends"]["EndEntry"]["Value"])

                loop_will_unoll = loop_unroll_status_by_idx[idx]["will_unroll"]
                loop = seq_loop_stack.pop()

                if unroll and loop_will_unoll:
                    unrolled_loop = self._unroll_loop(
                        loop, num_loops, loop_start_seq_num, seq_num
                    )
                    assert len(unrolled_loop) == (len(loop) * (num_loops + 1))
                    print(
                        "loop ending at step {} unrolled to {} seqs".format(
                            step_num, len(unrolled_loop)
                        )
                    )
                    loop = unrolled_loop
                else:
                    loop_seq = self._create_loop_seq(
                        seq_num, loop_start_seq_num, num_loops - 1
                    )
                    loop.append(loop_seq)

                seq_loop_stack[-1] += loop
            elif step_type == "AdvCycle":
                if not adv_cycle_ignore_status_by_idx[idx]["ignore"]:
                    step_num_goto_lower_bound = step_num

                    blank_loop_seq, adv_cycle_loop_seq = self._create_advance_cyle_seqs(
                        seq_num
                    )
                    seq_loop_stack[-1].append(blank_loop_seq)
                    seq_loop_stack[-1].append(adv_cycle_loop_seq)
            elif get(step, "Limits.Voltage") or get(step, "Limits.Current"):
                step_part1, step_part2 = self._split_combined_step(step, step_num)
                seq1 = self._proc_step_to_seq(
                    step_part1,
                    step_num,
                    seq_num_by_step_num,
                    step_num_goto_lower_bound,
                    end_step_num,
                    split_step="pt1",
                )
                seq_loop_stack[-1].append(seq1)
                seq2 = self._proc_step_to_seq(
                    step_part2,
                    step_num,
                    seq_num_by_step_num,
                    step_num_goto_lower_bound,
                    end_step_num,
                    split_step="pt2",
                )
                seq_loop_stack[-1].append(seq2)
            else:
                seq = self._proc_step_to_seq(
                    step,
                    step_num,
                    seq_num_by_step_num,
                    step_num_goto_lower_bound,
                    end_step_num,
                )
                seq_loop_stack[-1].append(seq)

        assert len(seq_loop_stack) == 1

        seqs = seq_loop_stack.pop()
        assert len(seqs) == pre_computed_seq_count

        print("conversion created {} seqs".format(pre_computed_seq_count))

        return seqs, step_num_by_seq_num

    """
    returns the AST for a Maccor diagnostic file
    """

    def load_maccor_ast(self, maccorFilePath, encoding="UTF-8"):
        with open(maccorFilePath, "rb") as f:
            text = f.read().decode(encoding)

        return xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)

    def _seqs_to_str(self, seqs, col_width=20):
        seq_str = ""
        for key in OrderedDict.keys(self.blank_seq):
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
    converts biologic seqs to biologic protocol string
    resulting string assumes generated file will have
    LATIN-1 i.e. ISO-8859-1 encoding
    """

    def biologic_seqs_to_protocol_str(self, seqs, col_width=20):
        # encoding is assumed due to superscript 2 here, as well as
        # micro sign elsewhere in code, they would presumably be
        # handled by their unicode alternatives in UTF-8 but we
        # haven't seen that fileformat so we're not sure

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

    """
    convert loaded maccor AST to biologic procedure file
    """

    def maccor_ast_to_protocol_file(self, maccor_ast, fp, col_width=20):
        file_str = self.maccor_ast_to_protocol_str(maccor_ast, col_width)
        with open(fp, "wb") as f:
            f.write(file_str.encode("ISO-8859-1"))
            # f.write(file_str.encode("UTF-8"))

    """
    converts maccor AST to biologic protocol
    biologic fp should include a .mps extension
    file has LATIN-1 i.e. ISO-8859-1 encoding
        str - maccor filepath to load
        str - biologic filepath to output
        str - maccor encoding
        int - col-width for seqs, defaults to Biologic standard
    """

    def convert(
        self, maccor_fp, biologic_fp, maccor_encoding="ISO-8859-1", col_width=20
    ):
        maccor_ast = self.load_maccor_ast(maccor_fp, maccor_encoding)
        self.maccor_ast_to_protocol_file(maccor_ast, biologic_fp, col_width)

    """
    accepts a maccor AST and a predicate to filter EndEntries in each step
    by a predicate, does not mutate input AST
      OrderedDict() - maccor AST
      (OrderedDtict(), int) -> bool - Maccor EndEntry AST and step number
    """

    def remove_end_entries_by_pred(self, maccor_ast, pred):
        new_ast = copy.deepcopy(maccor_ast)
        steps = get(new_ast, "MaccorTestProcedure.ProcSteps.TestStep")
        if steps is None:
            print("Could not find any Maccor steps loaded")
            return maccor_ast

        for i, step in enumerate(steps):
            step_num = i + 1
            if get(step, "Ends.EndEntry") is None:
                continue
            elif type(get(step, "Ends.EndEntry")) == list:
                filtered = list(
                    filter(
                        lambda end_entry: pred(end_entry, step_num),
                        step["Ends"]["EndEntry"],
                    )
                )

                if len(filtered) == 0:
                    unset(step, "Ends.EndEntry")
                elif len(filtered) == 1:
                    set_(step, "Ends.EndEntry", filtered[0])
                else:
                    set_(step, "Ends.EndEntry", filtered)
            else:
                if not pred(get(step, "Ends.EndEntry"), step_num):
                    unset(step, "Ends.EndEntry")

        return new_ast


class CycleTransitionRules:
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
            + "  adv_cycle_seq_transitions: {},\n".format(
                self.adv_cycle_seq_transitions
            )
            + "  debug_adv_cycle_on_step_transitions: {},\n".format(
                self.debug_adv_cycle_on_step_transitions
            )
            + "}\n"
        )


class CycleTransitionRulesSerializer:
    def json(self, cycle_transition_rules, indent=2):
        parseable_adv_cycle_seq_transitions = []
        for (
            s,
            t,
        ), adv_cycle_count in cycle_transition_rules.adv_cycle_seq_transitions.items():
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
        ), adv_cycle_count in (
            cycle_transition_rules.debug_adv_cycle_on_step_transitions.items()
        ):
            parseable_debug_adv_cycle_on_step_transitions.append(
                {
                    "source": s,
                    "target": t,
                    "adv_cycle_count": adv_cycle_count,
                }
            )

        obj = {
            "tech_num": cycle_transition_rules.tech_num,
            "tech_does_loop": cycle_transition_rules.tech_does_loop,
            "adv_cycle_on_start": cycle_transition_rules.adv_cycle_on_start,
            "adv_cycle_on_tech_loop": cycle_transition_rules.adv_cycle_on_tech_loop,
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

        return CycleTransitionRules(
            tech_num,
            tech_does_loop,
            adv_cycle_on_start,
            adv_cycle_on_tech_loop,
            adv_cycle_seq_transitions,
            debug_adv_cycle_on_step_transitions,
        )


# Will Powelson May 18, 2021
# The problem:
# We need a way to calculate cycle index from a maccor protocol, not all protocols
# are representable in biolgic. Nested loops require breaking the conversion into
# multiple techniques which may break certain GOTO functionality, which we are accepting.
#
# Biologic also lacks the ability to control cycles, we need to infer cycle index from
# the output data using changes to sequence number, technique number, technique loops
# and number of changes in sequence number.
#
# The plan:
# How splitting works
# Given a list of steps we want to break out any nested loop into a technique
# consider this beautful ascii art visualizing a maccor protocol's control flow
# sans GOTOs
# 1 |\
# 2 | |
# 3 |/
# 4 |\
# 5 |  \
# 6 |\  |
# 7 | | |
# 8 |/  |
# 9 |  /
# 10|/
# 11|
# 12|
# 13|
#
# the loop from 4-10 has an inner loop which is not easily representable in
# in a single biologic technique, we will break this into a single technique,
# and use a loop technique in place of the outer loop, so we end up with
#
# technique 1
# 1 |\
# 2 | |
# 3 |/
#
# technique 2
# 4 |
# 5 |
# 6 |\
# 7 | |
# 8 |/
# 9 |
# 10|
#
# technique 3 loops back to technique 2
#
# technique 4
# 11|
# 12|
# 13|
#
# Now we can calculate how many cycle advances there are between techniques
# or when a technique loops based on analysis of a much simpler structure,
# each technique only needs to know if there were any cycle advances not applied from
# its predecessor.
#
def get_cycle_adv_data_by_tech_num(maccor_test_steps):
    assert get(maccor_test_steps[-1], "StepType") == "End"
    maccor_test_steps = maccor_test_steps[:-1]

    tech_edges = []
    techs_that_loop = set()
    loop_1_start = -1
    new_tech_flag = False
    for i, step in enumerate(maccor_test_steps):
        step_type = get(step, "StepType")
        if step_type == "Do 1":
            loop_1_start = i
        elif step_type == "Do 2":
            new_tech_flag = True
        elif step_type == "Loop 1" and new_tech_flag:
            # if two nested loops were back to back, edge was already added
            if len(tech_edges) == 0 or tech_edges[-1] != loop_1_start:
                tech_edges.append(loop_1_start)
            techs_that_loop.add(len(tech_edges))
            tech_edges.append(i + 1)

            new_tech_flag = False

    techs = []

    remaining = list(map(lambda x: (x[0] + 1, x[1]), enumerate(maccor_test_steps)))
    for edge in reversed(tech_edges):
        tech = remaining[edge:]
        techs.insert(0, remaining[edge:])
        remaining = remaining[:edge]

    if len(remaining) > 0:
        techs.insert(0, remaining)

    loop_open_types = ["Do 1", "Do 2"]
    loop_types = ["Loop 1", "Loop 2"]

    cycle_adv_data_by_tech_num = {}
    open_adv_cycles = 0
    tech_num = 1
    for i, tech in enumerate(techs):
        tech_loops = i in techs_that_loop
        cycle_data = {
            "tech_num": tech_num,
            "tech_loops": tech_loops,
            "adv_cycle_on_start": open_adv_cycles,
            "adv_cycle_on_tech_loop": 0,
            "adv_cycle_step_transitions": {},
        }

        open_adv_cycles = 0
        prev_physical_step_num = -1
        # remove outer loop edges for loop tech, we don't need them
        enumerable_steps = tech[1:-1] if tech_loops else tech
        loop_to_idx = -1
        for curr_idx, (step_num, step) in enumerate(enumerable_steps):
            step_type = get(step, "StepType")

            if step_type in loop_open_types:
                loop_to_idx = curr_idx + 1
            elif step_type in loop_types:
                if prev_physical_step_num == -1:
                    # can probably be handled if necessary, but don't want to deal rn
                    # add adv cyles from loop to cycle start, and cycle loop (if necessary)
                    raise Exception(
                        "step {} is a loop with no physical steps".format(
                            step_num
                        ).format(step_num)
                    )

                # use the previously recorded open loop idx to figure out
                # where we loop to, find the next physical step after a loop
                # and calculate the number of cycles to advance (if any)
                looped_physical_step_num = -1
                loop_adv_cycles = open_adv_cycles
                for looped_step_num, looped_step in enumerable_steps[
                    loop_to_idx:curr_idx
                ]:
                    step_type = get(looped_step, "StepType")
                    if step_type == "AdvCycle":
                        loop_adv_cycles += 1
                    else:
                        # inner loops not possible within a loop since we split
                        # out nested loops into different techniques so step is physical
                        looped_physical_step_num = looped_step_num
                        break

                if looped_physical_step_num == -1:
                    # can probably just increase open advanced cycles, but don't want to deal rn
                    raise Exception(
                        "step {} is a loop with no physical steps".format(
                            step_num
                        ).format(step_num)
                    )
                elif loop_adv_cycles > 0:
                    transition = (prev_physical_step_num, looped_physical_step_num)
                    cycle_data["adv_cycle_step_transitions"][
                        transition
                    ] = loop_adv_cycles
            elif step_type == "AdvCycle" and prev_physical_step_num == -1:
                cycle_data["adv_cycle_on_start"] += 1
                if tech_loops:
                    cycle_data["adv_cycle_on_loop"] += 1
            elif step_type == "AdvCycle":
                open_adv_cycles += 1
            else:
                if prev_physical_step_num != -1 and open_adv_cycles > 0:
                    transition = (prev_physical_step_num, step_num)
                    cycle_data["adv_cycle_step_transitions"][
                        transition
                    ] = open_adv_cycles
                    # exhaust the adv cycle
                    open_adv_cycles = 0
                prev_physical_step_num = step_num

        if tech_loops:
            cycle_data["adv_cycle_on_loop"] = open_adv_cycles
        cycle_adv_data_by_tech_num[tech_num] = cycle_data

        tech_num += 1
        if tech_loops:
            tech_num += 1

    return cycle_adv_data_by_tech_num


# have unalloyed transition pairs (66, 67) | (s, t)
# have (offset_step_num, [tech_nums])
# without loop unrolling, [tech_nums] is contiguous sequence of integers
#
# translate (s, t) -> (s', t')
def create_seq_num_adv_cycle_data(cycle_adv_data, step_num_by_seq_num, step_num_offset):
    seq_nums_by_step_num = {}
    # we want the first seq num a step was split into to appear at index 0, last and -1
    for seq_num, step_num in sorted(
        step_num_by_seq_num.items(), key=lambda x: (x[1], x[0])
    ):
        if step_num not in seq_nums_by_step_num:
            seq_nums_by_step_num[step_num] = []
        seq_nums_by_step_num[step_num].append(seq_num)

    tech_num = cycle_adv_data["tech_num"]
    tech_loops = cycle_adv_data["tech_loops"]
    adv_cycle_on_start = cycle_adv_data["adv_cycle_on_start"]
    adv_cycle_on_tech_loop = cycle_adv_data["adv_cycle_on_tech_loop"]
    adv_cycle_step_transitions = cycle_adv_data["adv_cycle_step_transitions"]

    adv_cycle_seq_transitions = {}
    for (s_, t_), num_adv_cycles in adv_cycle_step_transitions.items():
        s = s_ - step_num_offset
        t = t_ - step_num_offset
        s_seq = seq_nums_by_step_num[s][-1]
        t_seq = seq_nums_by_step_num[t][0]

        transition = (s_seq, t_seq)

        adv_cycle_seq_transitions[transition] = num_adv_cycles

    return CycleTransitionRules(
        tech_num,
        tech_loops,
        adv_cycle_on_start,
        adv_cycle_on_tech_loop,
        adv_cycle_seq_transitions,
        adv_cycle_step_transitions,
    )


"""
helper function for converting diagnosticV5.000
"""


def convert_diagnostic_v5():
    def pred(end_entry, step_num):
        # remove end entries going to step 70 or 94 unless
        # except when they are the next step
        goto_step = int(end_entry["Step"])

        goto_70_not_next = goto_step == 70 and step_num != 69
        goto_94_not_next = goto_step == 94 and step_num != 93

        return not (goto_70_not_next or goto_94_not_next)

    converter = MaccorToBiologicMb()
    ast = converter.load_maccor_ast(
        os.path.join(PROCEDURE_TEMPLATE_DIR, "diagnosticV5.000")
    )

    # set main looping value to 20
    set_(ast, "MaccorTestProcedure.ProcSteps.TestStep.68.Ends.EndEntry.Value", "20")
    filtered = converter.remove_end_entries_by_pred(ast, pred)

    # gotos for step 94 were in case of unsafe Voltage levels
    # we'll set them in the output seqs
    seqs = converter.maccor_ast_to_biologic_seqs(filtered)
    for seq in seqs:
        seq["E range min (V)"] = "2.000"
        seq["E range max (V)"] = "4.400"

    converter.biologic_seqs_to_protocol_file(
        seqs, os.path.join(BIOLOGIC_TEMPLATE_DIR, "diagnosticV5_70.mps")
    )


def convert_diagnostic_v5_segment_files():
    def is_acceptable_goto(end_entry, step_num):
        # remove end entries going to step 70 or 94 unless
        # except when they are the next step
        goto_step = int(end_entry["Step"])

        goto_70_not_next = goto_step == 70 and step_num != 69
        goto_94_not_next = goto_step == 94 and step_num != 93

        return not (goto_70_not_next or goto_94_not_next)

    converter = MaccorToBiologicMb()
    ast = converter.remove_end_entries_by_pred(
        converter.load_maccor_ast(
            os.path.join(PROCEDURE_TEMPLATE_DIR, "diagnosticV5.000")
        ),
        is_acceptable_goto,
    )

    closing_loop_step_by_index = dict()
    open_loop_start_indices = []
    original_steps = get(ast, "MaccorTestProcedure.ProcSteps.TestStep")
    for i, step in enumerate(original_steps):
        step_type = get(step, "StepType")
        if step_type[0:2] == "Do":
            open_loop_start_indices.append(i)
        elif step_type[0:4] == "Loop":
            set_(step, "Ends.EndEntry.Value", "1")

            open_loop_start = open_loop_start_indices.pop()
            closing_loop_step_by_index[open_loop_start] = step

    end_step = original_steps[-1]
    steps = []
    open_loop_starts = deque()
    loop_lens = [0]
    segment_count = 0
    segment_filename_template = "diagnosticV5segment{}.mps"
    for i, step in enumerate(get(ast, "MaccorTestProcedure.ProcSteps.TestStep")):
        step_type = get(step, "StepType")

        if (step_type[0:2] == "Do" or step_type[0:4] == "Loop") and loop_lens[-1] > 0:
            segment = copy.deepcopy(steps)
            segment_count += 1

            for loop_start_index in open_loop_starts:
                loop_close_step = copy.deepcopy(
                    closing_loop_step_by_index[loop_start_index]
                )

                next_step = len(segment) + 2
                set_(loop_close_step, "Ends.EndEntry.Step", "{:03d}".format(next_step))

                segment.append(loop_close_step)
            segment.append(end_step)

            max_step_num = len(segment) + 1

            def is_less_than_max_step_num(end_entry, step_num):
                goto_step = int(end_entry["Step"])
                return goto_step <= max_step_num

            filtered = converter.remove_end_entries_by_pred(
                set_(ast, "MaccorTestProcedure.ProcSteps.TestStep", segment),
                is_less_than_max_step_num,
            )

            seqs = converter.maccor_ast_to_biologic_seqs(filtered)
            for seq in seqs:
                seq["E range min (V)"] = "2.900"
                seq["E range max (V)"] = "4.300"

            filename = segment_filename_template.format(segment_count)
            converter.biologic_seqs_to_protocol_file(
                seqs, os.path.join(BIOLOGIC_TEMPLATE_DIR, "segments", filename)
            )
            print("created {}".format(filename))

        steps.append(step)
        if step_type[0:2] == "Do":
            open_loop_starts.appendleft(i)
            loop_lens.append(0)
        elif step_type[0:4] == "Loop":
            open_loop_starts.popleft()
            loop_lens.pop()
        else:
            loop_lens[-1] += 1


def convert_diagnostic_v5_multi_techniques(source_file="BioTest_000001.000"):
    def is_acceptable_goto(end_entry, step_num):
        # remove end entries going to step 70 or 94 unless
        # except when they are the next step
        goto_step = int(end_entry["Step"])

        goto_70_not_next = goto_step == 70 and step_num != 69
        goto_94_not_next = goto_step == 94 and step_num != 93
        leaves_technique_1 = step_num < 37 and goto_step > 37
        leaves_technique_2 = step_num < 70 and goto_step > 70
        redundant_voltage = (
            step_num in [94, 95]
            and end_entry["EndType"] == "Voltage"
            and goto_step == 96
        )

        return not (
            goto_70_not_next
            or goto_94_not_next
            or leaves_technique_1
            or leaves_technique_2
            or redundant_voltage
        )

    def sub_goto_step_nums(steps, sub):
        subbed_steps = copy.deepcopy(steps)
        for step in subbed_steps:
            end_entries = get(step, "Ends.EndEntry")
            if not end_entries:
                continue

            if type(end_entries) != list:
                end_entries = [end_entries]

            for end_entry in end_entries:
                step_num = int(get(end_entry, "Step"))
                set_(end_entry, "Step", "{:03d}".format(step_num - sub))

        return subbed_steps

    def set_global_fields(seqs):
        for seq in seqs:
            seq["E range min (V)"] = "2.900"
            seq["E range max (V)"] = "4.300"

    converter = MaccorToBiologicMb()
    ast = converter.remove_end_entries_by_pred(
        converter.load_maccor_ast(os.path.join(PROCEDURE_TEMPLATE_DIR, source_file)),
        is_acceptable_goto,
    )

    steps = get(ast, "MaccorTestProcedure.ProcSteps.TestStep")
    main_loop_start_i = 36
    main_loop_end_i = 68

    end_step = steps[-1]

    tech1_steps = steps[0:main_loop_start_i]
    tech1_steps.append(end_step)
    tech1_steps = copy.deepcopy(tech1_steps)

    # remove loop start/end
    tech2_start_i = main_loop_start_i + 1

    tech2_main_loop_steps = steps[tech2_start_i:main_loop_end_i]
    tech2_main_loop_steps.append(end_step)
    tech2_main_loop_steps = sub_goto_step_nums(
        tech2_main_loop_steps,
        tech2_start_i,
    )

    # bring loop level down 1
    for step in tech2_main_loop_steps:
        if get(step, "StepType") == "Loop 2":
            set_(step, "StepType", "Loop 1")

        if get(step, "StepType") == "Do 2":
            set_(step, "StepType", "Do 1")

    tech4_start_i = main_loop_end_i + 1
    tech4_steps = steps[tech4_start_i:]
    tech4_steps = sub_goto_step_nums(
        tech4_steps,
        tech4_start_i,
    )

    assert get(tech4_steps[1], "Ends.EndEntry.Step") == "003"

    print("*** creating technique 1")
    tech1_seqs, tech1_seq_map = converter._maccor_steps_to_biologic_seqs(tech1_steps)
    set_global_fields(tech1_seqs)

    print("*** creating technique 2")
    tech2_seqs, tech2_seq_map = converter._maccor_steps_to_biologic_seqs(
        tech2_main_loop_steps
    )
    set_global_fields(tech2_seqs)

    print("*** creating technique 4")
    tech4_seqs, tech4_seq_map = converter._maccor_steps_to_biologic_seqs(tech4_steps)
    set_global_fields(tech4_seqs)

    modulo_bat_template = "\r\n" "Technique : {}\r\n" "Modulo Bat\r\n"

    technique_1_str = modulo_bat_template.format(1)
    technique_1_str += converter._seqs_to_str(tech1_seqs)

    technique_2_str = modulo_bat_template.format(2)
    technique_2_str += converter._seqs_to_str(tech2_seqs)

    technique_3_str = (
        "\r\n"
        "Technique : 3\r\n"
        "Loop\r\n"
        "goto Ne             2                   \r\n"
        "nt times            1000                \r\n"
    )

    technique_4_str = modulo_bat_template.format(4)
    technique_4_str += converter._seqs_to_str(tech4_seqs)

    techniques = [technique_1_str, technique_2_str, technique_3_str, technique_4_str]

    file_str = (
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
    ).format(len(techniques))

    for technique in techniques:
        file_str += technique

    filename = "{}_split_techniques.mps".format(source_file)
    fp = os.path.join(BIOLOGIC_TEMPLATE_DIR, filename)
    with open(fp, "wb") as f:
        f.write(file_str.encode("ISO-8859-1"))
        print("created {}".format(fp))

    # create output mapping:
    mappings = [
        (tech1_seqs, tech1_seq_map, 0, "Technique 1:"),
        (tech1_seqs, tech2_seq_map, tech2_start_i, "Technique 2:"),
        (tech2_seqs, tech4_seq_map, tech4_start_i, "Technique 4:"),
    ]
    print("creating transition objects")
    step_trans_data = get_cycle_adv_data_by_tech_num(steps)
    t1_trans = create_seq_num_adv_cycle_data(step_trans_data[1], tech1_seq_map, 0)
    t2_trans = create_seq_num_adv_cycle_data(
        step_trans_data[2], tech2_seq_map, tech2_start_i
    )
    t4_trans = create_seq_num_adv_cycle_data(
        step_trans_data[4], tech4_seq_map, tech4_start_i
    )

    serializer = CycleTransitionRulesSerializer()
    for tech_num, cycle_transition_rules in [
        (1, t1_trans),
        (2, t2_trans),
        (4, t4_trans),
    ]:
        file_str = serializer.json(cycle_transition_rules)
        print(serializer.parse_json(file_str))

        filename = "{}.technique_{}_cycle_rules.json".format(source_file, tech_num)
        fp = os.path.join(BIOLOGIC_TEMPLATE_DIR, filename)
        with open(fp, "wb") as f:
            f.write(file_str.encode("utf-8"))
            print("created {}".format(fp))

    file_str = ""
    for _, tech_map, step_num_offset, header in mappings:
        file_str += header + "\r\n"

        pairs = list(tech_map.items())
        pairs.sort(key=lambda x: x[0])

        for seq_num, step_num in pairs:
            file_str += "seq {} generated from step {}\r\n".format(
                seq_num, step_num + step_num_offset
            )
        file_str += "\r\n"

    filename = "{}_split_techniques.mps.step-mapping.txt".format(source_file)
    fp = os.path.join(BIOLOGIC_TEMPLATE_DIR, filename)
    with open(fp, "wb") as f:
        f.write(file_str.encode("ISO-8859-1"))
        print("created {}".format(fp))

    file_str = ""
    for tech_seqs, tech_map, step_num_offset, tech_num_str in mappings:
        pairs = list(tech_map.items())
        pairs.sort(key=lambda x: x[0])

        for seq_num, adjusted_step_num in pairs:
            step_num = adjusted_step_num + step_num_offset

            seq = tech_seqs[seq_num]
            step = steps[step_num - 1]

            file_str += "{} seq:{}, step num:{}\r\n".format(
                tech_num_str[:-1], seq_num, step_num
            )

            step_lines = xmltodict.unparse({"TestStep": step}, pretty=True).split("\n")
            # remove xmlheader
            step_lines = step_lines[1:]

            seq_lines = converter._seqs_to_str([seq]).split("\r\n")
            # remove blankline
            seq_lines = seq_lines[:-1]

            num_lines = max(len(seq_lines), len(step_lines))
            for i in range(0, num_lines):
                seq_str = seq_lines[i] if i < len(seq_lines) else ""
                step_str = step_lines[i] if i < len(step_lines) else ""
                file_str += "{:<40s}{}\r\n".format(seq_str, step_str)

            file_str += "\r\n"

    file_str += "\r\n"

    filename = "{}_split_techniques.mps.step-mapping-verbose.txt".format(source_file)
    fp = os.path.join(BIOLOGIC_TEMPLATE_DIR, filename)
    with open(fp, "wb") as f:
        f.write(file_str.encode("ISO-8859-1"))
        print("created {}".format(fp))


"""
Processes CSV files generated from several biologic techniques
and creates a new set of CSVs with an additional "cycle_index" column.

accepts
  - technique_csv_file_paths: list of file paths to Biologic CSVs
  - technique_serialized_transition_rules_file_paths: list of file paths to serialized CycleTransitionRules
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
    assert len(technique_csv_file_paths) == len(
        technique_serialized_transition_rules_file_paths
    )

    technique_conversion_filepaths = zip(
        technique_csv_file_paths,
        technique_serialized_transition_rules_file_paths,
        technique_csv_out_file_paths,
    )

    serializer = CycleTransitionRulesSerializer()
    cycle_num = 1
    for csv_fp, serialized_transtion_fp, csv_out_fp in technique_conversion_filepaths:
        with open(serialized_transtion_fp, "r") as f:
            data = f.read()
            cycle_transition_rules = serializer.parse_json(data)

        df = pd.read_csv(csv_fp, sep=";")

        cycle_num += cycle_transition_rules.adv_cycle_on_start

        prev_seq_num = int(df.iloc[0]["Ns"])
        prev_loop_num = int(df.iloc[0]["Loop"])
        cycle_nums = []
        for _, row in df.iterrows():
            seq_num = int(row["Ns"])
            loop_num = int(row["Loop"])

            # a transition may occur because of a loop technique or a loop seq,
            # it is possible to double count cycle advances if we don't handle them separately
            if loop_num != prev_loop_num:
                cycle_num += cycle_transition_rules.adv_cycle_on_tech_loop

            elif seq_num != prev_seq_num:
                transition = (prev_seq_num, seq_num)
                cycle_num += cycle_transition_rules.adv_cycle_seq_transitions.get(
                    transition, 0
                )

            prev_loop_num = loop_num
            prev_seq_num = seq_num

            cycle_nums.append(cycle_num)

        df["cycle_index"] = cycle_nums
        df.to_csv(csv_out_fp, sep=";")
