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
from pydash import get, unset, set_

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
            return "{:.3f}".format(int(hours)), "hr"

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
        self, proc_step, step_num, seq_from_step_num, goto_lower_bound, end_step_num
    ):
        """
        converts steps that are not related to control flow to sequence dicts
        (control flow steps are DO, LOOP, ADV CYCLE)
        """
        seq_num = seq_from_step_num[step_num]
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

            voltage_lim = get(proc_step, "Limits.Voltage")
            if voltage_lim is not None:
                voltage_lim_val, voltage_lim_unit = self._convert_volts(voltage_lim)
                new_seq["ctrl2_val"] = voltage_lim_val
                new_seq["ctrl2_val_unit"] = voltage_lim_unit
        elif step_mode == "Voltage":
            # does this need to be formatted? e.g. 1.0 from Maccor vs 1.000 for biologic
            assert type(step_value) == str

            ctrl1_val, ctrl1_val_unit = self._convert_amps(step_value)
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

            current_lim = get(proc_step, "Limits.Current")
            if current_lim is not None:
                current_lim_val, current_lim_unit = self._convert_amps(current_lim)
                new_seq["ctrl2_val"] = current_lim_val
                new_seq["ctrl2_val_unit"] = current_lim_unit
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
        num_lims = len(end_entries_list)
        if num_lims > 3:
            raise Exception(
                (
                    "Step {} has more than 3 EndEntries, the max allowed"
                    " by Biologic. Either remove some limits from the source"
                    " loaded diagnostic file or filter by number using the"
                    " remove_end_entries_by_pred method"
                ).format(step_num)
            )

        new_seq["lim_nb"] = num_lims

        for idx, end_entry in enumerate(end_entries_list):
            lim_num = idx + 1

            end_type = end_entry["EndType"]
            assert type(end_type) == str

            end_oper = end_entry["Oper"]
            assert type(end_oper) == str

            end_value = end_entry["Value"]
            assert type(end_value) == str

            goto_step_num_str = end_entry["Step"]
            assert type(goto_step_num_str) == str
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
                new_seq["lim{0}_type".format(lim_num)] = "I"
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

        return new_seq

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
    """

    def maccor_ast_to_biologic_seqs(self, maccor_ast):
        steps = get(maccor_ast, "MaccorTestProcedure.ProcSteps.TestStep")
        if steps is None:
            print(
                'Failure: could not find path: "MaccorTestProcedure.ProcSteps.TestStep" to steps'
            )
            return

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
                loop_seq_count_stack[-1] += 2
                curr_seq_num += 2
            elif step_type[0:4] == "Loop":
                assert step["Ends"]["EndEntry"]["EndType"] == "Loop Cnt"
                num_loops = int(step["Ends"]["EndEntry"]["Value"])
                curr_loop_seq_count = loop_seq_count_stack.pop()

                loop_unroll_status_by_idx[idx] = {"will_unroll": curr_loop_will_unroll}
                loop_will_unroll_stack.pop()

                if curr_loop_will_unroll:
                    # add unrolled loop seqs
                    # curr_loop_seq_count *= num_loops
                    curr_seq_num += curr_loop_seq_count * num_loops
                    curr_loop_seq_count *= num_loops + 1
                else:
                    # add loop seq
                    curr_seq_num += 1
                    curr_loop_seq_count += 1

                loop_seq_count_stack[-1] += curr_loop_seq_count
            elif is_last_step_in_loop:
                # last step is not adv cycle, must unroll
                loop_will_unroll_stack[-1] = True

                loop_seq_count_stack[-1] += 1
                curr_seq_num += 1
            else:
                # physical step
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

                if loop_will_unoll:
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
                        seq_num, loop_start_seq_num, num_loops
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
        return seqs

    """
    returns the AST for a Maccor diagnostic file
    """

    def load_maccor_ast(self, maccorFilePath, encoding="UTF-8"):
        with open(maccorFilePath, "rb") as f:
            text = f.read().decode(encoding)

        return xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)

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
            file_str += field_row + "\r\n"

        return file_str

    """
    converts maccor AST to biologic protocol
    resulting string assumes generated file will have
    LATIN-1 i.e. ISO-8859-1 encoding
    """

    def maccor_ast_to_protocol_str(self, maccor_ast, col_width=20):
        seqs = self.maccor_ast_to_biologic_seqs(maccor_ast)
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
                seq["E range min (V)"] = "0.000"
                seq["E range max (V)"] = "4.100"

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
