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
import xmltodict
from beep.protocol import PROTOCOL_SCHEMA_DIR
from monty.serialization import loadfn
from collections import OrderedDict

# magic number for biologic
END_STEP_NUM = 9999

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


    """
    Converts Maccor protcol file to biologic file
    if loops do not advance a step immediately before looping, will attempt to unroll the loop
    and provide a [encoding-REPLACE] in the "initial state" field of the biologic file.
    """
    def convert(self, maccorFilePath, outputPath, encoding="UTF-8"):
      with open(maccorFilePath, "rb") as f:
          text = f.read().decode(encoding)
      maccor_ast = xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)

      steps = maccor_ast["MaccorTestProcedure"]["ProcSteps"]["TestStep"]
      step_loop_structure = self._steps_to_loop_structure(steps)
      print(self._count_seqs(step_loop_structure))
      print(self._steps_to_loop_structure2(steps))

    def _steps_to_loop_structure(self, steps):
        step_loop_structure = [{
            "loop_count": 1,
            "steps": [],
            "isBase": True,
        }]

        for step in steps:
            if step["StepType"][0:2] == "Do":
                step_loop_structure.append({
                    "loop_count": 1,
                    "steps": [],
                    "isBase": False
                })
            elif step["StepType"][0:4] == "Loop":
                loop = step_loop_structure.pop()
                
                loop_count_str = step["Ends"]["EndEntry"]["Value"]
                loop["loop_count"] = int(loop_count_str)
                step_loop_structure[-1]["steps"].append(loop)

            else:
                step_loop_structure[-1]["steps"].append(step)

        assert len(step_loop_structure) == 1
        return step_loop_structure[-1]

    
    def _count_seqs(self, step_loop_structure, depth = 0):
        num_seqs = 0
        will_unroll = False
        loop_count = step_loop_structure["loop_count"]
        steps = step_loop_structure["steps"]


        # print(step_loop_structure["steps"][3])
        for i, step in enumerate(steps):
            if "loop_count" in step:
                will_unroll = True
                num_seqs += self._count_seqs(step, depth + 1)
            elif step["StepType"] == "AdvCycle" and i != len(steps) - 1:
                will_unroll = True
                num_seqs += 2
            elif step["StepType"] == "AdvCycle" and i == len(steps) - 1 and not will_unroll:
                num_seqs += 0
            elif step["StepType"] != "AdvCycle" and i == len(steps) - 1:
                will_unroll = True
                num_seqs += 1
            else:
                num_seqs += 1


        final_seq_count = num_seqs + 1 if not will_unroll else num_seqs * loop_count

        return final_seq_count


    def _steps_to_loop_structure2(self, steps):
        curr_seq_num = 0
        loop_seq_count_stack = [0]
        loop_will_unroll_stack = [False]

        seq_num_by_step_num = {}
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
                loop_will_unroll_stack[-1] = True

                loop_will_unroll_stack.append(False)
                loop_seq_count_stack.append(0)
            elif step_type == "AdvCycle" and is_last_step_in_loop and not curr_loop_will_unroll:
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
                
                loop_will_unroll_stack.pop()
                loop_unroll_status_by_idx[idx] = {"will_unroll": curr_loop_will_unroll}

                if curr_loop_will_unroll:
                    # add unrolled loop seqs
                    curr_seq_num += curr_loop_seq_count * (num_loops - 1) 
                    curr_loop_seq_count *= num_loops
                else:
                    # add loop seq
                    curr_seq_num += 1
                    curr_loop_seq_count += 1
                    
                loop_seq_count_stack[-1] += curr_loop_seq_count
            elif is_last_step_in_loop:
                #last step is not adv cycle, must unroll
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

        seq_loop_stack = [[]]
        step_num_goto_lower_bound = 0
        end_step_num = len(steps)

        #ignore end step
        for idx, step in enumerate(steps[0:-1]):
            step_num = idx + 1
            step_type = step["StepType"]

            if step_type[0:2] == "Do":
                step_num_goto_lower_bound = step_num
                seq_loop_stack.append([])
            elif step_type[0:4] == "Loop" and step_num:
                step_num_goto_lower_bound = step_num
                
                assert step["Ends"]["EndEntry"]["EndType"] == "Loop Cnt"
                num_loops = int(step["Ends"]["EndEntry"]["Value"])

                loop_will_unoll = loop_unroll_status_by_idx[idx]["will_unroll"]
                loop = seq_loop_stack.pop()
                
                if loop_will_unoll:
                    loop = loop * num_loops
                else:
                    loop.append({"loop_seq": True})

                seq_loop_stack[-1] += loop
            elif step_type == "AdvCycle":
                if not adv_cycle_ignore_status_by_idx[idx]["ignore"]:
                    step_num_goto_lower_bound = step_num
                    
                    seq_loop_stack[-1].append({"adv_dummy": True})
                    seq_loop_stack[-1].append({"adv": True})
            else:
                seq = self._proc_step_to_seq(step, step_num, seq_num_by_step_num, step_num_goto_lower_bound, end_step_num)
                seq_loop_stack[-1].append({"step": True})


        assert len(seq_loop_stack) == 1

        seqs = seq_loop_stack.pop() 

        return len(seqs)

    def _create_pseudo_adv_cycle(self, seq_num):
        print("beep")

    def _proc_step_to_seq(self, test_step, step_num, seq_from_step_num, goto_lower_bound, end_step_num):
        """
        converts steps that are not related to control flow to sequence dicts
        (control flow steps are DO, LOOP, ADV CYCLE)
        """
        seq_num = seq_from_step_num[step_num]
        assert seq_num is not None

        print(step_num, seq_num, test_step["StepType"])

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
        # assert num_lims <= 3
        if num_lims <= 3:
            end_entries_list = end_entries_list[0:3]

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

            goto_seq = seq_from_step_num[goto_step_num]
            assert goto_seq is not None

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
            elif goto_step_num != step_num + 1:
                new_seq["lim{}_action".format(lim_num)] = "Goto sequence"
            
            new_seq["lim{}_action".format(lim_num)] = goto_seq

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
                new_seq["lim{0}_value".format(lim_num)] = ms
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
        new_seq["rec_nb"] = num_reports

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


converter = MaccorToBiologicMb()
converter.convert("/home/cal/tri/beep/beep/protocol/procedure_templates/diagnosticV4.000", "")