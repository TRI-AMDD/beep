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
""" Parsing and conversion of biologic Modulo Bat files to Maccor Procedure Files"""
import os
import re
from copy import deepcopy
from collections import OrderedDict
import xmltodict
from monty.serialization import loadfn
from beep.protocol.biologic import Settings
from beep.protocol.maccor import Procedure
from beep.protocol import PROCEDURE_TEMPLATE_DIR


MACCOR_TEMPLATE = loadfn(os.path.join(PROCEDURE_TEMPLATE_DIR, "maccor_schemas.yaml"))

_default_maccor_header = OrderedDict(MACCOR_TEMPLATE["default_maccor_header"])
_blank_step = OrderedDict(MACCOR_TEMPLATE["blank_step"])
_blank_end_entry = OrderedDict(MACCOR_TEMPLATE["blank_end_entry"])
_blank_report_entry = OrderedDict(MACCOR_TEMPLATE["blank_report_entry"])
_blank_maccor_body = OrderedDict(MACCOR_TEMPLATE["blank_maccor_body"])


class BiologicMbToMaccorProcedure:
    @classmethod
    def convert(
        cls,
        mps_filepath,
        out_filepath,
        mps_file_encoding="ISO-8859-1",
        maccor_header=_default_maccor_header,
    ):
        """
        Biologic Modulo Bat file ingestion from
        *.mps file with a single Modulo Bat technique

        Args:
            mps_filepath(str): *.mps filepath
            out_filepath(str): target destination for Maccor Procedure
            (optional) mps_file_encoding(str): text encoding of *.mps
            (optional) maccor_header(OrderedDict): This defines the shape of the <header> in
                the output file

        Returns: None on success, otherwise throws an error
        """
        if not os.path.isfile(mps_filepath):
            raise Exception(
                "Could not load {}, filepath does not exist".format(mps_filepath)
            )

        with open(mps_filepath, "rb") as mps_file:
            raw_text = mps_file.read()
            text = raw_text.decode(mps_file_encoding)

        protocol_xml = cls.biologic_mb_text_to_maccor_xml(text, maccor_header)

        with open(out_filepath, "w+") as out_file:
            out_file.write(protocol_xml)

        return None

    """
    Biologic Modulo Bat text ingestion from
    *.mps file with a single Modulo Bat technique

    Args:
        text (str): the text from a *.mps file

    Returns:
        (xml): string containing the XML for a Maccor Procedure
    """

    @classmethod
    def biologic_mb_text_to_maccor_xml(
        cls, mps_text, maccor_header=_default_maccor_header
    ):
        schedule_dict = Settings.mps_text_to_schedule_dict(mps_text)

        seqs = cls._get_seqs(schedule_dict)
        steps = cls._create_steps(seqs)

        body = deepcopy(_blank_maccor_body)
        body["MaccorTestProcedure"]["header"] = maccor_header
        body["MaccorTestProcedure"]["ProcSteps"] = OrderedDict({"TestStep": steps})

        newLine = "\r\n"
        xml_with_dirty_empty_elemnts = xmltodict.unparse(
            body, encoding="utf-8", pretty=True, indent="  ", newl=newLine
        )
        xml = Procedure.fixup_empty_elements(xml_with_dirty_empty_elemnts)
        # xmltodict's unparse does not accept custom processing instructions,
        # we must add Maccor's ourselves
        xml_with_maccor_processing_instruction = re.sub(
            r"^[^\r^\n]*",
            '<?xml version="1.0" encoding="UTF-8"?>\r\n<?maccor-application progid="Maccor Procedure File"?>',
            xml,
        )

        # newline at EOF
        return xml_with_maccor_processing_instruction + newLine

    @classmethod
    def _get_seqs(cls, shcedule_dict):
        techniques = shcedule_dict["Technique"]
        assert len(techniques.items()) == 1

        modulo_bat = techniques["1"]
        assert modulo_bat["Type"] == "Modulo Bat"

        # Steps are equivelant to seqs
        numbered_seqs = list(modulo_bat["Step"].items())

        # biologic parser may create empty seq(s) at the end
        # we intentionally _don't_ filter because if there is
        # a blank seq at in the middle
        seqs = list(map(lambda pair: pair[1], numbered_seqs))

        if len(seqs) > 0 and seqs[-1]["Ns"] == "":
            seqs = seqs[:-1]

        return seqs

    @classmethod
    def _stringify_step_num(cls, step_num):
        return "{0:0=3d}".format(step_num)

    # Maccor time can be implemented in
    # [hours]:[minutes]:[seconds]
    #
    # section 4.5.3.4.1 of the maccor manual on Sept, 20, 2020 specifies
    # any combination of numbers may be used as long as the colons
    # for hours minutes and seconds are used e.g.
    # ::.1 = 1 ms
    # ::3600 = 1h or 3600s
    # 05:42:21 = 5h, 42m, 21s
    #
    # no examples of decimals for minutes or hours are given, so that
    # may be a lie.
    #
    # no mention is made of a requirement for leading zeros.
    @staticmethod
    def _convert_time(str_val, unit, field, seq_num):
        if unit == "h":
            return "{}:00:00".format(str_val)
        elif unit == "mn":
            return "00:{}:00".format(str_val)
        elif unit == "s":
            return "00:00:{}".format(str_val)
        elif unit == "ms":
            milliseconds = float(str_val)
            if 0.1 <= milliseconds % 10 <= 9.9:
                raise Exception(
                    "Tried to convert time with specifity at the millisecond level,"
                    + "\nMaccor can only specify up to centiseconds."
                )
            seconds = milliseconds / 1000
            return "00:00:{:.2f}".format(seconds)
        else:
            raise Exception(
                "Unexpected time unit {} for {}, at seq {}".format(unit, field, seq_num)
            )

    # maccor measures only in amps
    @staticmethod
    def _convert_current(str_val, unit, field, seq_num):
        if unit == "A":
            return "{}".format(str_val)
        elif unit == "mA":
            return "{}E-3".format(str_val)
        elif unit in ("\N{Greek Small Letter Mu}A", "\N{Micro Sign}A"):
            return "{}E-6".format(str_val)
        elif unit == "nA":
            return "{}E-9".format(str_val)
        elif unit == "pA":
            return "{}E-12".format(str_val)
        else:
            raise Exception(
                "Unsupported current unit: {} for {}, at seq {}".format(
                    unit, field, seq_num
                )
            )

    @staticmethod
    def _convert_voltage(str_val, unit, field, seq_num):
        if unit == "V":
            return "{}".format(str_val)
        elif unit == "mV":
            return "{}E-3".format(str_val)
        else:
            raise Exception(
                "Unsupported voltage unit: {} for {}, at seq {}".format(
                    unit, field, seq_num
                )
            )

    @staticmethod
    def _convert_resistance(str_val, unit, field, seq_num):
        if unit == "MOhm":
            return "{}E6".format(str_val)
        elif unit == "kOhm":
            return "{}E3".format(str_val)
        elif unit == "Ohm":
            return "{}".format(str_val)
        elif unit == "mOhm":
            return "{}E-3".format(str_val)
        elif unit in ("\N{Greek Small Letter Mu}Ohm", "\N{Micro Sign}Ohm"):
            return "{}E-6".format(str_val)
        else:
            raise Exception(
                "Unsupported resistance unit: {} for {}, at seq {}".format(
                    unit, field, seq_num
                )
            )

    @staticmethod
    def _convert_power(str_val, unit, field, seq_num):
        if unit == "W":
            return "{}".format(str_val)
        elif unit == "mW":
            return "{}E-3".format(str_val)
        elif unit in ("\N{Greek Small Letter Mu}W", "\N{Micro Sign}W"):
            return "{}E-6".format(str_val)
        else:
            raise Exception(
                "Unsupported power unit: {} for {}, at seq {}".format(
                    unit, field, seq_num
                )
            )

    @classmethod
    def _create_loop_step(cls, seq, step_num_by_seq_num):
        step = deepcopy(_blank_step)
        # spacing may be important
        step["StepType"] = " Loop 1 "
        # unclear if spacing is necessary
        step["StepMode"] = "        "

        loop_end_entry = deepcopy(_blank_end_entry)

        loop_count = int(seq["ctrl_repeat"])
        loop_to_seq = int(seq["ctrl_seq"])
        # Maccor requires us to loop to the DO step
        # not the actual procedural step like in Biologic seqs
        loop_to_step = step_num_by_seq_num[loop_to_seq] - 1

        loop_end_entry["EndType"] = "Loop Cnt"
        loop_end_entry["Oper"] = " = "
        loop_end_entry["StepValue"] = "{}".format(loop_count)
        loop_end_entry["Step"] = cls._stringify_step_num(loop_to_step)
        assert len(loop_end_entry["Step"]) == 3

        step["Ends"] = OrderedDict({"EndEntry": loop_end_entry})

        return step

    @classmethod
    def _create_adv_cycle_step(cls):
        # spacing may be important
        step = deepcopy(_blank_step)
        # spacing may be important
        step["StepType"] = "AdvCycle"
        # unclear if spacing is necessary
        step["StepMode"] = "        "
        return step

    @classmethod
    def _create_do_step(cls):
        step = deepcopy(_blank_step)
        # spacing may be important
        step["StepType"] = "  Do 1  "
        # unclear if spacing is necessary
        step["StepMode"] = "        "
        return step

    @classmethod
    def _create_step(
        cls, seq, step_num_by_seq_num, seq_num_is_active_loop_start, end_step_num
    ):
        step = deepcopy(_blank_step)

        seq_num = int(seq["Ns"])
        seq_type = seq["ctrl_type"]
        seq_val = seq["ctrl1_val"]
        seq_unit = seq["ctrl1_val_unit"]
        seq_val_vs = seq["ctrl1_val_vs"]

        step_num = step_num_by_seq_num[seq_num]
        if not isinstance(step_num, int):
            raise Exception(
                "Internal error, could not find step num for seq: {}, this is a bug".format(
                    seq_num
                )
            )

        # all steps except AdvCycle, Do, and Loop seem to have these values
        step["Range"] = "4"
        step["Option1"] = "N"
        step["Option2"] = "N"
        step["Option3"] = "N"

        charge_or_discharge = seq["charge/discharge"]
        if charge_or_discharge != "Charge" and charge_or_discharge != "Discharge":
            raise Exception(
                "unsupported value in charge/discharge field: {}, in seq {}".format(
                    charge_or_discharge, seq_num
                )
            )
        # Dischrge mispelling intentional
        maccor_charge_or_discharge = (
            "Charge " if charge_or_discharge == " Charge " else "Dischrge "
        )

        if seq_type == "CC":
            step["StepMode"] = "Current "
            step["StepType"] = maccor_charge_or_discharge
            step["StepValue"] = cls._convert_current(
                seq_val, seq_unit, "ctrl1_val_vs", seq_num
            )
        elif seq_type == "CV" and seq_val_vs == "Ref":
            step["StepMode"] = "Voltage "
            step["StepType"] = maccor_charge_or_discharge
            step["StepValue"] = cls._convert_voltage(
                seq_val, seq_unit, "ctrl1_val_vs", seq_num
            )
        elif seq_type == "CV":
            raise Exception(
                "Unsupported constant voltage type, ctrl1_val_vs: {} at seq {}".format(
                    seq_val_vs, seq_num
                )
            )
        elif seq_type == "CR":
            step["StepMode"] = "Resistance "
            step["StepType"] = maccor_charge_or_discharge
            step["StepValue"] = cls._convert_resistance(
                seq_val, seq_unit, "ctrl1_val_vs", seq_num
            )
        elif seq_type == "CP":
            step["StepMode"] = "Power "
            step["StepType"] = maccor_charge_or_discharge
            step["StepValue"] = cls._convert_power(
                seq_val, seq_unit, "ctrl1_val_vs", seq_num
            )
        elif seq_type == "Rest":
            step["StepType"] = "  Rest  "
            step["StepMode"] = "        "
        else:
            raise Exception(
                "Unspported ctrl_type: {} at seq_num {}".format(seq_type, seq_num)
            )

        num_limits = int(seq["lim_nb"])
        if num_limits not in [0, 1, 2, 3]:
            raise Exception(
                "Unsupported number of limits, lim_nb: {} at seq {}".format(
                    num_limits, seq_num
                )
                + "\nvalue must in [0, 1, 2, 3]"
            )

        ends = []
        for lim_num in range(1, num_limits + 1):
            end_entry = deepcopy(_blank_end_entry)

            lim_type = seq["lim{}_type".format(lim_num)]
            lim_comp = seq["lim{}_comp".format(lim_num)]
            lim_unit = seq["lim{}_value_unit".format(lim_num)]
            lim_action = seq["lim{}_action".format(lim_num)]
            lim_val = float(seq["lim{}_value".format(lim_num)])
            goto_seq = int(seq["lim{}_seq".format(lim_num)])

            if lim_comp == "<":
                # trailing space intentional
                end_entry["Oper"] = "<= "
            elif lim_comp == ">" and lim_type == "Time":
                # trailing/leading space intentional
                end_entry["Oper"] = " = "
            elif lim_comp == ">":
                # trailing space intentional
                end_entry["Oper"] = ">= "
            else:
                raise Exception(
                    "Unsupported comparator, lim{}_comp: {} at seq {}".format(
                        lim_num, lim_comp, seq_num
                    )
                )

            if lim_type in ("Ece", "Ecell"):
                # trailing space intentional
                end_entry["EndType"] = "Voltage "
                end_entry["Value"] = cls._convert_voltage(
                    lim_val, lim_unit, "lim{}_value_unit".format(lim_num), seq_num
                )
            elif lim_type == "Time":
                # no space intentional
                end_entry["EndType"] = "StepTime"
                end_entry["Value"] = cls._convert_time(
                    lim_val, lim_unit, "lim{}_value_unit".format(lim_num), seq_num
                )
            elif lim_type == "I":
                # trailing space intentional
                end_entry["EndType"] = "Current "
                end_entry["Value"] = cls._convert_current(
                    lim_val, lim_unit, "lim{}_value_unit".format(lim_num), seq_num
                )
            else:
                raise Exception(
                    "Unsupported limit type, lim{}_type: {} at seq {}".format(
                        lim_num, lim_type, seq_num
                    )
                )

            if lim_action == "End":
                end_entry["Step"] = cls._stringify_step_num(end_step_num)
            elif lim_action == "Next sequence":
                next_step = step_num + 1
                end_entry["Step"] = cls._stringify_step_num(next_step)
            elif lim_action == "Goto sequence" and goto_seq not in step_num_by_seq_num:
                raise Exception(
                    "Could not convert goto at seq {}, either lim{}_seq: {}".format(
                        seq_num, lim_num, goto_seq
                    )
                    + "\n{} is not a valid goto seq, or this is a bug".format(goto_seq)
                )
            elif (
                lim_action == "Goto sequence"
                and goto_seq > seq_num
                and goto_seq in seq_num_is_active_loop_start
            ):
                # to go straight to a loop, must go to
                goto_step = step_num_by_seq_num[goto_seq] - 1
                end_entry["Step"] = cls._stringify_step_num(goto_step)
            elif lim_action == "Goto sequence":
                goto_step = step_num_by_seq_num[goto_seq]
                end_entry["Step"] = cls._stringify_step_num(goto_step)
            else:
                raise Exception(
                    "Unsupported goto construct, lim{}_action: {} at seq {}".format(
                        lim_num, lim_action, seq_num
                    )
                )

            ends.append(end_entry)

        assert len(ends) == num_limits
        step["Ends"] = (
            step["Ends"] if len(ends) == 0 else OrderedDict({"EndEntry": ends})
        )

        num_records = int(seq["rec_nb"])
        if num_records not in [0, 1, 2, 3]:
            raise Exception(
                "Unsupported number of records, rec_nb: {} value must in [0, 1, 2, 3]".format(
                    num_records
                )
            )

        reports = []
        for rec_num in range(1, num_records + 1):
            rec_type = seq["rec{}_type".format(rec_num)]
            rec_unit = seq["rec{}_value_unit".format(rec_num)]
            rec_val = float(seq["rec{}_value".format(rec_num)])

            report_entry = deepcopy(_blank_report_entry)

            if rec_type == "Time":
                # no space intentional
                report_entry["ReportType"] = "StepTime"
                report_entry["Value"] = cls._convert_time(
                    rec_val, rec_unit, "rec{}_value_unit".format(rec_num), seq_num
                )
            elif rec_type in ("Ece", "Ecell"):
                # trailing space intentional
                report_entry["ReportType"] = "Voltage "
                report_entry["Value"] = cls._convert_voltage(
                    rec_val, rec_unit, "rec{}_value_unit".format(rec_num), seq_num
                )
            elif rec_type == "I":
                # trailing space intentional
                report_entry["ReportType"] = "Current "
                report_entry["Value"] = cls._convert_current(
                    rec_val, rec_unit, "rec{}_value_unit".format(rec_num), seq_num
                )
            elif rec_type == "Power":
                # trailing space intentional
                report_entry["ReportType"] = "Power "
                report_entry["Value"] = cls._convert_power(
                    rec_val, rec_unit, "rec{}_value_unit".format(rec_num), seq_num
                )
            else:
                raise Exception(
                    "Unsuported record type, rec{}_type: {} at seq {}".format(
                        rec_num, rec_type, seq_num
                    )
                )

            reports.append(report_entry)

        assert len(reports) == num_records
        step["Reports"] = (
            step["Reports"]
            if len(reports) == 0
            else OrderedDict({"ReportEntry": reports})
        )

        return step

    @classmethod
    def _create_steps(cls, seqs):
        # DANGER
        #
        # This code has a lot of complexity and subtlties I couldn't figure out
        # how to remove
        #
        # all of the various passes over the list of sequences are _very_ tightly coupled

        # start build building up a mapping of where loops start and stop
        # since loops are defined at their end instead of beginning, we will
        # iterate from the right.
        #
        # during this process we want to assert loop overlap invariants

        containing_loop_range_by_seq_num = {}
        seq_num_is_active_loop_start = set()
        curr_loop_start = float("inf")
        curr_loop_end = float("inf")

        seqs_in_reverse_order = seqs.copy()
        list.reverse(seqs_in_reverse_order)
        for seq in seqs_in_reverse_order:
            is_loop_seq = seq["ctrl_type"] == "Loop"
            seq_num = int(seq["Ns"])
            loop_to = int(seq["ctrl_seq"])
            # Biologic will pass over this seq without looping
            # If cycle advancement is set to Loop, this seq will not cause the cycle number to advance
            loop_is_inactive = seq["ctrl_repeat"] == "0"

            # handle loops
            if is_loop_seq and not loop_is_inactive and loop_to >= curr_loop_start:
                # Biologic does not have nested loops, maccor does not allow
                # you to loop back into already completed loops
                # to handle this we're disallowing any overlapping loops
                # empty loops that do nothing are excepted
                raise Exception(
                    "Illegal loop at Seq {}. Loops are not allowed to".format(seq_num)
                    + "overlap and Seq {} overlaps with the loop from".format(seq_num)
                    + " seqs {}-{}".format(curr_loop_start, curr_loop_end)
                )
            elif is_loop_seq and not loop_is_inactive:
                curr_loop_end = seq_num
                curr_loop_start = loop_to
                seq_num_is_active_loop_start.update([loop_to])

            if seq_num >= curr_loop_start:
                containing_loop_range_by_seq_num.update(
                    {seq_num: (curr_loop_start, curr_loop_end)}
                )

        for loop_start in seq_num_is_active_loop_start:
            assert loop_start in containing_loop_range_by_seq_num
            assert loop_start == containing_loop_range_by_seq_num[loop_start][0]

        # once we know where loops start and end, we know where to include the
        # "Do" steps, as well as where to create "Adv cycle" steps constructed from
        # a blank loop followed immediately by a loop that goes back to the
        # blank loop exactly once, causing the maccor system to advance the cycle
        #
        # now we'll build up a mapping of seq numbers to step numbers to build out our
        # steps later
        curr_step = 1
        curr_loop_start = -1
        curr_loop_end = -1
        step_num_is_active_loop_start = set()
        seq_num_is_adv_cycle_step = set()
        seq_num_is_blank_loop = set()
        step_num_by_seq_num = {}
        for i, seq in enumerate(seqs):
            seq_num = int(seq["Ns"])
            loop_count = int(seq["ctrl_repeat"])
            loop_to = int(seq["ctrl_seq"])
            is_loop = seq["ctrl_type"] == "Loop"

            if is_loop and loop_count == 0:
                seq_num_is_blank_loop.update([seq_num])
                continue
            elif (
                is_loop
                and loop_to == seq_num - 1
                and (seq_num - 1) in seq_num_is_blank_loop
            ):
                seq_num_is_adv_cycle_step.update([seq_num])
                # add adv cycle step
                step_num_by_seq_num.update({seq_num: curr_step})
                curr_step += 1
                continue

            if seq_num in seq_num_is_active_loop_start:
                curr_loop_start, curr_loop_end = containing_loop_range_by_seq_num[
                    seq_num
                ]
                # add do step
                curr_step += 1
                step_num_is_active_loop_start.update([curr_step])
                seq_num_is_active_loop_start.update([seq_num])

            step_num_by_seq_num[seq_num] = curr_step
            if is_loop and loop_to == i - 1 and (i - 1) in seq_num_is_blank_loop:
                # will adv cycle
                curr_step += 1
            elif is_loop and seqs[loop_to]["ctrl_type"] == "Loop":
                raise Exception(
                    "unhandled looping construct at seq {}, cannot loop to empty loop".format(
                        seq_num
                    )
                    + "\nthat is not immediately followed by a cycle that loops back"
                    + "\nto it (used to force a cycle advance)"
                )
            elif is_loop:
                # will adv cycle and loop
                curr_step += 2
            else:
                # add normal step
                curr_step += 1

        end_step_num = curr_step

        # check for illegal GOTOs
        #
        # Maccor explicitly disallows us to enter a loop from outside a loop (other than to start one)
        # and going back before a loop end is similarly illegal
        steps = []
        last_loop_end = -1
        for seq in seqs:
            seq_num = int(seq["Ns"])
            num_limits = int(seq["lim_nb"])

            if seq_num in seq_num_is_blank_loop:
                continue
            if seq["ctrl_type"] == "Loop":
                last_loop_end = seq_num
                continue

            seq_belongs_to_loop = seq_num in containing_loop_range_by_seq_num

            for lim in range(1, num_limits + 1):
                goto = int(seq["lim{}_seq".format(lim)])
                if goto < last_loop_end:
                    raise Exception(
                        "Illegal GOTO at lim{}_seq, at seq {}\n".format(lim, seq_num)
                        + "\ncannot GOTO any seq that occurs before the end an earlier non-empty loop"
                        + "\nthere was a loop end at seq {}".format(last_loop_end)
                    )

                goto_belongs_to_loop = goto in containing_loop_range_by_seq_num
                goto_is_start_of_loop = (
                    goto == containing_loop_range_by_seq_num[goto][0]
                    if goto_belongs_to_loop
                    else False
                )

                if seq_belongs_to_loop and goto_belongs_to_loop:
                    goto_loop_start, goto_loop_end = containing_loop_range_by_seq_num[
                        goto
                    ]
                    seq_loop_start, seq_loop_end = containing_loop_range_by_seq_num[
                        seq_num
                    ]
                    if (
                        goto_loop_start != seq_loop_start
                        or goto_loop_end != seq_loop_end
                    ):
                        raise Exception(
                            "Illegal GOTO at lim{}_seq: {} at seq {}".format(
                                lim, goto, seq_num
                            )
                            + "\nas {} is contained in loop range {}-{}".format(
                                goto, goto_loop_start, goto_loop_end
                            )
                            + "\nMaccor only allows you to GOTO the interior of a loop from inside the same loop"
                            + "\notherwise you must goto the start of the loop, seq: {}".format(
                                goto_loop_start
                            )
                        )
                if (
                    not seq_belongs_to_loop
                    and goto_belongs_to_loop
                    and not goto_is_start_of_loop
                ):
                    goto_loop_start, goto_loop_end = containing_loop_range_by_seq_num[
                        goto
                    ]
                    raise Exception(
                        "Illegal goto at lim{}_seq: {} at seq {}".format(
                            lim, goto, seq_num
                        )
                        + "\nMaccor only allows you to GOTO the interior of a loop from inside the same loop"
                        + "\nthe loop. Seq {} is in the loop from {}-{} ".format(
                            goto, goto_loop_start, goto_loop_end
                        )
                        + "\nSeq{} is not contained in a loop".format(seq)
                        + "\nYou can only goto the start of the loop, seq {}.".format(
                            goto_loop_start
                        )
                    )

        # tightly coupled with the loop that constructs the seq num/step num mapping
        steps = []
        for i, seq in enumerate(seqs):
            seq_num = int(seq["Ns"])
            seq_is_loop = seq["ctrl_type"] == "Loop"
            loop_to = int(seq["ctrl_seq"])

            if seq_num in seq_num_is_blank_loop:
                continue

            if seq_num in seq_num_is_active_loop_start:
                do_step = cls._create_do_step()
                steps.append(do_step)

            if (
                seq_is_loop
                and loop_to == seq_num - 1
                and seq_num - 1 in seq_num_is_blank_loop
            ):
                adv_cycle_step = cls._create_adv_cycle_step()
                steps.append(adv_cycle_step)
            elif seq_is_loop:
                adv_cycle_step = cls._create_adv_cycle_step()
                steps.append(adv_cycle_step)

                loop_step = cls._create_loop_step(seq, step_num_by_seq_num)
                steps.append(loop_step)
            else:
                step = cls._create_step(
                    seq,
                    step_num_by_seq_num,
                    step_num_is_active_loop_start,
                    end_step_num,
                )
                steps.append(step)

        end_step = deepcopy(_blank_step)
        end_step["StepType"] = "  End   "
        end_step["StepMode"] = "        "
        steps.append(end_step)

        return steps
