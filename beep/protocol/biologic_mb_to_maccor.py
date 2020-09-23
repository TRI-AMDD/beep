# Copyright 2020 Toyota Research Institute. All rights reserved.
""" Parsing and conversion of biologic Modulo Bat files to Maccor Procedure Files"""
import os
from copy import deepcopy
from collections import OrderedDict
from monty.serialization import loadfn
from beep.protocol import PROCEDURE_TEMPLATE_DIR


MACCOR_TEMPLATE = loadfn(os.path.join(PROCEDURE_TEMPLATE_DIR, "maccor_schemas.yaml"))

_default_maccor_header = OrderedDict(MACCOR_TEMPLATE["default_maccor_header"])
_blank_step = OrderedDict(MACCOR_TEMPLATE["blank_step"])
_blank_end_entry = OrderedDict(MACCOR_TEMPLATE["blank_end_entry"])
_blank_report_entry = OrderedDict(MACCOR_TEMPLATE["blank_report_entry"])
_blank_maccor_body = OrderedDict(MACCOR_TEMPLATE["blank_maccor_body"])


class BiologicMbToMaccorProcedure:
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
    def _create_loop_step(cls, seq, seq_num_by_step_num):
        step = deepcopy(_blank_step)
        # spacing may be important
        step["StepType"] = " Loop 1 "
        # unclear if spacing is necessary
        step["StepMode"] = "        "

        loop_end_entry = deepcopy(_blank_end_entry)

        loop_count = int(seq["ctrl_repeat"])
        loop_to_seq = int(seq["ctrl_seq"])
        loop_to_step = seq_num_by_step_num[loop_to_seq]

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
        cls, seq, step_num_by_seq_num, seq_num_is_non_empty_loop_start, end_step_num
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
                and goto_seq in seq_num_is_non_empty_loop_start
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
