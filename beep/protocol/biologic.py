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
Module for generating maccor procedure files from
input parameters and procedure templates
"""

import re
from collections import OrderedDict
from copy import deepcopy

from beep.utils import DashOrderedDict


class Settings(DashOrderedDict):
    """
    Settings file object. Provides factory methods
    to read a Biologic-type settings file and invoke
    from templates for specific experimental
     parameters

    """

    @classmethod
    def from_file(
        cls, filename, encoding="ISO-8859-1", column_width=20, step_entry_length=63
    ):
        """
        Settings file ingestion. Invokes Settings object
        from standard Biologic *.mps file.

        Args:
            filename (str): settings file.
            encoding (str): file encoding to use when reading the file
            column_width (int): number of characters per step column
            step_entry_length (int): This is the number of lines in the steps. It is based on the format seen
                in one file, and might need to be changed or dynamically determined upon file read

        Returns:
            (Settings): Ordered dictionary with keys corresponding to options or
                control variables. Section headers are nested dicts or lists
                within the dict.
        """
        with open(filename, "rb") as f:
            text = f.read()
            text = text.decode(encoding)

        return cls.mps_text_to_schedule_dict(text, column_width, step_entry_length)

    @classmethod
    def mps_text_to_schedule_dict(cls, text, column_width=20, step_entry_length=63):
        obj = cls()
        split_text = re.split(r"\r?\n|\r\n?", text)

        extra_lines = list(range(len(split_text)))
        technique_lines = [
            indx for indx, val in enumerate(split_text) if "Technique" in val
        ]
        technique_pos = []
        for technique_start_line in technique_lines:
            technique_num = split_text[technique_start_line].split(":")[-1].strip()
            start = (
                technique_start_line + 1
            )  # This +1 offset is based on the format seen in one file
            end = start + step_entry_length
            technique_pos.append((technique_num, start, end))
            lines_to_parse = [
                i for i in extra_lines if i not in list(range(start, end))
            ]

        section = "Metadata"
        metadata = []
        for line_num in lines_to_parse:
            if ":" in split_text[line_num]:
                if "Technique" in split_text[line_num]:
                    metadata.append(
                        [
                            "_".join(split_text[line_num].split(" : ", 1)),
                            split_text[line_num].split(":", 1)[-1].strip(),
                        ]
                    )
                else:
                    metadata.append(split_text[line_num].split(" : ", 1))
            elif split_text[line_num] == "":
                metadata.append(["line{}".format(line_num), "blank"])
            else:
                metadata.append([split_text[line_num], None])
        meta = OrderedDict(metadata)
        obj.set("{}".format(section), meta)

        section = "Technique"
        for technique in technique_pos:
            technique_num = technique[0]
            start = technique[1]
            end = technique[2]
            obj.set(
                "{}.{}.{}".format(section, technique_num, "Type"), split_text[start]
            )
            start = start + 1
            number_of_columns = (
                max([len(l) for l in split_text[start:end]]) // column_width
            )
            technique_steps = []
            for line in split_text[start:end]:
                steps_values = []
                for col in range(number_of_columns):
                    steps_values.append(
                        line[col * column_width: (col + 1) * column_width].strip()
                    )
                technique_steps.append(steps_values)
            step_matrix = list(zip(*technique_steps))
            step_headers = step_matrix[0]

            for step_number in range(1, number_of_columns):
                step = OrderedDict(zip(step_headers, step_matrix[step_number]))
                obj.set(
                    "{}.{}.{}.{}".format(
                        section, technique_num, "Step", str(step_number)
                    ),
                    step,
                )

        return obj

    def to_file(self, filename, encoding="ISO-8859-1", column_width=20, linesep="\r\n"):
        """
        Write DashOrderedDict to a settings file in the Biologic format with a *.mps extension.

        Args:
            filename (str): output name for settings file, full path
            encoding (str): file encoding to use when reading the file
            column_width (int): number of characters per step column
            linesep (str): characters to use for line separation, usually CRLF for Windows based machines

        Returns:
            (None): Writes out data to filename
        """
        data = deepcopy(self)
        blocks = []
        meta_data_keys = list(data["Metadata"].keys())
        for indx, meta_key in enumerate(meta_data_keys):
            if "Technique_" in meta_key:
                blocks.append(
                    " : ".join([meta_key.split("_")[0], data["Metadata"][meta_key]])
                )
                tq_number = data["Metadata"][meta_key]

                blocks.append(data["Technique"][tq_number]["Type"])
                technique_keys = list(data["Technique"][tq_number]["Step"]["1"].keys())
                for tq_key in technique_keys:
                    line = tq_key.ljust(column_width)
                    for step in data["Technique"][tq_number]["Step"].keys():
                        var = str(data["Technique"][tq_number]["Step"][step][tq_key])
                        line = line + var.ljust(column_width)
                    blocks.append(line)
                continue
            elif data["Metadata"][meta_key] is None:
                line = meta_key
            elif data["Metadata"][meta_key] == "blank":
                line = ""
            elif data["Metadata"][meta_key] is not None:
                line = " : ".join([meta_key, data["Metadata"][meta_key]])
            blocks.append(line)
            data.unset(data["Metadata"][meta_key])

        contents = linesep.join(blocks)

        with open(filename, "wb") as f:
            f.write(contents.encode(encoding))

    def formation_protocol_bcs(self, params):
        """
       Modify a settings file in the Biologic format with a *.mps extension.

        Args:
            params (pandas.DataFrame): Single row of the csv containing the parameters for the project

        Returns:
            (self): returns the parameterized protocol
        """
        assert self.get_path("Metadata.Cycle Definition") == "Charge/Discharge alternance"

        # Initial charge
        assert self.get_path("Technique.1.Step.2.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.2.ctrl1_val_unit") == "A"
        value = float(round(params["initial_charge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.2.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.2.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.2.lim1_value_unit") == "V"
        value = float(round(params["initial_charge_voltage_1"], 3))
        self.set("Technique.1.Step.2.lim1_value", value)
        assert self.get_path("Technique.1.Step.3.ctrl_type") == "CV"
        assert self.get_path("Technique.1.Step.3.ctrl1_val_unit") == "V"
        self.set("Technique.1.Step.3.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.3.lim1_type") == "Time"
        assert self.get_path("Technique.1.Step.3.lim1_value_unit") == "mn"
        value = float(round(params["initial_charge_cvhold_1"], 3))
        self.set("Technique.1.Step.3.lim1_value", value)

        assert self.get_path("Technique.1.Step.4.lim1_type") == "Time"
        assert self.get_path("Technique.1.Step.4.lim1_value_unit") == "mn"
        value = float(round(params["initial_charge_rest_1"], 3))
        self.set("Technique.1.Step.4.lim1_value", value)

        assert self.get_path("Technique.1.Step.5.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.5.ctrl1_val_unit") == "A"
        value = float(round(params["initial_charge_current_2"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.5.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.5.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.5.lim1_value_unit") == "V"
        value = float(round(params["initial_charge_voltage_2"], 3))
        self.set("Technique.1.Step.5.lim1_value", value)
        assert self.get_path("Technique.1.Step.6.ctrl_type") == "CV"
        assert self.get_path("Technique.1.Step.6.ctrl1_val_unit") == "V"
        self.set("Technique.1.Step.6.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.6.lim1_type") == "Time"
        assert self.get_path("Technique.1.Step.6.lim1_value_unit") == "mn"
        value = float(round(params["initial_charge_cvhold_2"], 3))
        self.set("Technique.1.Step.6.lim1_value", value)

        assert self.get_path("Technique.1.Step.7.lim1_type") == "Time"
        assert self.get_path("Technique.1.Step.7.lim1_value_unit") == "mn"
        value = float(round(params["initial_charge_rest_2"], 3))
        self.set("Technique.1.Step.7.lim1_value", value)

        # Initial discharge
        assert self.get_path("Technique.1.Step.8.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.8.ctrl1_val_unit") == "A"
        value = float(round(params["initial_discharge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.8.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.8.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.8.lim1_value_unit") == "V"
        value = float(round(params["initial_discharge_voltage_1"], 3))
        self.set("Technique.1.Step.8.lim1_value", value)

        # Mid cycle
        assert self.get_path("Technique.1.Step.9.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.9.ctrl1_val_unit") == "A"
        value = float(round(params["mid_charge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.9.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.9.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.9.lim1_value_unit") == "V"
        value = float(round(params["mid_charge_voltage_1"], 3))
        self.set("Technique.1.Step.9.lim1_value", value)

        assert self.get_path("Technique.1.Step.10.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.10.ctrl1_val_unit") == "A"
        value = float(round(params["mid_discharge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.10.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.10.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.10.lim1_value_unit") == "V"
        value = float(round(params["mid_discharge_voltage_1"], 3))
        self.set("Technique.1.Step.10.lim1_value", value)

        assert self.get_path("Technique.1.Step.11.ctrl_type") == "Loop"
        assert self.get_path("Technique.1.Step.11.ctrl_seq") == "8"
        value = int(params["mid_cycle_reps"] - 1)
        self.set("Technique.1.Step.11.ctrl_repeat", value)

        # Final cycle
        assert self.get_path("Technique.1.Step.12.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.12.ctrl1_val_unit") == "A"
        value = float(round(params["final_charge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.12.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.12.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.12.lim1_value_unit") == "V"
        value = float(round(params["final_charge_voltage_1"], 3))
        self.set("Technique.1.Step.12.lim1_value", value)

        assert self.get_path("Technique.1.Step.13.ctrl_type") == "CC"
        assert self.get_path("Technique.1.Step.13.ctrl1_val_unit") == "A"
        value = float(round(params["final_discharge_current_1"] * params['capacity_nominal'], 3))
        self.set("Technique.1.Step.13.ctrl1_val", value)

        assert self.get_path("Technique.1.Step.13.lim1_type") == "Ecell"
        assert self.get_path("Technique.1.Step.13.lim1_value_unit") == "V"
        value = float(round(params["final_discharge_voltage_1"], 3))
        self.set("Technique.1.Step.13.lim1_value", value)
        assert self.get_path("Technique.1.Step.14.ctrl_type") == "CV"
        assert self.get_path("Technique.1.Step.14.ctrl1_val_unit") == "V"
        self.set("Technique.1.Step.14.ctrl1_val", value)
        assert self.get_path("Technique.1.Step.14.lim1_type") == "Time"
        assert self.get_path("Technique.1.Step.14.lim1_value_unit") == "mn"
        value = float(round(params["final_discharge_cvhold_1"], 3))
        self.set("Technique.1.Step.14.lim1_value", value)

        return self
