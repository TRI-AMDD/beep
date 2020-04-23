# Copyright 2019 Toyota Research Institute. All rights reserved.
"""
Module and script for generating protocol files from
input parameters and procedure templates

Usage:
    generate_protocol [INPUT_JSON]

Options:
    -h --help        Show this screen
    --version        Show version


The `generate_protocol` script will generate a protocol file from input
parameters defined in the rows of a CSV-formatted input file.
It stores its outputs in `/data-share/protocols/`|
                                                |-`/procedures/`
                                                |-`/schedules/`
                                                |-`/names/`
For Maccor procedures the output procedures will be stored in `/data-share/protocols/procedures/`
For Arbin schedules the schedules will be stored in `/data-share/protocols/schedules/`
Additionally a file containing the names for the test files generated during the function call
will be stored in `/data-share/protocols/names/` with a date and time in the name to differentiate it
This file is to facilitate the process of starting tests on the cycling machines, by making it easier to
enter data into the appropriate fields.

The input json must contain the following fields
* `file_list` - filenames/paths corresponding to the input csv files

The output json will contain the following fields
* `file_list` - list of protocol files corresponding to the input files

Example:
$ generate_protocol '{"file_list": ["/data-share/raw/parameters/procedure_params_000112e3.csv"]}'
{
    "file_list": ["/data-share/protocols/procedures/name_1.000", "/data-share/protocols/procedures/name_2.000"]
}
"""

import os
import warnings
import json
import time
import datetime
import csv
from copy import deepcopy

import pandas as pd
import xmltodict
from docopt import docopt
from monty.serialization import loadfn
from beep import logger, __version__, PROCEDURE_TEMPLATE_DIR
from beep.utils import KinesisEvents, DashOrderedDict
s = {'service': 'ProtocolGenerator'}


class Procedure(DashOrderedDict):
    """
    Procedure file object. Provides factory methods
    to read a Maccor-type procedure file and invoke
    from templates for specific experimental
    procedure parameters

    """
    @classmethod
    def from_file(cls, filename, encoding='UTF-8'):
        """
        Procedure file ingestion. Invokes Procedure object
        from standard Maccor xml file.

        Args:
            filename (str): xml procedure file.

        Returns:
            (Procedure): Ordered dictionary with keys corresponding to options or
                control variables. Section headers are nested dicts or lists
                within the dict.
        """
        with open(filename, 'rb') as f:
            text = f.read().decode(encoding)
        data = xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)
        return cls(data)

    # TODO: check on the necessity of this with MACCOR instrument
    def _format_maccor(self):
        """
        Dictionary reformatting of the entries in the procedure in
        order to match the maccor formats. Mainly re-adding whitespace
        to entries that were stripped on injestion.

        Returns:
            dict: Ordered dictionary with reformatted entries to match the
                formatting used in the maccor procedure files.
        """
        formatted = deepcopy(self)
        for step in formatted['MaccorTestProcedure']['ProcSteps']['TestStep']:
            # print(json.dumps(step['StepType'], indent=2))
            while len(step['StepType']) < 8:
                step['StepType'] = step['StepType'].center(8)
            if step['StepMode'] is None:
                step['StepMode'] = " "
            while len(step['StepMode']) < 8:
                step['StepMode'] = step['StepMode'].center(8)
            if step['Ends'] is not None:
                # If the Ends Element is a list we need to
                # check each entry in the list
                if isinstance(step['Ends']['EndEntry'], list):
                    # print(json.dumps(step['Ends'], indent=2))
                    for end_entry in step['Ends']['EndEntry']:
                        self.ends_whitespace(end_entry)
                if isinstance(step['Ends']['EndEntry'], dict):
                    self.ends_whitespace(step['Ends']['EndEntry'])
            if step['Reports'] is not None:
                if isinstance(step['Reports']['ReportEntry'], list):
                    for rep_entry in step['Reports']['ReportEntry']:
                        self.reports_whitespace(rep_entry)
                if isinstance(step['Reports']['ReportEntry'], dict):
                    self.reports_whitespace(step['Reports']['ReportEntry'])

        return formatted

    @staticmethod
    def ends_whitespace(end_entry):
        if end_entry['SpecialType'] is None:
            end_entry['SpecialType'] = " "
        while len(end_entry['EndType']) < 8:
            end_entry['EndType'] = end_entry['EndType'].center(8)
        if end_entry['Oper'] is not None:
            if len(end_entry['Oper']) < 2:
                end_entry['Oper'] = end_entry['Oper'].center(3)
            else:
                end_entry['Oper'] = end_entry['Oper'].ljust(3)

    @staticmethod
    def reports_whitespace(rep_entry):
        while len(rep_entry['ReportType']) < 8:
            rep_entry['ReportType'] = rep_entry['ReportType'].center(8)

    def to_file(self, filename, encoding='UTF-8'):
        """
        Writes object to maccor-formatted xml file using xmltodict
        unparse function.

        encoding (str): text encoding of output file

        Args:
            filename (str):file name to save xml to.
        """
        formatted = self._format_maccor()
        contents = xmltodict.unparse(
            input_dict=formatted,
            output=None,
            encoding=encoding,
            short_empty_elements=False,
            pretty=True,
            newl="\n",
            indent="  ")

        # Manually inject processing instructions on line 2
        line0, remainder = contents.split('\n', 1)
        line1 = "<?maccor-application progid=\"Maccor Procedure File\"?>"
        contents = "\n".join([line0, line1, remainder])
        contents = self._fixup_empty_elements(contents)
        contents += "\n"
        with open(filename, 'w') as f:
            f.write(contents)

    @staticmethod
    def _fixup_empty_elements(text):
        """
        xml reformatting to match the empty elements that are used
        in the maccor procedure format. Writes directly back to the file
        and assumes that the empty elements to be replaced are all on a
        single line.

        Args:
            text (str): xml file raw text to be formatted

        """
        text = text.replace(r"<Limits></Limits>", "<Limits/>")
        text = text.replace(r"<Reports></Reports>", "<Reports/>")
        text = text.replace(r"<Ends></Ends>", "<Ends/>")
        return text

    def modify_step_value(self, step_num, step_type, step_value):
        """
        Modifies the procedure parameters to set a step value at at given step num and type.

        Args:
            step_num (int): step id at which to set value
            step_type (str): step type at which to set value
            step_value (str): value to set

        Returns:
            dict: modified proc_dict with set value
        """
        for step_idx, step in enumerate(self['MaccorTestProcedure']['ProcSteps']['TestStep']):
            if step_idx == step_num and step['StepType'] == step_type:
                step['StepValue'] = step_value
        return self

    @classmethod
    def from_template(cls, template_name):
        """
        Utility method to load procedures from
        beep template directory

        Args:
            template_name (str): template name, e. g.
                'EXP', 'diagnosticV2'

        Returns:
            (Procedure): procedure loaded from template

        """
        template_filename = "{}.000".format(template_name)
        template_filename = os.path.join(PROCEDURE_TEMPLATE_DIR, template_filename)
        return cls.from_file(template_filename)

    @classmethod
    def from_exp(cls, cutoff_voltage, charge_rate, discharge_rate):
        """
        Generates a procedure according to the EXP-style template.

        Args:
            cutoff_voltage (float): cutoff voltage for.
            charge_rate (float): charging C-rate in 1/h.
            discharge_rate (float): discharging C-rate in 1/h.

        Returns:
            dict: dictionary of procedure parameters.

        """
        # Load EXP template
        obj = cls.from_template("EXP")

        # Modify according to params
        loop_idx_start, loop_idx_end = None, None
        for step_idx, step in enumerate(obj['MaccorTestProcedure']['ProcSteps']['TestStep']):
            if step['StepType'] == "Do 1":
                loop_idx_start = step_idx
            if step['StepType'] == "Loop 1":
                loop_idx_end = step_idx

        if loop_idx_start is None or loop_idx_end is None:
            raise UnboundLocalError("Loop index is not set")

        for step_idx, step in enumerate(obj['MaccorTestProcedure']['ProcSteps']['TestStep']):
            if step['StepType'] == 'Charge':
                if step['Limits'] is not None and 'Voltage' in step['Limits']:
                    step['Limits']['Voltage'] = cutoff_voltage
                if step['StepMode'] == 'Current' and loop_idx_start < step_idx < loop_idx_end:
                    step['StepValue'] = charge_rate
            if step['StepType'] == "Dischrge" and step['StepMode'] == 'Current' \
                    and loop_idx_start < step_idx < loop_idx_end:
                step['StepValue'] = discharge_rate

        return obj

    # TODO: rename this diagnosticv2?
    @classmethod
    def from_regcyclev2(cls, reg_param):
        """
        Generates a procedure according to the diagnosticV2 template.

        Args:
            reg_param (pandas.Dataframe): containing the following quantities
                charge_constant_current_1 (float): C
                charge_percent_limit_1 (float): % of nominal capacity
                charge_constant_current_2 (float): C
                charge_cutoff_voltage (float): V
                charge_constant_voltage_time (integer): mins
                charge_rest_time (integer): mins
                discharge_constant_current (float): C
                discharge_cutoff_voltage (float): V
                discharge_rest_time (integer): mins
                cell_temperature_nominal (float): ˚C
                capacity_nominal (float): Ah
                diagnostic_start_cycle (integer): cycles
                diagnostic_interval (integer): cycles

        Returns:
            dict: dictionary of procedure parameters.
        """

        assert reg_param['charge_cutoff_voltage'] > reg_param['discharge_cutoff_voltage']

        dc_idx = 1

        # Load template
        obj = cls.from_template("diagnosticV2")
        obj.insert_resistance_regcyclev2(dc_idx, reg_param)

        # Start of initial set of regular cycles
        reg_charge_idx = 27 + 1
        obj.insert_charge_regcyclev2(reg_charge_idx, reg_param)
        reg_discharge_idx = 27 + 5
        obj.insert_discharge_regcyclev2(reg_discharge_idx, reg_param)

        # Start of main loop of regular cycles
        reg_charge_idx = 59 + 1
        obj.insert_charge_regcyclev2(reg_charge_idx, reg_param)
        reg_discharge_idx = 59 + 5
        obj.insert_discharge_regcyclev2(reg_discharge_idx, reg_param)

        # Storage cycle
        reg_storage_idx = 69
        obj.insert_storage_regcyclev2(reg_storage_idx, reg_param)

        return obj

    def insert_maccor_waveform_discharge(self, waveform_idx, waveform_filename):
        """
        Inserts a waveform into procedure dictionary at given id.

        Args:
            waveform_idx (int): Step in the procedure file to
                insert waveform at
            waveform_filename (str): Path to .MWF waveform file.
                Waveform needs to be pre-scaled for current/power
                capabilities of the cell and cycler

        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']
        assert steps[waveform_idx]['StepType'] == "FastWave"

        steps[waveform_idx]['StepValue'] = waveform_filename.split('.MWF')[0]

        return self

    def insert_resistance_regcyclev2(self, resist_idx, reg_param):
        """
        Inserts resistance into procedure dictionary at given id.

        Args:
            resist_idx (int):
            reg_param (pandas.DataFrame):

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # Initial resistance check
        assert steps[resist_idx]['StepType'] == "Charge"
        assert steps[resist_idx]['StepMode'] == "Current"
        steps[resist_idx]['StepValue'] = float(round(1.0 * reg_param['capacity_nominal'], 3))

        return self

    def insert_charge_regcyclev2(self, charge_idx, reg_param):
        """
        Inserts charge into procedure dictionary at given id.

        Args:
            charge_idx (int)
            reg_param (pandas.DataFrame):

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # Regular cycle constant current charge part 1
        step_idx = charge_idx
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(reg_param['charge_constant_current_1']
                                                   * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(3600 * (reg_param['charge_percent_limit_1'] / 100)
                           / reg_param['charge_constant_current_1']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime(
            '%H:%M:%S', time.gmtime(time_s))

        # Regular cycle constant current charge part 2
        step_idx = charge_idx + 1
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(reg_param['charge_constant_current_2']
                                                   * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "Voltage"
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = float(round(reg_param['charge_cutoff_voltage'], 3))

        # Regular cycle constant voltage hold
        step_idx = charge_idx + 2
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Voltage"
        steps[step_idx]['StepValue'] = reg_param['charge_cutoff_voltage']
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * reg_param['charge_constant_voltage_time']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        # Regular cycle rest at top of charge
        step_idx = charge_idx + 3
        assert steps[step_idx]['StepType'] == "Rest"
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * reg_param['charge_rest_time']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        return self

    def insert_discharge_regcyclev2(self, discharge_idx, reg_param):
        """
        Inserts discharge into procedure dictionary at given id.

        Args:
            discharge_idx (int):
            reg_param (pandas.DataFrame):

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # Regular cycle constant current discharge part 1
        step_idx = discharge_idx
        assert steps[step_idx]['StepType'] == "Dischrge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(reg_param['discharge_constant_current']
                                                   * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "Voltage"
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = float(round(reg_param['discharge_cutoff_voltage'], 3))

        # Regular cycle rest after discharge
        step_idx = discharge_idx + 1
        assert steps[step_idx]['StepType'] == "Rest"
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * reg_param['discharge_rest_time']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        # Regular cycle number of times to repeat regular cycle for initial offset and main body
        step_idx = discharge_idx + 3
        assert steps[step_idx]['StepType'][0:4] == "Loop"
        if steps[step_idx]['StepType'] == "Loop 1":
            assert steps[step_idx]['Ends']['EndEntry']['EndType'] == "Loop Cnt"
            steps[step_idx]['Ends']['EndEntry']['Value'] = reg_param['diagnostic_start_cycle']
        elif steps[step_idx]['StepType'] == "Loop 2":
            assert steps[step_idx]['Ends']['EndEntry']['EndType'] == "Loop Cnt"
            steps[step_idx]['Ends']['EndEntry']['Value'] = reg_param['diagnostic_interval']

        return self

    def insert_storage_regcyclev2(self, storage_idx, reg_param):
        """
        Inserts storage into procedure dictionary at given id.

        Args:
            storage_idx (int):
            reg_param (pandas.DataFrame):

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # Storage condition
        step_idx = storage_idx
        assert steps[step_idx]['StepType'] == "Dischrge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(0.5 * reg_param['capacity_nominal'], 3))
        steps[step_idx]['Limits']['Voltage'] = float(round(reg_param['discharge_cutoff_voltage'], 3))
        assert steps[step_idx]['Ends']['EndEntry']['EndType'] == "Current"
        steps[step_idx]['Ends']['EndEntry']['Value'] = float(round(0.05 * reg_param['capacity_nominal'], 3))
        step_idx = storage_idx + 1
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(0.5 * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * 12))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        return self

    @classmethod
    def generate_procedure_regcyclev3(cls, protocol_index, reg_param):
        """
        Generates a procedure according to the diagnosticV3 template.

        Args:
            protocol_index (int): number of the protocol file being generated from this file.
            reg_param (pandas.DataFrame): containing the following quantities
                charge_constant_current_1 (float): C
                charge_percent_limit_1 (float): % of nominal capacity
                charge_constant_current_2 (float): C
                charge_cutoff_voltage (float): V
                charge_constant_voltage_time (integer): mins
                charge_rest_time (integer): mins
                discharge_constant_current (float): C
                discharge_cutoff_voltage (float): V
                discharge_rest_time (integer): mins
                cell_temperature_nominal (float): ˚C
                capacity_nominal (float): Ah
                diagnostic_start_cycle (integer): cycles
                diagnostic_interval (integer): cycles

        Returns:
            (Procedure): dictionary invoked using template/parameters
        """
        assert reg_param['charge_cutoff_voltage'] > reg_param['discharge_cutoff_voltage']
        assert reg_param['charge_constant_current_1'] <= reg_param['charge_constant_current_2']

        rest_idx = 0

        obj = cls.from_template("diagnosticV3")
        obj.insert_initialrest_regcyclev3(rest_idx, protocol_index)

        dc_idx = 1
        obj.insert_resistance_regcyclev2(dc_idx, reg_param)

        # Start of initial set of regular cycles
        reg_charge_idx = 27 + 1
        obj.insert_charge_regcyclev3(reg_charge_idx, reg_param)
        reg_discharge_idx = 27 + 5
        obj.insert_discharge_regcyclev2(reg_discharge_idx, reg_param)

        # Start of main loop of regular cycles
        reg_charge_idx = 59 + 1
        obj.insert_charge_regcyclev3(reg_charge_idx, reg_param)
        reg_discharge_idx = 59 + 5
        obj.insert_discharge_regcyclev2(reg_discharge_idx, reg_param)

        # Storage cycle
        reg_storage_idx = 93
        obj.insert_storage_regcyclev3(reg_storage_idx, reg_param)

        # Check that the upper charge voltage is set the same for both charge current steps
        reg_charge_idx = 27 + 1
        assert obj['MaccorTestProcedure']['ProcSteps']['TestStep'][reg_charge_idx][
                   'Ends']['EndEntry'][1]['Value'] \
            == obj['MaccorTestProcedure']['ProcSteps']['TestStep'][reg_charge_idx + 1
               ]['Ends']['EndEntry'][0]['Value']

        reg_charge_idx = 59 + 1
        assert obj['MaccorTestProcedure']['ProcSteps']['TestStep'][reg_charge_idx
               ]['Ends']['EndEntry'][1]['Value'] \
               == obj['MaccorTestProcedure']['ProcSteps']['TestStep'][reg_charge_idx + 1
                                                                       ]['Ends']['EndEntry'][0]['Value']

        return obj

    def insert_initialrest_regcyclev3(self, rest_idx, index):
        """
        Inserts initial rest into procedure dictionary at given id.

        Args:
            rest_idx (int):
            index (int):

        Returns:
            dict:

        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']
        # Initial rest
        offset_seconds = 720
        assert steps[rest_idx]['StepType'] == "Rest"
        assert steps[rest_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(3 * 3600 + offset_seconds * (index % 96)))
        steps[rest_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        return self

    def insert_charge_regcyclev3(self, charge_idx, reg_param):
        """
        Inserts charge into procedure dictionary at given id.

        Args:
            charge_idx (int):
            reg_param (pandas.DataFrame):

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # Regular cycle constant current charge part 1
        step_idx = charge_idx
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(reg_param['charge_constant_current_1']
                                                     * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(3600 * (reg_param['charge_percent_limit_1'] / 100)
                           / reg_param['charge_constant_current_1']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))
        assert steps[step_idx]['Ends']['EndEntry'][1]['Value'] != 4.4
        steps[step_idx]['Ends']['EndEntry'][1]['Value'] = float(round(reg_param['charge_cutoff_voltage'], 3))

        # Regular cycle constant current charge part 2
        step_idx = charge_idx + 1
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(reg_param['charge_constant_current_2']
                                                         * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "Voltage"
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = float(round(reg_param['charge_cutoff_voltage'], 3))

        # Regular cycle constant voltage hold
        step_idx = charge_idx + 2
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Voltage"
        steps[step_idx]['StepValue'] = reg_param['charge_cutoff_voltage']
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * reg_param['charge_constant_voltage_time']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        # Regular cycle rest at top of charge
        step_idx = charge_idx + 3
        assert steps[step_idx]['StepType'] == "Rest"
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * reg_param['charge_rest_time']))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        return self

    def insert_storage_regcyclev3(self, storage_idx, reg_param):
        """
        Inserts storage into procedure dictionary at given id.

        Args:
            self (dict):
            storage_idx (int):
            reg_param (pandas.DataFrame):

        Returns:
            dict:

        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']
        # Storage condition
        step_idx = storage_idx
        assert steps[step_idx]['StepType'] == "Dischrge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(0.5 * reg_param['capacity_nominal'], 3))
        steps[step_idx]['Limits']['Voltage'] = float(round(reg_param['discharge_cutoff_voltage'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "Current"
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = float(round(0.05 * reg_param['capacity_nominal'], 3))
        step_idx = storage_idx + 1
        assert steps[step_idx]['StepType'] == "Charge"
        assert steps[step_idx]['StepMode'] == "Current"
        steps[step_idx]['StepValue'] = float(round(0.5 * reg_param['capacity_nominal'], 3))
        assert steps[step_idx]['Ends']['EndEntry'][0]['EndType'] == "StepTime"
        time_s = int(round(60 * 12))
        steps[step_idx]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        return self

    def generate_procedure_diagcyclev2(self, nominal_capacity, diagnostic_params):
        """
        Generates a procedure according to the diagnosticV2 template.

        Args:
            nominal_capacity (float): Standard capacity for this cell.
            diagnostic_params (pandas.Series): Series containing all of the
                diagnostic parameters.

        Returns:
            (dict) dictionary of procedure parameters.

        """
        start_reset_cycle_1 = 4
        self.insert_reset_cyclev2(start_reset_cycle_1, nominal_capacity, diagnostic_params)
        start_hppc_cycle_1 = 8
        self.insert_hppc_cyclev2(start_hppc_cycle_1, nominal_capacity, diagnostic_params)
        start_rpt_cycle_1 = 18
        self.insert_rpt_cyclev2(start_rpt_cycle_1, nominal_capacity, diagnostic_params)

        start_reset_cycle_2 = 37
        self.insert_reset_cyclev2(start_reset_cycle_2, nominal_capacity, diagnostic_params)
        start_hppc_cycle_2 = 40
        self.insert_hppc_cyclev2(start_hppc_cycle_2, nominal_capacity, diagnostic_params)
        start_rpt_cycle_2 = 50
        self.insert_rpt_cyclev2(start_rpt_cycle_2, nominal_capacity, diagnostic_params)

        return self

    @classmethod
    def generate_procedure_diagcyclev3(cls, nominal_capacity, diagnostic_params):
        """
        Generates a procedure according to the diagnosticV3 template.

        Args:
            nominal_capacity (float): Standard capacity for this cell.
            diagnostic_params (pandas.Series): Series containing all of the
                diagnostic parameters.

        Returns:
            dict: dictionary of procedure parameters.

        """
        obj = cls.from_template("diagnosticV3")
        start_reset_cycle_1 = 4
        obj.insert_reset_cyclev2(start_reset_cycle_1, nominal_capacity, diagnostic_params)
        start_hppc_cycle_1 = 8
        obj.insert_hppc_cyclev2(start_hppc_cycle_1, nominal_capacity, diagnostic_params)
        start_rpt_cycle_1 = 18
        obj.insert_rpt_cyclev2(start_rpt_cycle_1, nominal_capacity, diagnostic_params)

        start_reset_cycle_2 = 37
        obj.insert_reset_cyclev2(start_reset_cycle_2, nominal_capacity, diagnostic_params)
        start_hppc_cycle_2 = 40
        obj.insert_hppc_cyclev2(start_hppc_cycle_2, nominal_capacity, diagnostic_params)
        start_rpt_cycle_2 = 50
        obj.insert_rpt_cyclev2(start_rpt_cycle_2, nominal_capacity, diagnostic_params)

        start_reset_cycle_3 = 70
        obj.insert_reset_cyclev2(start_reset_cycle_3, nominal_capacity, diagnostic_params)
        start_hppc_cycle_3 = 74
        obj.insert_hppc_cyclev2(start_hppc_cycle_3, nominal_capacity, diagnostic_params)
        start_rpt_cycle_3 = 84
        obj.insert_rpt_cyclev2(start_rpt_cycle_3, nominal_capacity, diagnostic_params)

        return obj

    def insert_reset_cyclev2(self, start, nominal_capacity, diagnostic_params):
        """
        Helper function for parameterizing the reset cycle.

        Args:
            start:
            nominal_capacity:
            diagnostic_params:

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']
        # Charge step for reset cycle
        assert steps[start]['StepType'] == 'Charge'
        assert steps[start]['StepMode'] == 'Current'
        steps[start]['StepValue'] = float(round(nominal_capacity * diagnostic_params['reset_cycle_current'], 4))
        steps[start]['Limits']['Voltage'] = float(round(diagnostic_params['diagnostic_charge_cutoff_voltage'], 3))
        assert steps[start]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                                   diagnostic_params['reset_cycle_cutoff_current'], 4))

        # Discharge step for reset cycle
        assert steps[start+1]['StepType'] == 'Dischrge'
        assert steps[start+1]['StepMode'] == 'Current'
        steps[start+1]['StepValue'] = float(round(nominal_capacity * diagnostic_params['reset_cycle_current'], 4))
        assert steps[start+1]['Ends']['EndEntry'][0]['EndType'] == 'Voltage'
        steps[start+1]['Ends']['EndEntry'][0]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        return self

    def insert_hppc_cyclev2(self, start, nominal_capacity, diagnostic_params):
        """
        Helper function for parameterizing the hybrid pulse power cycle

        Args:
            start:
            nominal_capacity:
            diagnostic_params:

        Returns:
            dict:
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']
        # Initial charge step for hppc cycle
        assert steps[start]['StepType'] == 'Charge'
        assert steps[start]['StepMode'] == 'Current'
        steps[start]['StepValue'] = float(round(nominal_capacity *
                                                diagnostic_params['HPPC_baseline_constant_current'], 3))
        steps[start]['Limits']['Voltage'] = float(round(diagnostic_params['diagnostic_charge_cutoff_voltage'], 3))
        assert steps[start]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                                   diagnostic_params['diagnostic_cutoff_current'], 3))

        # Rest step for hppc cycle
        assert steps[start+2]['StepType'] == 'Rest'
        assert steps[start+2]['Ends']['EndEntry'][0]['EndType'] == 'StepTime'
        time_s = int(round(60 * diagnostic_params['HPPC_rest_time']))
        steps[start+2]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        # Discharge step 1 for hppc cycle
        assert steps[start+3]['StepType'] == 'Dischrge'
        assert steps[start+3]['StepMode'] == 'Current'
        steps[start+3]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['HPPC_pulse_current_1'], 3))
        assert steps[start+3]['Ends']['EndEntry'][0]['EndType'] == 'StepTime'
        time_s = diagnostic_params['HPPC_pulse_duration_1']
        steps[start+3]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))
        assert steps[start+3]['Ends']['EndEntry'][1]['EndType'] == 'Voltage'
        steps[start+3]['Ends']['EndEntry'][1]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        # Pulse rest step for hppc cycle
        assert steps[start+4]['StepType'] == 'Rest'
        assert steps[start+4]['Ends']['EndEntry'][0]['EndType'] == 'StepTime'
        time_s = int(round(diagnostic_params['HPPC_pulse_rest_time']))
        steps[start+4]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))

        # Charge step 1 for hppc cycle
        assert steps[start+5]['StepType'] == 'Charge'
        assert steps[start+5]['StepMode'] == 'Current'
        steps[start+5]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['HPPC_pulse_current_2'], 3))
        assert steps[start+5]['Ends']['EndEntry'][0]['EndType'] == 'StepTime'
        time_s = diagnostic_params['HPPC_pulse_duration_2']
        steps[start+5]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))
        assert steps[start+5]['Ends']['EndEntry'][1]['EndType'] == 'Voltage'
        steps[start+5]['Ends']['EndEntry'][1]['Value'] = \
            float(round(diagnostic_params['HPPC_pulse_cutoff_voltage'], 3))

        # Discharge step 2 for hppc cycle
        assert steps[start+6]['StepType'] == 'Dischrge'
        assert steps[start+6]['StepMode'] == 'Current'
        steps[start+6]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['HPPC_baseline_constant_current'], 3))
        assert steps[start+6]['Ends']['EndEntry'][0]['EndType'] == 'StepTime'
        time_s = int(round(3600 * (diagnostic_params['HPPC_interval'] / 100) /
                           diagnostic_params['HPPC_baseline_constant_current'] - 1))
        steps[start+6]['Ends']['EndEntry'][0]['Value'] = time.strftime('%H:%M:%S', time.gmtime(time_s))
        assert steps[start+6]['Ends']['EndEntry'][1]['EndType'] == 'Voltage'
        steps[start+6]['Ends']['EndEntry'][1]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        # Final discharge step for hppc cycle
        assert steps[start+8]['StepType'] == 'Dischrge'
        assert steps[start+8]['StepMode'] == 'Voltage'
        steps[start+8]['StepValue'] = float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))
        steps[start+8]['Limits']['Current'] = float(round(nominal_capacity *
                                                          diagnostic_params['HPPC_baseline_constant_current'], 3))
        assert steps[start+8]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start+8]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                       diagnostic_params['diagnostic_cutoff_current'], 3))

        return self

    def insert_rpt_cyclev2(self, start, nominal_capacity, diagnostic_params):
        """
        Helper function for parameterizing the Rate Performance Test cycle

        Args:
            start:
            nominal_capacity:
            diagnostic_params:

        Returns:
            dict
        """
        steps = self['MaccorTestProcedure']['ProcSteps']['TestStep']

        # First charge step for rpt cycle
        assert steps[start]['StepType'] == 'Charge'
        assert steps[start]['StepMode'] == 'Current'
        steps[start]['StepValue'] = float(round(nominal_capacity *
                                                diagnostic_params['RPT_charge_constant_current'], 3))
        steps[start]['Limits']['Voltage'] = float(round(diagnostic_params['diagnostic_charge_cutoff_voltage'], 3))
        assert steps[start]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                                   diagnostic_params['diagnostic_cutoff_current'], 3))

        # 0.2C discharge step for rpt cycle
        assert steps[start+1]['StepType'] == 'Dischrge'
        assert steps[start+1]['StepMode'] == 'Current'
        steps[start+1]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['RPT_discharge_constant_current_1'], 3))
        assert steps[start+1]['Ends']['EndEntry'][0]['EndType'] == 'Voltage'
        steps[start+1]['Ends']['EndEntry'][0]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        # Second charge step for rpt cycle
        assert steps[start+3]['StepType'] == 'Charge'
        assert steps[start+3]['StepMode'] == 'Current'
        steps[start+3]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['RPT_charge_constant_current'], 3))
        steps[start+3]['Limits']['Voltage'] = float(round(diagnostic_params['diagnostic_charge_cutoff_voltage'], 3))
        assert steps[start+3]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start+3]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                                     diagnostic_params['diagnostic_cutoff_current'], 3))

        # 1C discharge step for rpt cycle
        assert steps[start+4]['StepType'] == 'Dischrge'
        assert steps[start+4]['StepMode'] == 'Current'
        steps[start+4]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['RPT_discharge_constant_current_2'], 3))
        assert steps[start+4]['Ends']['EndEntry'][0]['EndType'] == 'Voltage'
        steps[start+4]['Ends']['EndEntry'][0]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        # Third charge step for rpt cycle
        assert steps[start+6]['StepType'] == 'Charge'
        assert steps[start+6]['StepMode'] == 'Current'
        steps[start+6]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['RPT_charge_constant_current'], 3))
        steps[start+6]['Limits']['Voltage'] = float(round(diagnostic_params['diagnostic_charge_cutoff_voltage'], 3))
        assert steps[start+6]['Ends']['EndEntry'][0]['EndType'] == 'Current'
        steps[start+6]['Ends']['EndEntry'][0]['Value'] = float(round(nominal_capacity *
                                                                     diagnostic_params['diagnostic_cutoff_current'], 3))

        # 2C discharge step for rpt cycle
        assert steps[start+7]['StepType'] == 'Dischrge'
        assert steps[start+7]['StepMode'] == 'Current'
        steps[start+7]['StepValue'] = float(round(nominal_capacity *
                                                  diagnostic_params['RPT_discharge_constant_current_3'], 3))
        assert steps[start+7]['Ends']['EndEntry'][0]['EndType'] == 'Voltage'
        steps[start+7]['Ends']['EndEntry'][0]['Value'] = \
            float(round(diagnostic_params['diagnostic_discharge_cutoff_voltage'], 3))

        return self


def generate_protocol_files_from_csv(csv_filename, output_directory):
    """
    Generates a set of protocol files from csv filename input by
    reading protocol file input corresponding to each line of
    the csv file. Writes a csv file that.

    Args:
        csv_filename (str): CSV containing protocol file parameters.
        output_directory (str): directory in which to place the output files
    """
    # Read csv file
    protocol_params_df = pd.read_csv(csv_filename)

    new_files = []
    names = []
    result = ''
    message = {'comment': '',
               'error': ''}
    for index, protocol_params in protocol_params_df.iterrows():
        template = protocol_params['template']

        # Switch for template invocation
        if template == "EXP.000":
            procedure = Procedure.from_exp(
                **protocol_params[["cutoff_voltage", "charge_rate", "discharge_rate"]]
            )
        elif template == 'diagnosticV2.000':
            diag_params_df = pd.read_csv(os.path.join(PROCEDURE_TEMPLATE_DIR,
                                                      "PreDiag_parameters - DP.csv"))
            diagnostic_params = diag_params_df[diag_params_df['diagnostic_parameter_set'] ==
                                               protocol_params['diagnostic_parameter_set']].squeeze()

            # TODO: should these be separated?
            procedure = Procedure.generate_procedure_diagcyclev2(
                protocol_params
            )
            procedure.merge(
                Procedure.generate_procedure_diagcyclev2(
                    protocol_params["capacity_nominal"], diagnostic_params
                )
            )
        # TODO: how are these different?
        elif template in ['diagnosticV3.000', 'diagnosticV4.000']:
            diag_params_df = pd.read_csv(os.path.join(PROCEDURE_TEMPLATE_DIR,
                                                      "PreDiag_parameters - DP.csv"))
            diagnostic_params = diag_params_df[diag_params_df['diagnostic_parameter_set'] ==
                                               protocol_params['diagnostic_parameter_set']].squeeze()

            procedure = Procedure.generate_procedure_regcyclev3(index, protocol_params)
            procedure.merge(
                Procedure.generate_procedure_diagcyclev3(
                    protocol_params["capacity_nominal"], diagnostic_params
                )
            )
        else:
            warnings.warn("Unsupported file template {}, skipping.".format(template))
            result = "error"
            message = {'comment': 'Unable to find template: ' + template,
                       'error': 'Not Found'}
            continue

        filename_prefix = '_'.join(
            [protocol_params["project_name"], '{:06d}'.format(protocol_params["seq_num"])])
        filename = "{}.000".format(filename_prefix)
        filename = os.path.join(output_directory, 'procedures', filename)
        logger.info(filename, extra=s)
        if not os.path.isfile(filename):
            procedure.to_file(filename)
            new_files.append(filename)
            names.append(filename_prefix + '_')

        elif '.sdu' in template:
            logger.warning('Schedule file generation not yet implemented', extra=s)
            result = "error"
            message = {'comment': 'Schedule file generation is not yet implemented',
                       'error': 'Not Implemented'}

    # This block of code produces the file containing all of the run file
    # names produced in this function call. This is to make starting tests easier
    _, namefile = os.path.split(csv_filename)
    namefile = namefile.split('_')[0] + '_names_'
    namefile = namefile + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
    with open(os.path.join(output_directory, "names", namefile), 'w', newline='') as outputfile:
        wr = csv.writer(outputfile)
        for name in names:
            wr.writerow([name])
    outputfile.close()

    if not result:
        result = "success"
        message = {'comment': 'Generated {} protocols'.format(str(len(new_files))),
                   'error': ''}

    return new_files, result, message


def process_csv_file_list_from_json(
        file_list_json,
        processed_dir='data-share/protocols/'):
    """

    Args:
        file_list_json (str):
        processed_dir (str):

    Returns:
        str:
    """
    # Get file list and validity from json, if ends with .json,
    # assume it's a file, if not assume it's a json string
    if file_list_json.endswith(".json"):
        file_list_data = loadfn(file_list_json)
    else:
        file_list_data = json.loads(file_list_json)

    # Setup Events
    events = KinesisEvents(service='protocolGenerator', mode=file_list_data['mode'])

    file_list = file_list_data['file_list']
    all_output_files = []
    protocol_dir = os.path.join(os.environ.get("BEEP_ROOT", "/"),
                              processed_dir)
    for filename in file_list:
        output_files, result, message = generate_protocol_files_from_csv(
            filename, output_directory=protocol_dir)
        all_output_files.extend(output_files)

    output_data = {"file_list": all_output_files,
                   "result": result,
                   "message": message
                   }

    events.put_generate_event(output_data, "complete")

    return json.dumps(output_data)


def main():
    """Main function for the script"""
    logger.info('starting', extra=s)
    logger.info('Running version=%s', __version__, extra=s)
    try:
        args = docopt(__doc__)
        input_json = args['INPUT_JSON']
        print(process_csv_file_list_from_json(input_json), end="")
    except Exception as e:
        logger.error(str(e), extra=s)
        raise e
    logger.info('finish', extra=s)
    return None


if __name__ == "__main__":
    main()
