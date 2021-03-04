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
Module and script for processing cycler run CSVs into structured data
for featurization and analysis.  Contains RawCyclerRun and ProcessedCyclerRun
objects, which parse and structure data, along with utility
functions for manipulating tabular data.

Usage:
    structure [INPUT_JSON]

Options:
    -h --help       Show this screen
    --version       Show version


The `structure` script will run the data structuring on specified filenames corresponding
to validated raw cycler files.  It places the structured datafiles in `/data-share/structure`.

The input json must contain the following fields:
* `file_list` - a list of full path filenames which have been processed
* `validity` - a list of boolean validation results, e. g. `[True, True, False]`

The output json contains the following fields:

* `invalid_file_list` - a list of invalid files according to the validity
* `file_list` - a list of files which have been structured into processed_cycler_runs

Example:
```angular2
$ structure '{"validity": [false, false, true],
             file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
                          "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv",
                          "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"]}''
{"invalid_file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
                       "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv"],
 "file_list": ["/data-share/structure/FastCharge_2_CH29_structure.json"]}
```
"""

import json
import re
from datetime import datetime

import pandas as pd
import numpy as np
import os
import pytz
import time
from scipy import integrate
import itertools
import hashlib

from monty.json import MSONable
from docopt import docopt
from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir
from glob import glob
from beep import tqdm

from beep import StringIO, MODULE_DIR
from beep.validate import ValidatorBeep, BeepValidationError
from beep.collate import add_suffix_to_filename
from beep.conversion_schemas import (
    ARBIN_CONFIG,
    MACCOR_CONFIG,
    FastCharge_CONFIG,
    xTesladiag_CONFIG,
    INDIGO_CONFIG,
    NEWARE_CONFIG,
    BIOLOGIC_CONFIG,
    STRUCTURE_DTYPES,
)

from beep.utils import WorkflowOutputs, parameters_lookup
from beep import logger, __version__

s = {"service": "DataStructurer"}

VOLTAGE_RESOLUTION = 3


class RawCyclerRun(MSONable):
    """
    Object corresponding to parsed cycler run, includes a factory method to
    construct object from a file path (e. g. "test_file.csv")

    Attributes:
        data (pandas.DataFrame): DataFrame corresponding to cycler run data.
        metadata (dict): Dict corresponding to cycler run metadata.
        eis (beep.structure.EISpectrum): electrochemical impedence
            spectrum object. Defaults to None.
        validate (bool): whether or not to validate DataFrame upon
            instantiation. Defaults to None.
    """

    # These define float and int columns for numeric binary save files
    FLOAT_COLUMNS = [
        "test_time",
        "current",
        "voltage",
        "charge_capacity",
        "discharge_capacity",
        "charge_energy",
        "discharge_energy",
        "internal_resistance",
        "temperature",
    ]
    INT_COLUMNS = ["step_index", "cycle_index"]

    IMPUTABLE_COLUMNS = ["temperature", "internal_resistance"]

    def __init__(self, data, metadata, eis=None, validate=False, filename=None,
                 impute_missing=True):
        """
        Args:
            data (pandas.DataFrame): DataFrame corresponding to cycler run data
            metadata (dict): Dict corresponding to cycler run metadata
            eis (beep.structure.EISpectrum): electrochemical impedence
                spectrum object. Defaults to None
            validate (bool): whether or not to validate DataFrame upon
                instantiation. Defaults to None.
            impute_missing (bool): Impute missing columns required for
                downstream processing (e.g., temperature).

        """
        if validate:
            validator = ValidatorBeep()
            is_valid = validator.validate_arbin_dataframe(data)
            if not is_valid:
                raise BeepValidationError("Beep validation failed")

        if impute_missing:
            for col in self.IMPUTABLE_COLUMNS:
                if col not in data:
                    data[col] = np.NaN

        self.data = data
        self.metadata = metadata
        self.eis = eis
        self.filename = filename

    @classmethod
    def from_file(cls, path, validate=False):
        """
        Factory method to invoke RawCyclerRun from filename with recognition of
        type from filename, using corresponding class method as constructor.

        Args:
            path (str): string corresponding to file path.
            validate (bool): whether or not to validate file.

        Returns:
            beep.structure.RawCyclerRun: RawCyclerRun corresponding to parsed file(s).

        """
        if re.match(ARBIN_CONFIG["file_pattern"], path):
            return cls.from_arbin_file(path, validate)

        elif re.match(MACCOR_CONFIG["file_pattern"], path):
            return cls.from_maccor_file(path, False, validate)

        elif re.match(INDIGO_CONFIG["file_pattern"], path):
            return cls.from_indigo_file(path, validate)

        elif re.match(BIOLOGIC_CONFIG["file_pattern"], path):
            return cls.from_biologic_file(path, validate)

        elif re.match(NEWARE_CONFIG["file_pattern"], path):
            return cls.from_neware_file(path, validate)

        else:
            raise ValueError("{} does not match any known file pattern".format(path))

    def get_interpolated_steps(
        self,
        v_range,
        resolution,
        step_type="discharge",
        reg_cycles=None,
        axis="voltage",
    ):
        """
        Gets interpolated cycles for the step specified, charge or discharge.

        Args:
            v_range ([Float, Float]): list of two floats that define
                the voltage interpolation range endpoints.
            resolution (int): resolution of interpolated data.
            step_type (str): which step to interpolate i.e. 'charge' or 'discharge'
            reg_cycles (list): list containing cycle indicies of regular cycles
            axis (str): which column to use for interpolation

        Returns:
            pandas.DataFrame: DataFrame corresponding to interpolated values.
        """
        if step_type == "discharge":
            step_filter = determine_whether_step_is_discharging
        elif step_type == "charge":
            step_filter = determine_whether_step_is_charging
        else:
            raise ValueError("{} is not a recognized step type")
        incl_columns = [
            "test_time",
            "voltage",
            "current",
            "charge_capacity",
            "discharge_capacity",
            "charge_energy",
            "discharge_energy",
            "internal_resistance",
            "temperature",
        ]
        all_dfs = []
        cycle_indices = self.data.cycle_index.unique()
        cycle_indices = [c for c in cycle_indices if c in reg_cycles]
        cycle_indices.sort()

        for cycle_index in tqdm(cycle_indices):
            # Use a cycle_index mask instead of a global groupby to save memoryÍ
            new_df = (
                self.data.loc[self.data["cycle_index"] == cycle_index]
                .groupby("step_index")
                .filter(step_filter)
            )
            if new_df.size == 0:
                continue

            if axis in ["charge_capacity", "discharge_capacity"]:
                axis_range = [self.data[axis].min(), self.data[axis].max()]
                new_df = get_interpolated_data(
                    new_df,
                    axis,
                    field_range=axis_range,
                    columns=incl_columns,
                    resolution=resolution,
                )
            elif axis == "test_time":
                axis_range = [new_df[axis].min(), new_df[axis].max()]
                new_df = get_interpolated_data(
                    new_df,
                    axis,
                    field_range=axis_range,
                    columns=incl_columns,
                    resolution=resolution,
                )
            elif axis == "voltage":
                new_df = get_interpolated_data(
                    new_df,
                    axis,
                    field_range=v_range,
                    columns=incl_columns,
                    resolution=resolution,
                )
            else:
                raise NotImplementedError
            new_df["cycle_index"] = cycle_index
            new_df["step_type"] = step_type
            new_df["step_type"] = new_df["step_type"].astype("category")
            all_dfs.append(new_df)

        # Ignore the index to avoid issues with overlapping voltages
        result = pd.concat(all_dfs, ignore_index=True)

        # Cycle_index gets a little weird about typing, so round it here
        result.cycle_index = result.cycle_index.round()

        return result

    def get_interpolated_cycles(
        self, v_range=None, resolution=1000, diagnostic_available=None,
            charge_axis='charge_capacity', discharge_axis='voltage'):
        """
        Gets interpolated cycles for both charge and discharge steps.

        Args:
            v_range ([Float, Float]): list of two floats that define
                the voltage interpolation range endpoints.
            resolution (int): resolution of interpolated data.
            diagnostic_available (dict): dictionary containing information about
                location of diagnostic cycles

        Returns:
            pandas.DataFrame: DataFrame corresponding to interpolated values.
        """
        if diagnostic_available:
            diag_cycles = list(
                itertools.chain.from_iterable(
                    [
                        list(range(i, i + diagnostic_available["length"]))
                        for i in diagnostic_available["diagnostic_starts_at"]
                        if i <= self.data.cycle_index.max()
                    ]
                )
            )
            reg_cycles = [
                i for i in self.data.cycle_index.unique() if i not in diag_cycles
            ]
        else:
            reg_cycles = [i for i in self.data.cycle_index.unique()]

        v_range = v_range or [2.8, 3.5]

        # If any regular cycle contains a waveform step, interpolate on test_time.
        if self.data[self.data.cycle_index.isin(reg_cycles)].\
                groupby(["cycle_index", "step_index"]).\
                apply(determine_whether_step_is_waveform_discharge).any():
            discharge_axis = 'test_time'

        if self.data[self.data.cycle_index.isin(reg_cycles)].\
                groupby(["cycle_index", "step_index"]).\
                apply(determine_whether_step_is_waveform_charge).any():
            charge_axis = 'test_time'

        interpolated_discharge = self.get_interpolated_steps(
            v_range,
            resolution,
            step_type="discharge",
            reg_cycles=reg_cycles,
            axis=discharge_axis,
        )
        interpolated_charge = self.get_interpolated_steps(
            v_range,
            resolution,
            step_type="charge",
            reg_cycles=reg_cycles,
            axis=charge_axis,
        )
        result = pd.concat(
            [interpolated_discharge, interpolated_charge], ignore_index=True
        )

        return result.astype(STRUCTURE_DTYPES["cycles_interpolated"])

    def as_dict(self):
        """
        Method for dictionary/json serialization hook in MSONable

        Returns:
            dict: Representation as dictionary.
        """
        obj = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": self.data.to_dict("list"),
            "metadata": self.metadata,
            "eis": self.eis,
        }
        return obj

    @classmethod
    def from_dict(cls, d):
        """
        Method for dictionary/json deserialization hook in MSONable

        Returns:
            beep.structure.RawCyclerRun:
        """

        data = pd.DataFrame(d["data"])
        data = data.sort_index()
        return cls(data, d["metadata"], d["eis"])

    def get_summary(
        self,
        diagnostic_available=False,
        nominal_capacity=1.1,
        full_fast_charge=0.8,
        cycle_complete_discharge_ratio=0.97,
        cycle_complete_vmin=3.3,
        cycle_complete_vmax=3.3,
        error_threshold=1e6,
    ):
        """
        Gets summary statistics for data according to cycle number. Summary data
        must be float or int type for compatibility with other methods

        Args:
            diagnostic_available (dict): dictionary with diagnostic_types
            nominal_capacity (float): nominal capacity for summary stats
            full_fast_charge (float): full fast charge for summary stats
            cycle_complete_discharge_ratio (float): expected ratio
                discharge/charge at the end of any complete cycle
            cycle_complete_vmin (float): expected voltage minimum achieved
                in any complete cycle
            cycle_complete_vmax (float): expected voltage maximum achieved
                in any complete cycle
            error_threshold (float): threshold to consider the summary value
                an error (applied only to specific columns that should reset
                each cycle)

        Returns:
            pandas.DataFrame: summary statistics by cycle.

        """
        # Filter out only regular cycles for summary stats. Diagnostic summary computed separately
        if diagnostic_available:
            diag_cycles = list(
                itertools.chain.from_iterable(
                    [
                        list(range(i, i + diagnostic_available["length"]))
                        for i in diagnostic_available["diagnostic_starts_at"]
                        if i <= self.data.cycle_index.max()
                    ]
                )
            )
            reg_cycles_at = [
                i for i in self.data.cycle_index.unique() if i not in diag_cycles
            ]
        else:
            reg_cycles_at = [i for i in self.data.cycle_index.unique()]

        summary = self.data.groupby("cycle_index").agg(
            {
                "cycle_index": "first",
                "discharge_capacity": "max",
                "charge_capacity": "max",
                "discharge_energy": "max",
                "charge_energy": "max",
                "internal_resistance": "last",
                "temperature": ["max", "mean", "min"],
                "date_time_iso": "first",
            }
        )

        summary.columns = [
            "cycle_index",
            "discharge_capacity",
            "charge_capacity",
            "discharge_energy",
            "charge_energy",
            "dc_internal_resistance",
            "temperature_maximum",
            "temperature_average",
            "temperature_minimum",
            "date_time_iso",
        ]
        summary = summary[summary.index.isin(reg_cycles_at)]
        summary["energy_efficiency"] = (
            summary["discharge_energy"] / summary["charge_energy"]
        )
        summary.loc[
            ~np.isfinite(summary["energy_efficiency"]), "energy_efficiency"
        ] = np.NaN
        # This code is designed to remove erroneous energy values
        for col in ["discharge_energy", "charge_energy"]:
            summary.loc[summary[col].abs() > error_threshold, col] = np.NaN
        summary["charge_throughput"] = summary.charge_capacity.cumsum()
        summary["energy_throughput"] = summary.charge_energy.cumsum()

        # This method for computing charge start and end times implicitly
        # assumes that a cycle starts with a charge step and is then followed
        # by discharge step.
        charge_start_time = self.data.groupby("cycle_index", as_index=False)[
            "date_time_iso"
        ].agg("first")
        charge_finish_time = (
            self.data[self.data.charge_capacity >= nominal_capacity * full_fast_charge]
            .groupby("cycle_index", as_index=False)["date_time_iso"]
            .agg("first")
        )

        # Left merge, since some cells might not reach desired levels of
        # charge_capacity and will have NaN for charge duration
        merged = charge_start_time.merge(
            charge_finish_time, on="cycle_index", how="left"
        )

        # Charge duration stored in seconds - note that date_time_iso is only ~1sec resolution
        time_diff = np.subtract(
            pd.to_datetime(merged.date_time_iso_y, utc=True, errors="coerce"),
            pd.to_datetime(merged.date_time_iso_x, errors="coerce"),
        )
        summary["charge_duration"] = np.round(time_diff / np.timedelta64(1, "s"), 2)

        # Compute time since start of cycle in minutes. This comes handy
        # for featurizing time-temperature integral
        self.data["time_since_cycle_start"] = pd.to_datetime(
            self.data["date_time_iso"]
        ) - pd.to_datetime(
            self.data.groupby("cycle_index")["date_time_iso"].transform("first")
        )
        self.data["time_since_cycle_start"] = (
            self.data["time_since_cycle_start"] / np.timedelta64(1, "s")
        ) / 60

        # Group by cycle index and integrate time-temperature
        # using a lambda function.
        summary["time_temperature_integrated"] = self.data.groupby("cycle_index").apply(
            lambda g: integrate.trapz(g.temperature, x=g.time_since_cycle_start)
        )

        # Drop the time since cycle start column
        self.data.drop(columns=["time_since_cycle_start"])

        # Determine if any of the cycles has been paused
        summary["paused"] = self.data.groupby("cycle_index").apply(get_max_paused_over_threshold)

        summary = summary.astype(STRUCTURE_DTYPES["summary"])

        last_voltage = self.data.loc[
            self.data["cycle_index"] == self.data["cycle_index"].max()
        ]["voltage"]
        if (
            (last_voltage.min() < cycle_complete_vmin)
            and (last_voltage.max() > cycle_complete_vmax)
            and (
                (summary.iloc[[-1]])["discharge_capacity"].iloc[0]
                > cycle_complete_discharge_ratio
                * (summary.iloc[[-1]])["charge_capacity"].iloc[0]
            )
        ):
            return summary
        else:
            return summary.iloc[:-1]

    def get_diagnostic_summary(self, diagnostic_available):
        """
        Gets summary statistics for data according to location of
        diagnostic cycles in the data

        Args:
            diagnostic_available (dict): dictionary with diagnostic_types
                as list, 'length' of the diagnostic in cycles and location
                of the diagnostic by cycle index

        Returns:
            (DataFrame) of summary statistics by cycle

        """

        max_cycle = self.data.cycle_index.max()
        starts_at = [
            i for i in diagnostic_available["diagnostic_starts_at"] if i <= max_cycle
        ]
        diag_cycles_at = list(
            itertools.chain.from_iterable(
                [list(range(i, i + diagnostic_available["length"])) for i in starts_at]
            )
        )
        diag_summary = self.data.groupby("cycle_index").agg(
            {
                "discharge_capacity": "max",
                "charge_capacity": "max",
                "discharge_energy": "max",
                "charge_energy": "max",
                "temperature": ["max", "mean", "min"],
                "date_time_iso": "first",
                "cycle_index": "first",
            },
        )

        diag_summary.columns = [
            "discharge_capacity",
            "charge_capacity",
            "discharge_energy",
            "charge_energy",
            "temperature_maximum",
            "temperature_average",
            "temperature_minimum",
            "date_time_iso",
            "cycle_index",
        ]
        diag_summary["paused"] = self.data.groupby("cycle_index").apply(
            get_max_paused_over_threshold
        )

        diag_summary = diag_summary[diag_summary.index.isin(diag_cycles_at)]

        diag_summary["coulombic_efficiency"] = (
            diag_summary["discharge_capacity"] / diag_summary["charge_capacity"]
        )

        diag_summary.reset_index(drop=True, inplace=True)

        diag_summary["cycle_type"] = pd.Series(
            diagnostic_available["cycle_type"] * len(starts_at)
        )

        diag_summary = diag_summary.astype(STRUCTURE_DTYPES["diagnostic_summary"])

        return diag_summary

    def get_interpolated_diagnostic_cycles(
        self, diagnostic_available, resolution=1000, v_resolution=0.0005
    ):
        """
        Interpolates data according to location and type of diagnostic
        cycles in the data

        Args:
            diagnostic_available (dict): dictionary with diagnostic_types
                as list, 'length' of the diagnostic in cycles and location
                of the diagnostic
            resolution (int): resolution of interpolation
            v_resolution (float): voltage delta to set for range based interpolation

        Returns:
            (DataFrame) of interpolated diagnostic steps by step and cycle

        """
        # Get the project name and the parameter file for the diagnostic
        project_name_list = parameters_lookup.get_project_sequence(self.filename)
        diag_path = os.path.join(MODULE_DIR, "procedure_templates")
        v_range = parameters_lookup.get_diagnostic_parameters(
            diagnostic_available, diag_path, project_name_list[0]
        )

        # Determine the cycles and types of the diagnostic cycles
        max_cycle = self.data.cycle_index.max()
        starts_at = [
            i for i in diagnostic_available["diagnostic_starts_at"] if i <= max_cycle
        ]
        diag_cycles_at = list(
            itertools.chain.from_iterable(
                [range(i, i + diagnostic_available["length"]) for i in starts_at]
            )
        )
        # Duplicate cycle type list end to end for each starting index
        diag_cycle_type = diagnostic_available["cycle_type"] * len(starts_at)
        if not len(diag_cycles_at) == len(diag_cycle_type):
            errmsg = (
                "Diagnostic cycles, {}, and diagnostic cycle types, "
                "{}, are unequal lengths".format(diag_cycles_at, diag_cycle_type)
            )
            raise ValueError(errmsg)

        diag_data = self.data[self.data["cycle_index"].isin(diag_cycles_at)]

        # Counter to ensure non-contiguous repeats of step_index
        # within same cycle_index are grouped separately
        diag_data.loc[:, "step_index_counter"] = 0

        for cycle_index in diag_cycles_at:
            indices = diag_data.loc[diag_data.cycle_index == cycle_index].index
            step_index_list = diag_data.step_index.loc[indices]
            diag_data.loc[indices, "step_index_counter"] = step_index_list.ne(
                step_index_list.shift()
            ).cumsum()

        group = diag_data.groupby(["cycle_index", "step_index", "step_index_counter"])
        incl_columns = [
            "current",
            "charge_capacity",
            "discharge_capacity",
            "charge_energy",
            "discharge_energy",
            "internal_resistance",
            "temperature",
            "test_time",
        ]

        diag_dict = {}
        for cycle in diag_data.cycle_index.unique():
            diag_dict.update({cycle: None})
            steps = diag_data[diag_data.cycle_index == cycle].step_index.unique()
            diag_dict[cycle] = list(steps)

        all_dfs = []
        for (cycle_index, step_index, step_index_counter), df in tqdm(group):
            if len(df) < 2:
                continue
            if diag_cycle_type[diag_cycles_at.index(cycle_index)] == "hppc":
                v_hppc_step = [df.voltage.min(), df.voltage.max()]
                hppc_resolution = int(
                    (df.voltage.max() - df.voltage.min()) / v_resolution
                )
                new_df = get_interpolated_data(
                    df,
                    field_name="voltage",
                    field_range=v_hppc_step,
                    columns=incl_columns,
                    resolution=hppc_resolution,
                )
            else:
                new_df = get_interpolated_data(
                    df,
                    field_name="voltage",
                    field_range=v_range,
                    columns=incl_columns,
                    resolution=resolution,
                )

            new_df["cycle_index"] = cycle_index
            new_df["cycle_type"] = diag_cycle_type[diag_cycles_at.index(cycle_index)]
            new_df["step_index"] = step_index
            new_df["step_index_counter"] = step_index_counter
            new_df["step_type"] = diag_dict[cycle_index].index(step_index)
            new_df.astype(
                {
                    "cycle_index": "int32",
                    "cycle_type": "category",
                    "step_index": "uint8",
                    "step_index_counter": "int16",
                    "step_type": "uint8",
                }
            )
            new_df["discharge_dQdV"] = (
                new_df.discharge_capacity.diff() / new_df.voltage.diff()
            )
            new_df["charge_dQdV"] = (
                new_df.charge_capacity.diff() / new_df.voltage.diff()
            )
            all_dfs.append(new_df)

        # Ignore the index to avoid issues with overlapping voltages
        result = pd.concat(all_dfs, ignore_index=True)
        # Cycle_index gets a little weird about typing, so round it here
        result.cycle_index = result.cycle_index.round()

        result = result.astype(STRUCTURE_DTYPES["diagnostic_interpolated"])

        return result

    @classmethod
    def from_arbin_file(cls, path, validate=False):
        """
        Creates RawCyclerRun from an Arbin data file.

        Args:
            path (str): file path to data file
            validate (bool): True if data is to be validated.

        Returns:
            beep.structure.RawCyclerRun
        """
        data = pd.read_csv(path)
        data.rename(str.lower, axis="columns", inplace=True)

        for column, dtype in ARBIN_CONFIG["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(ARBIN_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata_path = path.replace(".csv", "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(ARBIN_CONFIG["metadata_fields"], axis="columns", inplace=True)
            # Note the to_dict, which scrubs numpy typing
            metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}
        else:
            logger.warning(f"No associated metadata file for Arbin: "
                           f"'{metadata_path}'. No metadata loaded.")
            metadata = {}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(
            lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.UTC).isoformat()
        )
        return cls(data, metadata, None, validate, filename=path)

    @classmethod
    def from_indigo_file(cls, path, validate=False):
        """
        Creates RawCyclerRun from an Indigo data file.

        Args:
            path (str): file path to data file
            validate (bool): True if data is to be validated.

        Returns:
            beep.structure.RawCyclerRun
        """

        data = pd.read_hdf(path, "time_series_data")
        metadata = dict()

        if len(list(data["cell_id"].unique())) > 1:
            raise ValueError("More than 1 cell_id exists in {}".format(path))

        metadata["indigo_cell_id"] = int(data["cell_id"].iloc[0])
        metadata["filename"] = path
        metadata["_today_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # transformations
        data = (
            data.reset_index().reset_index()
        )  # twice in case old index is stored in file
        data = data.drop(columns=["index"])
        data.rename(columns={"level_0": "data_point"}, inplace=True)
        data.loc[data.half_cycle_count % 2 == 1, "charge_capacity"] = (
            abs(data.cell_coulomb_count_c) / 3600
        )
        data.loc[data.half_cycle_count % 2 == 0, "charge_capacity"] = 0
        data.loc[data.half_cycle_count % 2 == 0, "discharge_capacity"] = (
            abs(data.cell_coulomb_count_c) / 3600
        )
        data.loc[data.half_cycle_count % 2 == 1, "discharge_capacity"] = 0
        data.loc[data.half_cycle_count % 2 == 1, "charge_energy"] = abs(
            data.cell_energy_j
        )
        data.loc[data.half_cycle_count % 2 == 0, "charge_energy"] = 0
        data.loc[data.half_cycle_count % 2 == 0, "discharge_energy"] = abs(
            data.cell_energy_j
        )
        data.loc[data.half_cycle_count % 2 == 1, "discharge_energy"] = 0
        data["internal_resistance"] = data.cell_voltage_v / data.cell_current_a
        data["date_time_iso"] = data["system_time_us"].apply(
            lambda x: datetime.utcfromtimestamp(x / 1000000)
            .replace(tzinfo=pytz.UTC)
            .isoformat()
        )

        data.rename(INDIGO_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata["start_datetime"] = data.sort_values(by="system_time_us")[
            "date_time_iso"
        ].iloc[0]

        return cls(data, metadata, None, validate, filename=path)

    @classmethod
    def from_neware_file(cls, filename, validate=False):
        """
        Method for ingestion of Neware format files.

        Args:
            filename (str): file path for neware format file.
            validate (bool): whether to validate on instantiation.

        Returns:
            beep.structure.RawCyclerRun
        """
        ir_column_name = '"DCIR(O)"'
        with open(filename, encoding="ISO-8859-1") as input:
            with ScratchDir("."):
                cycle_header = input.readline().replace("\t", "")
                cycle_file = open("cycle_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = cycle_header.encode("ascii", "ignore")
                cycle_header = encoded_string.decode()
                cycle_file.write(cycle_header)

                step_header = input.readline().replace("\t", "")
                ir_index = step_header.split(",").index(ir_column_name)
                step_file = open("step_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = step_header.encode("ascii", "ignore")
                step_header = encoded_string.decode()
                step_file.write(step_header)

                record_header = input.readline().replace("\t", "")
                record_header = record_header.split(",")
                record_header[0] = cycle_header.split(",")[0]
                record_header[1] = step_header.split(",")[1]
                record_header[22] = ir_column_name
                record_header = ",".join(record_header)
                record_file = open("record_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = record_header.encode("ascii", "ignore")
                record_header = encoded_string.decode()
                record_file.write(record_header)

                # Read file line by line and write to the appropriate file
                cycle_number = 0
                step_number = 0
                for row, line in enumerate(input):
                    if line[:2] == r',"':
                        step_file.write(line)
                        step_number = line.split(",")[1]
                        ir_value = line.split(",")[ir_index]
                    elif line[:2] == r",,":
                        line_list = line.split(",")
                        line_list[0] = cycle_number
                        line_list[1] = step_number
                        line_list[22] = ir_value
                        line = ",".join(line_list)
                        record_file.write(line)
                    else:
                        cycle_file.write(line)
                        cycle_number = line.split(",")[0]
                record_file.close()
                step_file.close()
                cycle_file.close()

                # Read in the data and convert the column values to MKS units
                data = pd.read_csv(
                    "record_file.csv", sep=",", skiprows=0, encoding="ISO-8859-1"
                )
                data = data.loc[:, ~data.columns.str.contains("Unnamed")]
                data["Time(h:min:s.ms)"] = data["Time(h:min:s.ms)"].apply(
                    neware_step_time
                )
                data["Current(mA)"] = data["Current(mA)"] / 1000
                data["Capacitance_Chg(mAh)"] = data["Capacitance_Chg(mAh)"] / 1000
                data["Capacitance_DChg(mAh)"] = data["Capacitance_DChg(mAh)"] / 1000
                data["Engy_Chg(mWh)"] = data["Engy_Chg(mWh)"] / 1000
                data["Engy_DChg(mWh)"] = data["Engy_DChg(mWh)"] / 1000

                # Deal with missing data in the internal resistance
                data["DCIR(O)"] = data["DCIR(O)"].apply(
                    lambda x: np.nan if x == "\t-" else x
                )
                data["DCIR(O)"] = data["DCIR(O)"].fillna(method="ffill")
                data["DCIR(O)"] = data["DCIR(O)"].fillna(method="bfill")

        data["test_time"] = (
            data["Time(h:min:s.ms)"]
            .diff()
            .fillna(0)
            .apply(lambda x: 0 if x < 0 else x)
            .cumsum()
        )
        # print(data.columns)
        # print(NEWARE_CONFIG["data_types"])
        data = data.astype(NEWARE_CONFIG["data_types"])

        data.rename(NEWARE_CONFIG["data_columns"], axis="columns", inplace=True)
        data["date_time"] = data["date_time"].apply(lambda x: x.replace("\t", ""))
        data["date_time_iso"] = data["date_time"].apply(maccor_timestamp)

        metadata = dict()
        metadata["filename"] = filename
        path = filename

        return cls(data, metadata, None, validate, filename=path)

    @classmethod
    def from_biologic_file(cls, path, validate=False):
        """
        Creates RawCyclerRun from an Biologic data file.

        Args:
            path (str): file path to data file
            validate (bool): True if data is to be validated.

        Returns:
            beep.structure.RawCyclerRun
        """

        header_line = 3  # specific to file layout
        data_starts_line = 4  # specific to file layout
        column_map = BIOLOGIC_CONFIG["data_columns"]

        raw = dict()
        i = 0
        with open(path, "rb") as f:

            empty_lines = 0  # used to find the end of the data entries.
            while empty_lines < 2:  # finding 2 empty lines in a row => EOF
                line = f.readline()
                i += 1

                if i == header_line:
                    header = str(line.strip())[2:-1]
                    columns = header.split("\\t")
                    for c in columns:
                        raw[c] = list()

                if i >= data_starts_line:
                    line = line.strip().decode()
                    if len(line) == 0:
                        empty_lines += 1
                        continue
                    items = line.split("\t")
                    for ci in range(len(items)):
                        column_name = columns[ci]
                        data_type = column_map.get(column_name, dict()).get(
                            "data_type", str
                        )
                        scale = column_map.get(column_name, dict()).get("scale", 1.0)
                        item = items[ci]
                        if data_type == "int":
                            item = int(float(item))
                        if data_type == "float":
                            item = float(item) * scale
                        raw[column_name].append(item)

        data = dict()
        for column_name in column_map.keys():
            data[column_map[column_name]["beep_name"]] = raw[column_name]
        data["data_point"] = list(range(1, len(raw["cycle number"]) + 1))

        data = pd.DataFrame(data)
        data.loc[data.step_index % 2 == 0, "charge_capacity"] = abs(
            data.charge_capacity
        )
        data.loc[data.step_index % 2 == 1, "charge_capacity"] = 0
        data.loc[data.step_index % 2 == 1, "discharge_capacity"] = abs(
            data.discharge_capacity
        )
        data.loc[data.step_index % 2 == 0, "discharge_capacity"] = 0

        metadata_path = path.replace(".mpt", ".mpl")
        metadata = cls.extract_biologic_metadata(metadata_path)
        metadata["filename"] = path
        metadata["_today_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return cls(data, metadata, None, validate, filename=path, impute_missing=False)

    @staticmethod
    def extract_biologic_metadata(metadata_path):
        """
        Iterates through a biologic metadata file and extracts the meta fields:
            - cell_id
            - barcode
            - protocol

        Args:
            metadata_path (str):                path to metadata file

        Returns:
            dict:                               dictionary with metadata fields
        """

        flag_cell_id = False
        flag_barcode = False
        flag_protocol = False
        flag_protocol_start = False
        protocol_text = ""
        max_lines = 10000

        metadata = dict()

        with open(metadata_path, "rb") as f:

            i = 0
            while True:
                line = f.readline()
                line = str(line)
                if line.startswith("b'"):
                    line = line[2:]
                if line.endswith("'"):
                    line = line[:-1]
                if line.endswith("\\r\\n"):
                    line = line[:-4]

                if line.strip().split(" : ")[0] == "Run on channel":
                    channel_id = line.strip().split(" : ")[1]
                    channel_id = channel_id.split(" ")[0]
                    if channel_id[0] != "C":
                        raise RuntimeError(
                            "Unexpected channel id format in biologic log file."
                        )
                    channel_id = int(channel_id[1:])
                    metadata["channel_id"] = channel_id
                    flag_cell_id = True

                if "Comments" in line.strip().split(" : ")[0]:
                    barcode = line.strip().split(" : ")[-1]
                    metadata["barcode"] = barcode
                    flag_barcode = True

                if "Cycle Definition" in line.strip().split(" : ")[0]:
                    protocol_text += line + "\n"
                    flag_protocol_start = True

                if flag_protocol_start:
                    if line == "":
                        flag_protocol = True
                        protocol = hashlib.md5(protocol_text.encode()).digest()
                        metadata["protocol"] = protocol
                    else:
                        protocol_text += line + "\n"

                done = flag_cell_id and flag_barcode and flag_protocol
                if done:
                    break

                i += 1
                if i > max_lines:
                    break

        return metadata

    @staticmethod
    def get_maccor_quantity_sum(data, quantity, state_type):
        """
        Computes non-decreasing capacity or energy (either charge or discharge)
        through multiple steps of a single cycle and resets capacity at the
        start of each new cycle. Input Maccor data resets to zero at each step.

        Args:
            data (pd.DataFrame): maccor data.
            quantity (str): capacity or energy.
            state_type (str): charge or discharge.

        Returns:
            Series: summed quantities.

        """
        state_code = MACCOR_CONFIG["{}_state_code".format(state_type)]
        quantity_agg = data['_' + quantity].where(data["_state"] == state_code, other=0, axis=0)

        # If a waveform step is present, maccor initializes waveform-specific quantities
        # that are to be used in place of '_capacity' and '_energy'

        if data['_wf_chg_cap'].notna().sum():
            if (state_type, quantity) == ('discharge', 'capacity'):
                quantity_agg = data['_wf_dis_cap'].where(data['_wf_dis_cap'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('charge', 'capacity'):
                quantity_agg = data['_wf_chg_cap'].where(data['_wf_chg_cap'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('discharge', 'energy'):
                quantity_agg = data['_wf_dis_e'].where(data['_wf_dis_e'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('charge', 'energy'):
                quantity_agg = data['_wf_chg_e'].where(data['_wf_chg_e'].notna(), other=quantity_agg, axis=0)
            else:
                pass

        end_step = data["_ending_status"].apply(
            lambda x: MACCOR_CONFIG["end_step_code_min"] <= x <= MACCOR_CONFIG["end_step_code_max"]
        )
        # For waveform discharges, maccor seems to trigger ending_status within a step multiple times
        # As a fix, compute the actual step change using diff() on step_index and set end_step to be
        # a logical AND(step_change, end_step)
        is_step_change = data['step_index'].diff(periods=-1).fillna(value=0) != 0
        end_step_inds = end_step.index[np.logical_and(list(end_step), list(is_step_change))]
        # If no end steps, quantity not reset, return it without modifying
        if end_step_inds.size == 0:
            return quantity_agg

        # Initialize accumulator and beginning step slice index
        cycle_sum = 0.0
        begin_step_ind = quantity_agg.index[0] + 1
        for end_step_ind in end_step_inds:
            # Detect whether cycle changed and reset accumulator if so
            if (
                data.loc[begin_step_ind - 1, "cycle_index"]
                != data.loc[begin_step_ind, "cycle_index"]
            ):
                cycle_sum = 0.0

            # Add accumulator to current reset step
            quantity_agg[begin_step_ind:end_step_ind + 1] += cycle_sum

            # Update accumulator
            cycle_sum = quantity_agg[end_step_ind]

            # Set new step slice initial index
            begin_step_ind = end_step_ind + 1

        # Update any dangling step without an end
        last_index = quantity_agg.index[-1]
        if end_step_inds[-1] < last_index:
            quantity_agg[begin_step_ind:] += cycle_sum
        return quantity_agg

    @classmethod
    def from_maccor_file(cls, filename, include_eis=True, validate=False):
        """
        Method for ingestion of Maccor format files.

        Args:
            filename (str): file path for maccor format file.
            include_eis (bool): whether to include the eis spectrum
                in the ingestion procedure.
            validate (bool): whether to validate on instantiation.
        """
        with open(filename) as f:
            metadata_line = f.readline().strip()

        # Parse data
        data = pd.read_csv(filename, delimiter="\t", skiprows=1)
        data.rename(str.lower, axis="columns", inplace=True)
        data = data.astype(MACCOR_CONFIG["data_types"])
        data.rename(MACCOR_CONFIG["data_columns"], axis="columns", inplace=True)
        data["charge_capacity"] = cls.get_maccor_quantity_sum(
            data, "capacity", "charge"
        )
        data["discharge_capacity"] = cls.get_maccor_quantity_sum(
            data, "capacity", "discharge"
        )
        data["charge_energy"] = cls.get_maccor_quantity_sum(data, "energy", "charge")
        data["discharge_energy"] = cls.get_maccor_quantity_sum(
            data, "energy", "discharge"
        )

        # Parse metadata - kinda hackish way to do it, but it works
        metadata = parse_maccor_metadata(metadata_line)
        metadata = pd.DataFrame(metadata)
        _, channel_number = os.path.splitext(filename)
        metadata["channel_id"] = int(channel_number.replace(".", ""))
        metadata.rename(str.lower, axis="columns", inplace=True)
        metadata.rename(MACCOR_CONFIG["metadata_fields"], axis="columns", inplace=True)
        # Note the to_dict, which scrubs numpy typing
        metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}

        # Check for EIS files
        if include_eis:
            eis_pattern = ".*.".join(filename.rsplit(".", 1))
            all_eis_files = glob(eis_pattern)
            eis = EISpectrum.from_maccor_file(all_eis_files[0])
        else:
            eis = None

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(maccor_timestamp)

        return cls(data, metadata, eis, validate, filename=filename)

    def determine_structuring_parameters(
        self,
        v_range=None,
        resolution=1000,
        nominal_capacity=1.1,
        full_fast_charge=0.8,
        parameters_path="data-share/raw/parameters",
    ):
        """
        Method for determining what values to use to convert raw run into processed run

        Args:
            v_range ([float, float]): voltage range for interpolation
            resolution (int): resolution for interpolation
            nominal_capacity (float): nominal capacity for summary stats
            full_fast_charge (float): full fast charge for summary stats
            parameters_path (str): path to parameters file

        Returns:
            v_range ([float, float]): voltage range for interpolation
            resolution (int): resolution for interpolation
            nominal_capacity (float): nominal capacity for summary stats
            full_fast_charge (float): full fast charge for summary stats
            diagnostic_available (dict): dictionary of values to use for
                finding and using the diagnostic cycles

        """
        run_parameter, all_parameters = parameters_lookup.get_protocol_parameters(
            self.filename, parameters_path
        )
        # Logic for interpolation variables and diagnostic cycles
        diagnostic_available = False
        if run_parameter is not None:
            if {"capacity_nominal"}.issubset(run_parameter.columns.tolist()):
                nominal_capacity = run_parameter["capacity_nominal"].iloc[0]
            if {"discharge_cutoff_voltage", "charge_cutoff_voltage"}.issubset(run_parameter.columns):
                v_range = [
                    all_parameters["discharge_cutoff_voltage"].min(),
                    all_parameters["charge_cutoff_voltage"].max(),
                ]
            if {"diagnostic_type", "diagnostic_start_cycle", "diagnostic_interval"}.issubset(run_parameter.columns):
                if run_parameter["diagnostic_type"].iloc[0] == "HPPC+RPT":
                    hppc_rpt = ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"]
                    hppc_rpt_len = 5
                    initial_diagnostic_at = [1, 1 + run_parameter["diagnostic_start_cycle"].iloc[0] + 1 * hppc_rpt_len]
                    # Calculate the number of steps present for each cycle in the diagnostic as the pattern for
                    # the diagnostic. If this pattern of steps shows up at the end of the file, that indicates
                    # the presence of a final diagnostic
                    diag_0_pattern = [len(self.data[self.data.cycle_index == x].step_index.unique()) for x in
                                      range(initial_diagnostic_at[0], initial_diagnostic_at[0] + hppc_rpt_len)]
                    diag_1_pattern = [len(self.data[self.data.cycle_index == x].step_index.unique()) for x in
                                      range(initial_diagnostic_at[1], initial_diagnostic_at[1] + hppc_rpt_len)]
                    # Find the steps present in the reset cycles for the first and second diagnostic
                    diag_0_steps = set(self.data[self.data.cycle_index == initial_diagnostic_at[0]].step_index.unique())
                    diag_1_steps = set(self.data[self.data.cycle_index == initial_diagnostic_at[1]].step_index.unique())
                    diagnostic_starts_at = []
                    for cycle in self.data.cycle_index.unique():
                        steps_present = set(self.data[self.data.cycle_index == cycle].step_index.unique())
                        cycle_pattern = [len(self.data[self.data.cycle_index == x].step_index.unique()) for x in
                                         range(cycle, cycle + hppc_rpt_len)]
                        if steps_present == diag_0_steps or steps_present == diag_1_steps:
                            diagnostic_starts_at.append(cycle)
                        # Detect final diagnostic if present in the data
                        elif cycle >= (self.data.cycle_index.max() - hppc_rpt_len - 1) and \
                                (cycle_pattern == diag_0_pattern or cycle_pattern == diag_1_pattern):
                            diagnostic_starts_at.append(cycle)

                    diagnostic_available = {
                        "parameter_set": run_parameter["diagnostic_parameter_set"].iloc[0],
                        "cycle_type": hppc_rpt,
                        "length": hppc_rpt_len,
                        "diagnostic_starts_at": diagnostic_starts_at,
                    }

        return (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        )

    def to_processed_cycler_run(self):
        """
        Method for converting to ProcessedCyclerRun

        Returns:
            beep.structure.ProcessedCyclerRun: ProcessedCyclerRun
                that corresponds to processed RawCyclerRun

        """
        (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        ) = self.determine_structuring_parameters()

        return ProcessedCyclerRun.from_raw_cycler_run(
            self,
            v_range=v_range,
            resolution=resolution,
            nominal_capacity=nominal_capacity,
            full_fast_charge=full_fast_charge,
            diagnostic_available=diagnostic_available,
        )

    def save_numpy_binary(self, name):
        """
        Save RawCyclerRun as a numeric array and metadata json

        Args:
            name (str): file prefix, saves to a .npz and .json file
        """
        float_array = np.array(self.data[self.FLOAT_COLUMNS].astype(np.float64))
        int_array = np.array(self.data[self.INT_COLUMNS].astype(np.int64))
        np.savez_compressed(name, float_array=float_array, int_array=int_array)
        dumpfn(self.metadata, "{}.json".format(name))

    @classmethod
    def load_numpy_binary(cls, name):
        """
        Load RawCyclerRun from numeric binary file

        Args:
            name (str): str prefix for numeric and metadata files

        Returns:
            beep_structure.RawCyclerRun loaded from binary files

        """
        loaded = np.load("{}.npz".format(name))
        data = dict(zip(cls.FLOAT_COLUMNS, np.transpose(loaded["float_array"])))
        data.update(dict(zip(cls.INT_COLUMNS, np.transpose(loaded["int_array"]))))
        data = pd.DataFrame(data)
        metadata = loadfn("{}.json".format(name))
        return cls(data, metadata)


class ProcessedCyclerRun(MSONable):
    """
    Processed cycler run file which is intended to reflect the old format,
    for facile use in featurization, fitting, etc.  Note that this
    purposefully departs from PEP 8 guidelines to maintain certain
    naming tropes from the legacy code, which may be updated in the future.

    Attributes:
        barcode (str): barcode for the experiment.
        protocol (str): protocol for the experiment.
        channel_id (int): id for the channel for the experiment.
        summary (pandas.DataFrame): data of summary data for each cycle.
        cycles_interpolated (pandas.DataFrame): interpolated data for
            discharge over 2.8-3.5.
    """

    def __init__(
        self,
        barcode,
        protocol,
        channel_id,
        summary,
        cycles_interpolated,
        diagnostic_summary=None,
        diagnostic_interpolated=None,
    ):
        """
        For the most part, this invocation method will not be
        used directly, since this object will be invoked either
        from a RawCyclerRun object or a file.

        Args:
            barcode (string): barcode for the experiment
            protocol (string): protocol for the experiment
            channel_id (int): id for the channel for the experiment
            summary (pandas.DataFrame): data of summary data for each cycle
            cycles_interpolated (pandas.DataFrame): interpolated data for
                regular cycles
        """
        self.barcode = barcode
        self.protocol = protocol
        self.channel_id = channel_id
        self.summary = summary

        # We can drop this restriction later if we don't need it
        min_index = cycles_interpolated.cycle_index.min()
        if "step_type" in cycles_interpolated.columns:
            min_index_df = cycles_interpolated[
                (cycles_interpolated.cycle_index == min_index)
                & (cycles_interpolated.step_type == "discharge")
            ]
        else:
            min_index_df = cycles_interpolated[
                (cycles_interpolated.cycle_index == min_index)
            ]
        self.v_interpolated = min_index_df.voltage.values

        self.cycles_interpolated = cycles_interpolated
        self.diagnostic_summary = diagnostic_summary
        self.diagnostic_interpolated = diagnostic_interpolated

    @classmethod
    def from_raw_cycler_run(
        cls,
        raw_cycler_run,
        v_range=None,
        resolution=1000,
        diagnostic_resolution=500,
        nominal_capacity=1.1,
        full_fast_charge=0.8,
        diagnostic_available=False,
        charge_axis='charge_capacity',
        discharge_axis='voltage'
    ):
        """
        Method to invoke ProcessedCyclerRun from RawCyclerRun object

        Args:
            raw_cycler_run (beep.structure.RawCyclerRun): RawCyclerRun
                object to create ProcessedCyclerRun from.
            v_range ([int, int]): range of voltages for cycle interpolation.
            resolution (int): resolution for cycle interpolation.
            diagnostic_resolution (int): number of datapoints per step for
                interpolating diagnostic cycles.
            nominal_capacity (float): nominal capacity for summary stats.
            full_fast_charge (float): full fast charge for summary stats.
            diagnostic_available (dict): project metadata for processing
                diagnostic cycles correctly.
        """
        if diagnostic_available:
            diagnostic_summary = raw_cycler_run.get_diagnostic_summary(
                diagnostic_available
            )
            diagnostic_interpolated = raw_cycler_run.get_interpolated_diagnostic_cycles(
                diagnostic_available, diagnostic_resolution
            )
        else:
            diagnostic_summary = None
            diagnostic_interpolated = None

        cycles_interpolated = raw_cycler_run.get_interpolated_cycles(
            v_range=v_range,
            resolution=resolution,
            diagnostic_available=diagnostic_available,
            charge_axis=charge_axis,
            discharge_axis=discharge_axis
        )
        return cls(
            raw_cycler_run.metadata.get("barcode"),
            raw_cycler_run.metadata.get("protocol"),
            raw_cycler_run.metadata.get("channel_id"),
            raw_cycler_run.get_summary(
                nominal_capacity=nominal_capacity, full_fast_charge=full_fast_charge,
                diagnostic_available=diagnostic_available
            ),
            cycles_interpolated,
            diagnostic_summary,
            diagnostic_interpolated,
        )

    @classmethod
    def auto_load(cls, filename, validate=False):
        """
        Method for loading processed cycler run from raw cycler filename,
        processing it according to prescribed logic corresponding to the
        file name and or the file's contents.

        Args:
            filename (str): filename associated with the project
            validate (bool): whether or not to validate file

        Returns:
            beep.structure.ProcessedCyclerRun: ProcessedCyclerRun corresponding
                to the read and processed data from the filename

        """
        # Arbin files are via standard pipeline
        if re.match(FastCharge_CONFIG["file_pattern"], filename):
            raw = RawCyclerRun.from_arbin_file(filename, validate)
            return raw.to_processed_cycler_run()
        elif re.match(xTesladiag_CONFIG["file_pattern"], filename):
            raw = RawCyclerRun.from_maccor_file(filename, validate)
            return raw.to_processed_cycler_run()

        else:
            raise ValueError(
                "File pattern or contents of {} not recognized".format(filename)
            )

    def get_cycle_life(self, n_cycles_cutoff=40, threshold=0.8):
        """
        Calculate cycle life for capacity loss below a certain threshold

        Args:
            n_cycles_cutoff (int): cutoff for number of cycles to sample
                for the cycle life in order to use median method.
            threshold (float): fraction of capacity loss for which
                to find the cycle index.

        Returns:
            float: cycle life.
        """
        # discharge_capacity has a spike and  then increases slightly between \
        # 1-40 cycles, so let us use take median of 1st 40 cycles for max.

        # If n_cycles <  n_cycles_cutoff, do not use median method
        if len(self.summary) > n_cycles_cutoff:
            max_capacity = np.median(
                self.summary.discharge_capacity.iloc[0:n_cycles_cutoff]
            )
        else:
            max_capacity = 1.1

        # If capacity falls below 80% of initial capacity by end of run
        if self.summary.discharge_capacity.iloc[-1] / max_capacity <= threshold:
            cycle_life = self.summary[
                self.summary.discharge_capacity < threshold * max_capacity
            ].index[0]
        else:
            # Some cells do not degrade below the threshold (low degradation rate)
            cycle_life = len(self.summary) + 1

        return cycle_life

    def capacities_at_set_cycles(
        self, cycle_min=200, cycle_max=1800, cycle_interval=200
    ):
        """
        Get discharge capacity at constant intervals of 200 cycles

        Args:
            cycle_min (int): Cycle number to being forecasting capacity at
            cycle_max (int): Cycle number to end forecasting capacity at
            cycle_interval (int): Intervals for forecasts

        Returns:
            pandas.DataFrame:
        """
        discharge_capacities = pd.DataFrame(
            np.zeros((1, int((cycle_max - cycle_min) / cycle_interval)))
        )
        counter = 0
        cycle_indices = np.arange(cycle_min, cycle_max, cycle_interval)
        for cycle_index in cycle_indices:
            try:
                discharge_capacities[counter] = self.summary.discharge_capacity.iloc[
                    cycle_index
                ]
            except IndexError:
                pass
            counter = counter + 1
        discharge_capacities.columns = np.core.defchararray.add(
            "cycle_", cycle_indices.astype(str)
        )
        return discharge_capacities

    def cycles_to_reach_set_capacities(
        self, thresh_max_cap=0.98, thresh_min_cap=0.78, interval_cap=0.03
    ):
        """
        Get cycles to reach set threshold capacities.

        Args:
            thresh_max_cap (float): Upper bound on capacity to compute cycles at.
            thresh_min_cap (float): Lower bound on capacity to compute cycles at.
            interval_cap (float): Interval/step size.

        Returns:
            pandas.DataFrame:
        """
        threshold_list = np.around(
            np.arange(thresh_max_cap, thresh_min_cap, -interval_cap), 2
        )
        counter = 0
        cycles = pd.DataFrame(np.zeros((1, len(threshold_list))))
        for threshold in threshold_list:
            cycles[counter] = self.get_cycle_life(threshold=threshold)
            counter = counter + 1
        cycles.columns = np.core.defchararray.add(
            "capacity_", threshold_list.astype(str)
        )
        return cycles

    def as_dict(self):
        """
        Method for dictionary serialization.

        Returns:
            dict: corresponding to dictionary for serialization.

        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "barcode": self.barcode,
            "protocol": self.protocol,
            "channel_id": self.channel_id,
            "summary": self.summary.to_dict("list"),
            "cycles_interpolated": self.cycles_interpolated.to_dict("list"),
            "diagnostic_summary": self.diagnostic_summary.to_dict("list")
            if self.diagnostic_summary is not None
            else None,
            "diagnostic_interpolated": self.diagnostic_interpolated.to_dict("list")
            if self.diagnostic_interpolated is not None
            else None,
        }

    @classmethod
    def from_dict(cls, d):
        """

        Args:
            d (dict): dictionary represenation.

        Returns:
            beep.structure.ProcessedCyclerRun: deserialized ProcessedCyclerRun.
        """
        """MSONable deserialization method"""
        for obj, dtype_dict in STRUCTURE_DTYPES.items():
            for column, dtype in dtype_dict.items():
                if d.get(obj) is not None:
                    if d[obj].get(column) is not None:
                        d[obj][column] = pd.Series(d[obj][column], dtype=dtype)

        d["cycles_interpolated"] = pd.DataFrame(d["cycles_interpolated"])
        d["summary"] = pd.DataFrame(d["summary"])
        d["diagnostic_summary"] = pd.DataFrame(d.get("diagnostic_summary"))
        d["diagnostic_interpolated"] = pd.DataFrame(d.get("diagnostic_interpolated"))
        return cls(**d)

    METADATA_ATTRIBUTE_ORDER = ["barcode", "protocol", "channel_id"]
    SUMMARY_COLUMN_ORDER = [
        "discharge_capacity",
        "charge_capacity",
        "dc_internal_resistance",
        "temperature_maximum",
        "temperature_average",
        "temperature_minimum",
        "charge_duration",
    ]
    CYCLES_INTERPOLATED_COLUMN_ORDER = [
        "cycle_index",
        "voltage",
        "current",
        "internal_resistance",
        "charge_capacity",
        "discharge_capacity",
        "temperature",
    ]

    def save_numpy_binary(self, name):
        """
        Save ProcessedCyclerRun as numpy binary.

        Args:
            name (str): filename to save numpy binary as.
        """
        meta_array = np.array(
            [getattr(self, mattribute) for mattribute in self.METADATA_ATTRIBUTE_ORDER]
        )
        summary_array = self.summary[self.SUMMARY_COLUMN_ORDER].values
        cycles_interpolated_array = self.cycles_interpolated[
            self.CYCLES_INTERPOLATED_COLUMN_ORDER
        ].values
        np.savez_compressed(
            name,
            meta=meta_array,
            summary=summary_array,
            cycles_interpolated=cycles_interpolated_array,
        )

    @classmethod
    def load_numpy_binary(cls, name):
        """
        Class method to load ProcessedCyclerRun from numpy binary.

        Args:
            name (str): filename for numpy binary to be loaded.

        Returns:
            beep.structure.ProcessedCyclerRun: loaded from numpy binary

        """
        if not name.endswith(".npz"):
            name += ".npz"
        data = np.load(name, allow_pickle=True)
        meta_kwargs = dict(zip(cls.METADATA_ATTRIBUTE_ORDER, data["meta"]))

        # Load summary into DataFrame
        summary = pd.DataFrame(
            dict(zip(cls.SUMMARY_COLUMN_ORDER, data["summary"].transpose()))
        )

        # Load cycles_interpolated into DataFrame
        cycles_interpolated = pd.DataFrame(
            dict(
                zip(
                    cls.CYCLES_INTERPOLATED_COLUMN_ORDER,
                    data["cycles_interpolated"].transpose(),
                )
            )
        )
        return cls(
            summary=summary, cycles_interpolated=cycles_interpolated, **meta_kwargs
        )


class EISpectrum(MSONable):
    """
    Class describing an Electrochemical Impedance Spectrum
    """

    def __init__(self, data, metadata):
        """

        Args:
            data:
            metadata:
        """
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_csv(cls, filename):
        """

        Args:
            filename(str): path to data file.

        Returns:
            beep.structure.EISpectrum: EISpectrum object representation of
                data.
        """
        raise NotImplementedError("from_csv not implemented for EISpectrum")

    @classmethod
    def from_maccor_file(cls, filename):
        """
        EISpectrum from Maccor file.

        Args:
            filename (str): file path to data.

        Returns:
            beep.strucure.EISpectrum: EISpectrum object representation of
                data.
        """
        with open(filename) as f:
            lines = f.readlines()
        # Parse freq sweep, method, and output filename
        freq_sweep = lines[1].split("Frequency Sweep:")[1].strip()
        freq_sweep = freq_sweep.replace("Circut", "Circuit")
        method = lines[2]
        filename = lines[3]

        # Parse start datetime and reformat in isoformat
        start = lines[6].split("Start Date:")[1].strip()
        date, time = start.split("Start Time:")
        start = ",".join([date.strip(), time.strip()])
        start = datetime.strptime(start, "%A, %B %d, %Y,%H:%M")
        start = start.isoformat()

        line_8 = lines[8].split()

        # Construct metadata dictionary
        metadata = {
            "frequency_sweep": freq_sweep,
            "method": method,
            "filename": filename,
            "start": start,
            "line_8": line_8,
        }

        data = "\n".join(lines[10:])
        data = pd.read_csv(StringIO(data), delimiter="\t")

        return cls(data=data, metadata=metadata)

    def as_dict(self):
        """
        Method for serialization as dictionary

        Returns:
            ({}): dictionary representation of this object

        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": self.data.to_dict("list"),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        """
        Method of invocation from dictionary

        Args:
            d ({}): dictionary from which to invoke

        Returns:
            (EISpectrum): object invoked from dictionary

        """
        data = pd.DataFrame(d["data"])
        data = data.sort_index()
        metadata = pd.DataFrame(d["metadata"])
        return cls(data, metadata)


def determine_whether_step_is_waveform_discharge(step_dataframe):
    """
    Helper function for driving profiles to determine whether a given dataframe corresponding
    to a single cycle_index/step is a waveform discharge.

    Args:
         step_dataframe (pandas.DataFrame): dataframe to determine whether waveform step is present
    """

    # Check for waveform in maccor
    if len([col for col in step_dataframe.columns if '_wf_' in col]):
        return (determine_whether_step_is_discharging(step_dataframe)) &\
               ((step_dataframe['_wf_chg_cap'].notna().any()) |
                (step_dataframe['_wf_dis_cap'].notna().any()))
    elif not np.round(step_dataframe.voltage, VOLTAGE_RESOLUTION).is_monotonic:
        # This is a placeholder logic for arbin waveform detection
        # This fails for some arbin files that nominally have a CC-CV step.
        # e.g. 2017-12-04_4_65C-69per_6C_CH29.csv
        # TODO: survey more files and include additional heuristics/logic based on the size of
        # and frequency of non-monotonicities to determine whether step is actually a waveform.

        # All non-maccor files will evaluate to False by default for now
        return False
    else:
        return False


def determine_whether_step_is_waveform_charge(step_dataframe):
    """
    Helper function for driving profiles to determine whether a given dataframe corresponding
    to a single cycle_index/step is a waveform charge.

    Args:
         step_dataframe (pandas.DataFrame): dataframe to determine whether waveform step is present
    """

    # Check for waveform in maccor
    if len([col for col in step_dataframe.columns if '_wf_' in col]):
        return (determine_whether_step_is_charging(step_dataframe)) & \
               ((step_dataframe['_wf_chg_cap'].notna().any()) |
                (step_dataframe['_wf_dis_cap'].notna().any()))
    elif not np.round(step_dataframe.voltage, VOLTAGE_RESOLUTION).is_monotonic:
        # This is a placeholder logic for arbin waveform detection
        # This fails for some arbin files that nominally have a CC-CV step.
        # e.g. 2017-12-04_4_65C-69per_6C_CH29.csv
        # TODO: survey more files and include additional heuristics/logic based on the size of
        # and frequency of non-monotonicities to determine whether step is actually a waveform.

        # All non-maccor files will evaluate to False by default for now
        return False
    else:
        return False


def determine_whether_step_is_discharging(step_dataframe):
    """
    Helper function to determine whether a given dataframe corresponding
    to a single cycle_index/step is charging or discharging, only intended
    to be used with a dataframe for single step/cycle

    Args:
         step_dataframe (pandas.DataFrame): dataframe to determine whether
         charging or discharging
    """
    cap = step_dataframe[["charge_capacity", "discharge_capacity"]]
    return cap.diff(axis=0).mean(axis=0).diff().iloc[-1] > 0


def determine_whether_step_is_charging(step_dataframe):
    """
    Helper function to determine whether a given dataframe corresponding
    to a single cycle_index/step is charging or discharging, only intended
    to be used with a dataframe for single step/cycle

    Args:
         step_dataframe (pandas.DataFrame): dataframe to determine whether
         charging or discharging
    """
    cap = step_dataframe[["charge_capacity", "discharge_capacity"]]
    return cap.diff(axis=0).mean(axis=0).diff().iloc[-1] < 0


def get_interpolated_data(
    dataframe, field_name="voltage", field_range=None, columns=None, resolution=1000
):
    """
    General method for creating uniform (i. e. same # of data points)
    dataframe using an by interpolation of supplied columns on the
    specified field, assumes input field data is monotonic

    Args:
        dataframe (pandas.DataFrame): dataframe to create interpolate
        field_name (str): column name to use as the dependent interpolation variable
        field_range (list): list of two values to use as endpoints, if None,
            range is the min/max of the dataframe field_name values
        columns (list): list of column names to provide interpolated values for,
            default value of None indicates all columns should be interpolated
        resolution (int): number of data points to sample in the uniform
            cycles, defaults to 1000

    Returns:
        pandas.DataFrame: DataFrame of interpolated values
    """
    columns = columns or dataframe.columns
    columns = list(set(columns) | {field_name})

    df = dataframe.reindex(columns=columns)
    field_range = field_range or [df[field_name].iloc[0], df[field_name].iloc[-1]]
    # If interpolating on datetime, change column to datetime series and
    # use date_range to create interpolating vector
    if field_name == "date_time_iso":
        df["date_time_iso"] = pd.to_datetime(df["date_time_iso"])
        interp_x = pd.date_range(
            start=df[field_name].iloc[0],
            end=df[field_name].iloc[-1],
            periods=resolution,
        )
    else:
        interp_x = np.linspace(*field_range, resolution)
    interpolated_df = pd.DataFrame({field_name: interp_x, "interpolated": True})

    df["interpolated"] = False

    # Merge interpolated and uninterpolated DFs to use pandas interpolation
    interpolated_df = interpolated_df.merge(df, how="outer", on=field_name, sort=True)
    interpolated_df = interpolated_df.set_index(field_name)
    interpolated_df = interpolated_df.interpolate("slinear")

    # Filter for only interpolated values
    interpolated_df[["interpolated_x"]] = interpolated_df[["interpolated_x"]].fillna(
        False
    )
    interpolated_df = interpolated_df[interpolated_df["interpolated_x"]]
    interpolated_df = interpolated_df.drop(["interpolated_x", "interpolated_y"], axis=1)
    interpolated_df = interpolated_df.reset_index()
    # Remove duplicates
    interpolated_df = interpolated_df[~interpolated_df[field_name].duplicated()]
    return interpolated_df


def diagnostic_function(df, column):
    """

    Args:
        df (pandas.DataFrame): A dataframe.
        column (str): A column name.

    Returns:
        float: median value of column.
    """
    value = df[column].agg("median")
    return value


def parse_maccor_metadata(metadata_string):
    """
    Parses maccor metadata string, which is annoyingly inconsistent.
    Basically just splits the string by a set of fields and creates
    a dictionary of pairs of fields and values with colons scrubbed
    from fields.

    Args:
        metadata_string (str): string corresponding to maccor metadata.

    Returns:
        dict: dictionary of metadata fields and values.

    """
    metadata_fields = [
        "Today's Date",
        "Date of Test:",
        "Filename:",
        "Procedure:",
        "Comment/Barcode:",
    ]
    metadata_values = split_string_by_fields(metadata_string, metadata_fields)
    metadata = {
        k.replace(":", ""): [v.strip()]
        for k, v in zip(metadata_fields, metadata_values)
    }
    return metadata


def split_string_by_fields(string, fields):
    """
    Helper function to split a string by a set of ordered strings,
    primarily used for Maccor metadata parsing.

    >>>split_string_by_fields("first name: Joey  last name Montoya",
    >>>                       ["first name:", "last name"])
    ["Joey", "Montoya"]

    Args:
        string (str): string input to be split
        fields (list): list of fields to split input string by.

    Returns:
        list: substrings corresponding to the split input strings.

    """
    # A bit brittle, there's probably something more clever with recursion
    substrings = []
    init, leftovers = string.split(fields[0])
    for field in fields[1:]:
        init, leftovers = leftovers.split(field)
        substrings.append(init)
    substrings.append(leftovers)
    return substrings


def add_file_prefix_to_path(path, prefix):
    """
    Helper function to add file prefix to path.

    Args:
        path (str): full path to file.
        prefix (str): prefix for file.

    Returns:
        str: path with prefix appended to filename.

    """
    split_path = list(os.path.split(path))
    split_path[-1] = prefix + split_path[-1]
    return os.path.join(*split_path)


def get_max_paused_over_threshold(group, paused_threshold=3600):
    """
    Evaluate a raw cycling dataframe to determine if there is a pause in cycling.
    The method looks at the time difference between each row and if this value
    exceeds a threshold, it returns that length of time in seconds. Otherwise it
    returns 0

    Args:
        group (pd.DataFrame): cycling dataframe with date_time_iso column
        paused_threshold (int): gap in seconds to classify as a pause in cycling

    Returns:
        float: number of seconds that test was paused

    """
    date_time_obj = pd.to_datetime(group["date_time_iso"])
    date_time_float = [
        time.mktime(t.timetuple()) if t is not pd.NaT else float("nan")
        for t in date_time_obj
    ]
    date_time_float = pd.Series(date_time_float)
    if date_time_float.diff().max() > paused_threshold:
        max_paused_duration = date_time_float.diff().max()
    else:
        max_paused_duration = 0
    return max_paused_duration


def maccor_timestamp(x):
    """
    Helper function with exception handling for cases where the
    maccor cycler mis-prints the datetime stamp for the row. This
    happens when data is being recorded rapidly as the date switches over
    ie. between 10/21/2019 23:59:59 and 10/22/2019 00:00:00.

    Args:
        x (str): The datetime string for maccor in format '%m/%d/%Y %H:%M:%S'

    Returns:
        datetime.Datetime: Datetime object in iso format (daylight savings aware)

    """
    pacific = pytz.timezone("US/Pacific")
    utc = pytz.timezone("UTC")
    try:
        iso = (
            pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"), is_dst=True)
            .astimezone(utc)
            .isoformat()
        )
    except ValueError:
        x = x + " 00:00:00"
        iso = (
            pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"), is_dst=True)
            .astimezone(utc)
            .isoformat()
        )
    return iso


def neware_step_time(x):
    """
    Helper function to convert the step time format from Neware h:min:s.ms into
    decimal seconds

    Args:
        x (str): The datetime string for neware in format 'h:min:s.ms'

    Returns:
        float: The time in seconds

    """
    time_list = x.split(":")
    time = (
        3600 * float(time_list[-3]) + 60 * float(time_list[-2]) + float(time_list[-1])
    )
    return time


def process_file_list_from_json(file_list_json, processed_dir="data-share/structure/"):
    """
    Function to take a json filename corresponding to a data structure
    with a 'file_list' and a 'validity' attribute, process each file
    with a corresponding True validity, dump the processed file into
    a predetermined directory, and return a jsonable dict of processed
    cycler run file locations

    Args:
        file_list_json (str): json string or json filename corresponding
            to a dictionary with a file_list and validity attribute,
            if this string ends with ".json", a json file is assumed
            and loaded, otherwise interpreted as a json string.
        processed_dir (str): location for processed cycler run output
            files to be placed.

    Returns:
        str: json string of processed files (with key "processed_file_list").
            Note that this list contains None values for every file that
            had a corresponding False in the validity list.

    """
    # Get file list and validity from json, if ends with .json,
    # assume it's a file, if not assume it's a json string
    if file_list_json.endswith(".json"):
        file_list_data = loadfn(file_list_json)
    else:
        file_list_data = json.loads(file_list_json)

    # Setup workflow
    outputs = WorkflowOutputs()

    # Prepend optional root to output directory
    processed_dir = os.path.join(
        os.environ.get("BEEP_PROCESSING_DIR", "/"), processed_dir
    )

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    file_list = file_list_data["file_list"]
    validities = file_list_data["validity"]
    run_ids = file_list_data["run_list"]
    processed_file_list = []
    processed_run_list = []
    processed_result_list = []
    processed_message_list = []
    invalid_file_list = []
    for filename, validity, run_id in zip(file_list, validities, run_ids):
        logger.info("run_id=%s structuring=%s", str(run_id), filename, extra=s)
        if validity == "valid":
            # Process raw cycler run and dump to file
            raw_cycler_run = RawCyclerRun.from_file(filename)
            processed_cycler_run = raw_cycler_run.to_processed_cycler_run()
            new_filename, ext = os.path.splitext(os.path.basename(filename))
            new_filename = new_filename + ".json"
            new_filename = add_suffix_to_filename(new_filename, "_structure")
            processed_cycler_run_loc = os.path.join(processed_dir, new_filename)
            processed_cycler_run_loc = os.path.abspath(processed_cycler_run_loc)
            dumpfn(processed_cycler_run, processed_cycler_run_loc)

            # Append file loc to list to be returned
            processed_file_list.append(processed_cycler_run_loc)
            processed_run_list.append(run_id)
            processed_result_list.append("success")
            processed_message_list.append({"comment": "", "error": ""})

        else:
            invalid_file_list.append(filename)

    output_json = {
        "file_list": processed_file_list,
        "run_list": processed_run_list,
        "result_list": processed_result_list,
        "message_list": processed_message_list,
        "invalid_file_list": invalid_file_list,
    }

    # Workflow outputs
    file_list_size = len(output_json["file_list"])
    if file_list_size > 1 or file_list_size == 0:
        logger.warning("{file_list_size} files being validated, should be 1")

    output_data = {
        "filename": output_json["file_list"][0],
        "run_id": output_json["run_list"][0],
        "result": output_json["result_list"][0],
    }

    outputs.put_workflow_outputs(output_data, "structuring")

    # Return jsonable file list
    return json.dumps(output_json)


def main():
    """
    Main function of this module, takes in arguments of an input
    and output filename and uses the input file to create a
    structured data output for analysis/ML processing.
    """
    logger.info("starting", extra=s)
    logger.info("Running version=%s", __version__, extra=s)
    try:
        args = docopt(__doc__)
        input_json = args["INPUT_JSON"]
        print(process_file_list_from_json(input_json))
    except Exception as e:
        logger.error(str(e), extra=s)
        raise e
    logger.info("finish", extra=s)
    return None


if __name__ == "__main__":
    main()
