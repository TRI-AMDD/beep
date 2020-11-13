import abc
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
from dataclasses import dataclass

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


# todo: ALEXTODO this should be set in the class, and should not be overwritten
VOLTAGE_RESOLUTION = 3


class BEEPDatapath(abc.ABC):

    def __init__(self, raw_data, metadata, paths=None):
        self.raw_data = raw_data
        self.metadata = metadata
        self.paths = paths if paths else {"raw": None}

        self.structured_summary = None     # equivalent of PCR.summary
        self.structured_data = None        # equivalent of PCR.cycles_interpolated
        self.diagnostic_data = None        # equivalent of PCR.diagnostic_interpolated
        self.diagnostic_summary = None     # same name as in PCR

    # @property
    # @abc.abstractmethod
    # def schema(self):
    #     raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_file(cls, path):
        raise NotImplementedError

    def validate(self):
        pass

    def structure(self,
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

        Args:
            v_range ([int, int]): range of voltages for cycle interpolation.
            resolution (int): resolution for cycle interpolation.
            diagnostic_resolution (int): number of datapoints per step for
                interpolating diagnostic cycles.
            nominal_capacity (float): nominal capacity for summary stats.
            full_fast_charge (float): full fast charge for summary stats.
            diagnostic_available (dict): project metadata for processing
                diagnostic cycles correctly.
        pass
        """

        logger.info("Beginning structuring...")

        if diagnostic_available:
            self.diagnostic_summary = self.summarize_diagnostic(
                diagnostic_available
            )
            self.diagnostic_data = self.interpolate_diagnostic_cycles(
                diagnostic_available, diagnostic_resolution
            )

        self.structured_data = self.interpolate_cycles(
            v_range=v_range,
            resolution=resolution,
            diagnostic_available=diagnostic_available,
            charge_axis=charge_axis,
            discharge_axis=discharge_axis
        )

        self.structured_summary = self.summarize_cycles(
            nominal_capacity=nominal_capacity,
            full_fast_charge=full_fast_charge,
            diagnostic_available=diagnostic_available
        )

    # todo: ALEXTODO check docstring
    def interpolate_step(
            self,
            v_range,
            resolution,
            step_type="discharge",
            reg_cycles=None,
            axis="voltage",
            desc=None
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
            step_filter = step_is_dchg
        elif step_type == "charge":
            step_filter = step_is_chg
        else:
            raise ValueError("{} is not a recognized step type")
        incl_columns = [
            "test_time",
            "voltage",
            "current",
            "charge_capacity",
            "discharge_capacity",
            "internal_resistance",
            "temperature",
        ]
        all_dfs = []
        cycle_indices = self.raw_data.cycle_index.unique()
        cycle_indices = [c for c in cycle_indices if c in reg_cycles]
        cycle_indices.sort()

        for cycle_index in tqdm(cycle_indices, desc=desc):
            # Use a cycle_index mask instead of a global groupby to save memory
            new_df = (
                self.raw_data.loc[self.raw_data["cycle_index"] == cycle_index]
                    .groupby("step_index")
                    .filter(step_filter)
            )
            if new_df.size == 0:
                continue

            if axis in ["charge_capacity", "discharge_capacity"]:
                axis_range = [self.raw_data[axis].min(),
                              self.raw_data[axis].max()]
                new_df = interpolate_df(
                    new_df,
                    axis,
                    field_range=axis_range,
                    columns=incl_columns,
                    resolution=resolution,
                )
            elif axis == "test_time":
                axis_range = [new_df[axis].min(), new_df[axis].max()]
                new_df = interpolate_df(
                    new_df,
                    axis,
                    field_range=axis_range,
                    columns=incl_columns,
                    resolution=resolution,
                )
            elif axis == "voltage":
                new_df = interpolate_df(
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

    def interpolate_cycles(
            self,
            v_range=None,
            resolution=1000,
            diagnostic_available=None,
            charge_axis='charge_capacity',
            discharge_axis='voltage'
    ):
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
                        if i <= self.raw_data.cycle_index.max()
                    ]
                )
            )
            reg_cycles = [
                i for i in self.raw_data.cycle_index.unique() if
                i not in diag_cycles
            ]
        else:
            reg_cycles = [i for i in self.raw_data.cycle_index.unique()]

        v_range = v_range or [2.8, 3.5]

        # If any regular cycle contains a waveform step, interpolate on test_time.
        if self.raw_data[self.raw_data.cycle_index.isin(reg_cycles)]. \
                groupby(["cycle_index", "step_index"]). \
                apply(step_is_waveform_dchg).any():
            discharge_axis = 'test_time'

        if self.raw_data[self.raw_data.cycle_index.isin(reg_cycles)]. \
                groupby(["cycle_index", "step_index"]). \
                apply(step_is_waveform_chg).any():
            charge_axis = 'test_time'

        interpolated_discharge = self.interpolate_step(
            v_range,
            resolution,
            step_type="discharge",
            reg_cycles=reg_cycles,
            axis=discharge_axis,
            desc="interpolating discharge cycles"
        )
        interpolated_charge = self.interpolate_step(
            v_range,
            resolution,
            step_type="charge",
            reg_cycles=reg_cycles,
            axis=charge_axis,
            desc="interpolating charge cycles"
        )
        result = pd.concat(
            [interpolated_discharge, interpolated_charge], ignore_index=True
        )

        return result.astype(STRUCTURE_DTYPES["cycles_interpolated"])

    # equivalent of get_summary
    def summarize_cycles(
            self,
            diagnostic_available=False,
            nominal_capacity=1.1,
            full_fast_charge=0.8,
            cycle_complete_discharge_ratio=0.97,
            cycle_complete_vmin=3.3,
            cycle_complete_vmax=3.3,
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
                        if i <= self.raw_data.cycle_index.max()
                    ]
                )
            )
            reg_cycles_at = [
                i for i in self.raw_data.cycle_index.unique() if
                i not in diag_cycles
            ]
        else:
            reg_cycles_at = [i for i in self.raw_data.cycle_index.unique()]

        summary = self.raw_data.groupby("cycle_index").agg(
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
        summary["charge_throughput"] = summary.charge_capacity.cumsum()
        summary["energy_throughput"] = summary.charge_energy.cumsum()

        # This method for computing charge start and end times implicitly
        # assumes that a cycle starts with a charge step and is then followed
        # by discharge step.
        charge_start_time = \
            self.raw_data.groupby("cycle_index", as_index=False)[
                "date_time_iso"
            ].agg("first")
        charge_finish_time = (
            self.raw_data[
                self.raw_data.charge_capacity >= nominal_capacity * full_fast_charge]
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
        summary["charge_duration"] = np.round(
            time_diff / np.timedelta64(1, "s"), 2)

        # Compute time since start of cycle in minutes. This comes handy
        # for featurizing time-temperature integral
        self.raw_data["time_since_cycle_start"] = pd.to_datetime(
            self.raw_data["date_time_iso"]
        ) - pd.to_datetime(
            self.raw_data.groupby("cycle_index")["date_time_iso"].transform(
                "first")
        )
        self.raw_data["time_since_cycle_start"] = (self.raw_data[
                                                       "time_since_cycle_start"] / np.timedelta64(
            1, "s")) / 60

        # Group by cycle index and integrate time-temperature
        # using a lambda function.
        summary["time_temperature_integrated"] = self.raw_data.groupby(
            "cycle_index").apply(
            lambda g: integrate.trapz(g.temperature, x=g.time_since_cycle_start)
        )

        # Drop the time since cycle start column
        self.raw_data.drop(columns=["time_since_cycle_start"])

        # Determine if any of the cycles has been paused
        summary["paused"] = self.raw_data.groupby("cycle_index").apply(
            get_max_paused_over_threshold)

        summary = summary.astype(STRUCTURE_DTYPES["summary"])

        last_voltage = self.raw_data.loc[
            self.raw_data["cycle_index"] == self.raw_data["cycle_index"].max()
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

    # equivalent of get_interpolated_diagnostic_cycles
    def interpolate_diagnostic_cycles(
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
        project_name_list = parameters_lookup.get_project_sequence(self.paths["raw"])
        diag_path = os.path.join(MODULE_DIR, "procedure_templates")
        v_range = parameters_lookup.get_diagnostic_parameters(
            diagnostic_available, diag_path, project_name_list[0]
        )

        # Determine the cycles and types of the diagnostic cycles
        max_cycle = self.raw_data.cycle_index.max()
        starts_at = [
            i for i in diagnostic_available["diagnostic_starts_at"] if
            i <= max_cycle
        ]
        diag_cycles_at = list(
            itertools.chain.from_iterable(
                [range(i, i + diagnostic_available["length"]) for i in
                 starts_at]
            )
        )
        # Duplicate cycle type list end to end for each starting index
        diag_cycle_type = diagnostic_available["cycle_type"] * len(starts_at)
        if not len(diag_cycles_at) == len(diag_cycle_type):
            errmsg = (
                "Diagnostic cycles, {}, and diagnostic cycle types, "
                "{}, are unequal lengths".format(diag_cycles_at,
                                                 diag_cycle_type)
            )
            raise ValueError(errmsg)

        diag_data = self.raw_data[self.raw_data["cycle_index"].isin(diag_cycles_at)]

        # Convert date_time_iso field into pd.datetime object
        diag_data.loc[:, "date_time_iso"] = pd.to_datetime(
            diag_data["date_time_iso"])

        # Convert datetime into seconds to allow interpolation of time
        diag_data.loc[:, "datetime_seconds"] = [
            time.mktime(t.timetuple()) if t is not pd.NaT else float("nan")
            for t in diag_data["date_time_iso"]
        ]

        # Counter to ensure non-contiguous repeats of step_index
        # within same cycle_index are grouped separately
        diag_data.loc[:, "step_index_counter"] = 0

        for cycle_index in diag_cycles_at:
            indices = diag_data.loc[diag_data.cycle_index == cycle_index].index
            step_index_list = diag_data.step_index.loc[indices]
            diag_data.loc[indices, "step_index_counter"] = step_index_list.ne(
                step_index_list.shift()
            ).cumsum()

        group = diag_data.groupby(
            ["cycle_index", "step_index", "step_index_counter"])
        incl_columns = [
            "current",
            "charge_capacity",
            "discharge_capacity",
            "charge_energy",
            "discharge_energy",
            "internal_resistance",
            "temperature",
            "datetime_seconds",
            "test_time",
        ]

        diag_dict = {}
        for cycle in diag_data.cycle_index.unique():
            diag_dict.update({cycle: None})
            steps = diag_data[
                diag_data.cycle_index == cycle].step_index.unique()
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
                new_df = interpolate_df(
                    df,
                    field_name="voltage",
                    field_range=v_hppc_step,
                    columns=incl_columns,
                    resolution=hppc_resolution,
                )
            else:
                new_df = interpolate_df(
                    df,
                    field_name="voltage",
                    field_range=v_range,
                    columns=incl_columns,
                    resolution=resolution,
                )

            # Convert interpolated time in seconds back to datetime
            new_df["date_time_iso"] = [
                datetime.utcfromtimestamp(t).isoformat() if ~np.isnan(t) else t
                for t in new_df["datetime_seconds"]
            ]
            new_df = new_df.drop(columns="datetime_seconds")

            new_df["cycle_index"] = cycle_index
            new_df["cycle_type"] = diag_cycle_type[
                diag_cycles_at.index(cycle_index)]
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


    def summarize_diagnostic(self, diagnostic_available):
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

        max_cycle = self.raw_data.cycle_index.max()
        starts_at = [
            i for i in diagnostic_available["diagnostic_starts_at"] if i <= max_cycle
        ]
        diag_cycles_at = list(
            itertools.chain.from_iterable(
                [list(range(i, i + diagnostic_available["length"])) for i in starts_at]
            )
        )
        diag_summary = self.raw_data.groupby("cycle_index").agg(
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
        diag_summary = diag_summary[diag_summary.index.isin(diag_cycles_at)]

        diag_summary["coulombic_efficiency"] = (
            diag_summary["discharge_capacity"] / diag_summary["charge_capacity"]
        )
        diag_summary["paused"] = self.raw_data.groupby("cycle_index").apply(
            get_max_paused_over_threshold
        )

        diag_summary.reset_index(drop=True, inplace=True)

        diag_summary["cycle_type"] = pd.Series(
            diagnostic_available["cycle_type"] * len(starts_at)
        )

        diag_summary = diag_summary.astype(STRUCTURE_DTYPES["diagnostic_summary"])

        return diag_summary


class ArbinDatapath(BEEPDatapath):

    @classmethod
    def from_file(cls, path, metadata_path=None):
        """
        """
        data = pd.read_csv(path)
        data.rename(str.lower, axis="columns", inplace=True)

        for column, dtype in ARBIN_CONFIG["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(ARBIN_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata_path = metadata_path if metadata_path else path.replace(".csv",
                                                                         "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(ARBIN_CONFIG["metadata_fields"], axis="columns",
                            inplace=True)
            # Note the to_dict, which scrubs numpy typing
            metadata = {col: item[0] for col, item in
                        metadata.to_dict("list").items()}
        else:
            logger.warning(f"No associated metadata file for Arbin: "
                           f"'{metadata_path}'. No metadata loaded.")
            metadata = {}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(
            lambda x: datetime.utcfromtimestamp(x).replace(
                tzinfo=pytz.UTC).isoformat()
        )

        paths = {
            "raw_data": path,
            "metadata": metadata_path if metadata else None
        }

        return cls(data, metadata, paths)


def interpolate_df(
        dataframe,
        field_name="voltage",
        field_range=None,
        columns=None,
        resolution=1000
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

    df = dataframe.loc[:, columns]
    field_range = field_range or [df[field_name].iloc[0],
                                  df[field_name].iloc[-1]]
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
    interpolated_df = interpolated_df.merge(df, how="outer", on=field_name,
                                            sort=True)
    interpolated_df = interpolated_df.set_index(field_name)
    interpolated_df = interpolated_df.interpolate("slinear")

    # Filter for only interpolated values
    interpolated_df[["interpolated_x"]] = interpolated_df[
        ["interpolated_x"]].fillna(
        False
    )
    interpolated_df = interpolated_df[interpolated_df["interpolated_x"]]
    interpolated_df = interpolated_df.drop(["interpolated_x", "interpolated_y"],
                                           axis=1)
    interpolated_df = interpolated_df.reset_index()
    # Remove duplicates
    interpolated_df = interpolated_df[~interpolated_df[field_name].duplicated()]
    return interpolated_df


# todo: ALEXTODO: need docstring
def step_is_chg_state(step_df, chg):
    """
    Helper function to determine whether a given dataframe corresponding
    to a single cycle_index/step is charging or discharging, only intended
    to be used with a dataframe for single step/cycle

    Args:
         step_dataframe (pandas.DataFrame): dataframe to determine whether
         charging or discharging
    """
    cap = step_df[["charge_capacity", "discharge_capacity"]]
    cap = cap.diff(axis=0).mean(axis=0).diff().iloc[-1]

    if chg:  # Charging
        return cap < 0
    else:  # Discharging
        return cap > 0


def step_is_dchg(step_df):
    return step_is_chg_state(step_df, False)


def step_is_chg(step_df):
    return step_is_chg_state(step_df, True)


# todo: ALEXTODO: needs tests
# todo: Also, may only be applicable to maccor
def step_is_waveform(step_df, chg_filter):
    """
    Helper function for driving profiles to determine whether a given dataframe corresponding
    to a single cycle_index/step is a waveform discharge.

    Args:
         step_df (pandas.DataFrame): dataframe to determine whether waveform step is present
    """

    # Check for waveform in maccor
    if len([col for col in step_df.columns if '_wf_' in col]):
        return (chg_filter(step_df)) & \
               ((step_df['_wf_chg_cap'].notna().any()) |
                (step_df['_wf_dis_cap'].notna().any()))
    elif not np.round(step_df.voltage, VOLTAGE_RESOLUTION).is_monotonic:
        # This is a placeholder logic for arbin waveform detection
        # This fails for some arbin files that nominally have a CC-CV step.
        # e.g. 2017-12-04_4_65C-69per_6C_CH29.csv
        # TODO: survey more files and include additional heuristics/logic based on the size of
        # and frequency of non-monotonicities to determine whether step is actually a waveform.

        # All non-maccor files will evaluate to False by default for now
        return False
    else:
        return False


def step_is_waveform_dchg(step_df):
    return step_is_waveform(step_df, step_is_dchg)


def step_is_waveform_chg(step_df):
    return step_is_waveform(step_df, step_is_chg)


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
    date_time_objs = pd.to_datetime(group["date_time_iso"])
    date_time_float = [
        time.mktime(t.timetuple()) if t is not pd.NaT else float("nan")
        for t in date_time_objs
    ]
    date_time_float = pd.Series(date_time_float)
    if date_time_float.diff().max() > paused_threshold:
        max_paused_duration = date_time_float.diff().max()
    else:
        max_paused_duration = 0
    return max_paused_duration


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_arbin_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv"

    from beep.structure import RawCyclerRun as rcrv1, \
        ProcessedCyclerRun as pcrv1

    rcr = rcrv1.from_arbin_file(test_arbin_path)

    print(rcr.data)
    #
    # pcr = pcrv1.from_raw_cycler_run(rcr)
    #
    # print(pcr.cycles_interpolated)
    #
    # ad = ArbinDatapath.from_file(test_arbin_path)
    #
    # print(ad.raw_data)
    #
    # df = ad.interpolate_cycles(v_range=None, resolution=1000,
    #                            diagnostic_available=False,
    #                            charge_axis="charge_capacity",
    #                            discharge_axis="discharge_capacity")
    #
    # print(df)

    # todo: only processed_cycler run MSONable is used

    from beep.validate import ValidatorBeep, SimpleValidator


    df1 = rcr.data
    df2 = pd.read_csv(test_arbin_path, index_col=0)

    # vb = ValidatorBeep()
    # print(vb.validate_arbin_dataframe(df1))
    # print(vb.validate_arbin_dataframe(df2))

    # print(vb.errors)

    sv = SimpleValidator()
