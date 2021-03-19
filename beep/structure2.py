import abc
import json
import re
from datetime import datetime
import copy

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


# todo: ALEXTODO add more logging operations

class BEEPDatapath(abc.ABC, MSONable):
    """
    Each BEEPDatapath will have a maximum of four *very important* structured attributes dataframes:
        - structured_summary: A summary of the cycles
        - structured_data: The interpolated cycles
        - diagnostic_data: The interpolated diagnostic cycles
        - diagnostic_summary: A summary of diagnostic cycles
    """

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

    class StructuringDecorators:

        @classmethod
        def must_be_structured(cls, func):
            """
            Decorator to check if the datapath has been structured.
            Args:
                func: A function or method.
            Returns:
                A wrapper function for the input function/method.
            """

            def wrapper(*args, **kwargs):
                if not args[0].is_structured:
                    raise RuntimeError(
                        f"{args[0].__class__.__name__} has not been structured! Run .structure(*args)."
                    )
                else:
                    return func(*args, **kwargs)

            return wrapper


        @classmethod
        def must_not_be_legacy(cls, func):
            """
            Decorator to check if datapath has been serialized from legacy,
            as some operations depend on data which is not in (or not is not easily accessed from)
            legacy serialized files.

            Args:
                func: A function or method

            Returns:
                A wrapper function for the input function/method.

            """

            def wrapper(*args, **kwargs):
                if args[0]._is_legacy:
                    raise ValueError(
                        f"{args[0].__class__.__name__} is deserialized from a legacy file! Operation not allowed."
                    )
                else:
                    return func(*args, **kwargs)

            return wrapper

    class CyclerRunMetadata:
        def __init__(self, metadata_dict):
            self.barcode = metadata_dict.get("barcode")
            self.protocol = metadata_dict.get("protocol")
            self.channel_id = metadata_dict.get("channel_id")
            self.raw = metadata_dict


    def __init__(self, raw_data, metadata, paths=None):
        self.raw_data = raw_data


        # paths may include "raw", "metadata", and "structured", as well as others.
        if paths:
            for path_ref, path in paths.items():
                if not os.path.isabs(path):
                    raise ValueError(f"{path_ref}: '{path}' is not absolute! All paths must be absolute.")
            self.paths = paths
        else:
            self.paths = {"raw": None}

        self.structured_summary = None     # equivalent of PCR.summary
        self.structured_data = None        # equivalent of PCR.cycles_interpolated
        self.diagnostic_data = None        # equivalent of PCR.diagnostic_interpolated
        self.diagnostic_summary = None     # same name as in PCR

        self.metadata = self.CyclerRunMetadata(metadata)

        self._is_legacy = False

        # Setting aggregation/column ordering based on whether columns are present
        self._aggregation = {
            "cycle_index": "first",
            "discharge_capacity": "max",
            "charge_capacity": "max",
            "discharge_energy": "max",
            "charge_energy": "max",
            "internal_resistance": "last",
        }

        if self.raw_data is not None and "temperature" in self.raw_data.columns:
            self._aggregation["temperature"] = ["max", "mean", "min"]

            self._summary_cols = [
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
        else:
            self._summary_cols = [
                "cycle_index",
                "discharge_capacity",
                "charge_capacity",
                "discharge_energy",
                "charge_energy",
                "dc_internal_resistance",
                "date_time_iso",
            ]

        # Ensure this step of the aggregation is placed last
        self._aggregation["date_time_iso"] = "first"

        self._diag_aggregation = copy.deepcopy(self._aggregation)
        self._diag_aggregation.pop("internal_resistance")
        self._diag_summary_cols = copy.deepcopy(self._summary_cols)
        self._diag_summary_cols.pop(5) # internal_resistance

    # @property
    # @abc.abstractmethod
    # def schema(self):
    #     raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def from_file(cls, path):
        raise NotImplementedError


    def _cast_dtypes(self, result, structure_dtypes_key):
        available_dtypes = {}
        for field, dtype in STRUCTURE_DTYPES[structure_dtypes_key].items():
            if field in result.columns:
                # if not result[field].isna().all():
                available_dtypes[field] = dtype

        return result.astype(available_dtypes)

    @StructuringDecorators.must_not_be_legacy
    def as_dict(self):
        """
        Method for dictionary serialization.

        Returns:
            dict: corresponding to dictionary for serialization.

        """

        if not self.is_structured:
            summary = None
            cycles_interpolated = None
            diagnostic_summary = None
            diagnostic_interpolated = None
        else:
            summary = self.structured_summary.to_dict("list")
            cycles_interpolated = self.structured_data.to_dict("list")
            diagnostic_summary = self.diagnostic_summary.to_dict("list") if self.diagnostic_summary is not None else None
            diagnostic_interpolated = self.diagnostic_data.to_dict("list") if self.diagnostic_data is not None else None


        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,

            # Vital things needed for BEEPDatapath
            "raw_data": self.raw_data.to_dict("list"),
            "metadata": self.metadata.raw,
            "paths": self.paths,


            # For backwards compatibility
            # All this data is in the "metadata" key, this is just for redundancy
            "barcode": self.metadata.barcode,
            "protocol": self.metadata.protocol,
            "channel_id": self.metadata.channel_id,

            # Structured data, expensively obtained
            "summary": summary,
            "cycles_interpolated": cycles_interpolated,
            "diagnostic_summary": diagnostic_summary,
            "diagnostic_interpolated": diagnostic_interpolated
        }

    def to_json_file(self, filename):
        """
        Not named from_json to avoid conflict with MSONable.from_json(*)

        Args:
            filename:

        Returns:

        """
        with open(filename, "w") as f:
            json.dump(self.as_dict(), f)

    @StructuringDecorators.must_not_be_legacy
    def unstructure(self):
        """
        Cleanly remove all structuring data, to restructure using different parameters.

        Returns:
            None
        """
        self.structured_data = None
        self.diagnostic_data = None
        self.structured_summary = None
        self.diagnostic_summary = None
        # todo: print logging statement saying structuring has been reset

    @classmethod
    def from_dict(cls, d):
        """
        Args:
            d (dict): dictionary represenation.

        Returns:
            beep.structure.ProcessedCyclerRun: deserialized ProcessedCyclerRun.
        """
        for obj, dtype_dict in STRUCTURE_DTYPES.items():
            for column, dtype in dtype_dict.items():
                if d.get(obj) is not None:
                    if d[obj].get(column) is not None:
                        d[obj][column] = pd.Series(d[obj][column], dtype=dtype)

        paths = d.get("paths", None)

        # support legacy operations
        if any([k not in d for k in ("raw_data", "metadata")]):
            raw_data = None
            metadata = {k: d.get(k) for k in ("barcode", "protocol", "channel_id")}
            is_legacy = True
        else:
            raw_data = pd.DataFrame(d["raw_data"])
            metadata = d.get("metadata")
            is_legacy = False

        datapath = cls(raw_data=raw_data, metadata=metadata, paths=paths)
        datapath._is_legacy = is_legacy

        datapath.structured_data = pd.DataFrame(d["cycles_interpolated"])
        datapath.structured_summary = pd.DataFrame(d["summary"])

        diagnostic_summary = d.get("diagnostic_summary")
        diagnostic_data = d.get("diagnostic_interpolated")
        datapath.diagnostic_summary = diagnostic_summary if diagnostic_summary is None else pd.DataFrame(diagnostic_summary)
        datapath.diagnostic_data = diagnostic_data if diagnostic_data is None else pd.DataFrame(diagnostic_data)
        return datapath

    @classmethod
    def from_json_file(cls, filename):
        """

        Args:
            filename:

        Returns:

        """
        with open(filename, "r") as f:
            d = json.load(f)

        # Add this structured file path to the paths dict
        paths = d.get("paths", {})
        paths["structured"] = os.path.abspath(filename)
        d["paths"] = paths

        return cls.from_dict(d)

    def autostructure(self):
        """
        Automatically run structuring based on automatically determined structuring parameters.
        The parameters are determined from the raw input file, so ensure the raw input file paths
        are in the 'paths' attribute.

        Returns:
            None
        """
        v_range, resolution, nominal_capacity, full_fast_charge, diagnostic_available = \
            self.determine_structuring_parameters()
        logger.info(f"Autostructuring determined parameters of v_range={v_range}, "
                    f"resolution={resolution}, "
                    f"nominal_capacity={nominal_capacity}, "
                    f"full_fast_charge={full_fast_charge}, "
                    f"diagnostic_available={diagnostic_available}")
        return self.structure(
            v_range=v_range,
            resolution=resolution,
            nominal_capacity=nominal_capacity,
            full_fast_charge=full_fast_charge,
            diagnostic_available=diagnostic_available
        )

    @StructuringDecorators.must_not_be_legacy
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
        """
        logger.info(f"Beginning structuring along charge axis '{charge_axis}' and discharge axis '{discharge_axis}'.")


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

        if not desc:
            desc = \
                f"Interpolating {step_type} ({v_range[0]} - {v_range[1]})V " \
                f"({resolution} points)"

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
            "charge_energy",
            "discharge_energy",
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
        )
        interpolated_charge = self.interpolate_step(
            v_range,
            resolution,
            step_type="charge",
            reg_cycles=reg_cycles,
            axis=charge_axis,
        )
        result = pd.concat(
            [interpolated_discharge, interpolated_charge], ignore_index=True
        )

        return self._cast_dtypes(result, "cycles_interpolated")

    # equivalent of get_summary
    def summarize_cycles(
            self,
            diagnostic_available=False,
            nominal_capacity=1.1,
            full_fast_charge=0.8,
            cycle_complete_discharge_ratio=0.97,
            cycle_complete_vmin=3.3,
            cycle_complete_vmax=3.3,
            error_threshold=1e6
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

        summary = self.raw_data.groupby("cycle_index").agg(self._aggregation)

        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 1000)

        summary.columns = self._summary_cols

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
        if "temperature" in self.raw_data.columns:
            summary["time_temperature_integrated"] = self.raw_data.groupby(
                "cycle_index").apply(
                lambda g: integrate.trapz(g.temperature, x=g.time_since_cycle_start)
            )

        # Drop the time since cycle start column
        self.raw_data.drop(columns=["time_since_cycle_start"])

        # Determine if any of the cycles has been paused
        summary["paused"] = self.raw_data.groupby("cycle_index").apply(
            get_max_paused_over_threshold)

        summary = self._cast_dtypes(summary, "summary")

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

        diag_data = self.raw_data[self.raw_data["cycle_index"].isin(diag_cycles_at)]

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
        diag_summary = self.raw_data.groupby("cycle_index").agg(self._diag_aggregation)

        diag_summary.columns = self._diag_summary_cols

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

        diag_summary = self._cast_dtypes(diag_summary, "diagnostic_summary")

        return diag_summary


    # locate diagnostic cycles
    # determine voltage range to interpolate on
    # this is a function that TRI is using mostly for themselves
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
            self.paths["raw"], parameters_path
        )
        # Logic for interpolation variables and diagnostic cycles
        diagnostic_available = False
        if run_parameter is not None:
            if {"capacity_nominal"}.issubset(run_parameter.columns.tolist()):
                nominal_capacity = run_parameter["capacity_nominal"].iloc[0]
            if {"discharge_cutoff_voltage", "charge_cutoff_voltage"}.issubset(
                    run_parameter.columns):
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
                    diag_0_pattern = [len(self.raw_data[self.raw_data.cycle_index == x].step_index.unique())
                                      for x in range(initial_diagnostic_at[0], initial_diagnostic_at[0] + hppc_rpt_len)]
                    diag_1_pattern = [len(self.raw_data[self.raw_data.cycle_index == x].step_index.unique())
                                      for x in range(initial_diagnostic_at[1], initial_diagnostic_at[1] + hppc_rpt_len)]
                    # Find the steps present in the reset cycles for the first and second diagnostic
                    diag_0_steps = set(self.raw_data[self.raw_data.cycle_index == initial_diagnostic_at[0]].step_index.unique())
                    diag_1_steps = set(self.raw_data[self.raw_data.cycle_index == initial_diagnostic_at[1]].step_index.unique())
                    diagnostic_starts_at = []
                    for cycle in self.raw_data.cycle_index.unique():
                        steps_present = set(self.raw_data[
                                                self.raw_data.cycle_index == cycle].step_index.unique())
                        cycle_pattern = [len(self.raw_data[
                                                 self.raw_data.cycle_index == x].step_index.unique())
                                         for x in
                                         range(cycle, cycle + hppc_rpt_len)]
                        if steps_present == diag_0_steps or steps_present == diag_1_steps:
                            diagnostic_starts_at.append(cycle)
                        # Detect final diagnostic if present in the data
                        elif cycle >= (
                                self.raw_data.cycle_index.max() - hppc_rpt_len - 1) and \
                                (cycle_pattern == diag_0_pattern or cycle_pattern == diag_1_pattern):
                            diagnostic_starts_at.append(cycle)

                    diagnostic_available = {
                        "parameter_set":
                            run_parameter["diagnostic_parameter_set"].iloc[0],
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

    @StructuringDecorators.must_be_structured
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
        if len(self.structured_summary) > n_cycles_cutoff:
            max_capacity = np.median(
                self.structured_summary.discharge_capacity.iloc[0:n_cycles_cutoff]
            )
        else:
            max_capacity = 1.1

        # If capacity falls below 80% of initial capacity by end of run
        if self.structured_summary.discharge_capacity.iloc[-1] / max_capacity <= threshold:
            cycle_life = self.structured_summary[
                self.structured_summary.discharge_capacity < threshold * max_capacity
            ].index[0]
        else:
            # Some cells do not degrade below the threshold (low degradation rate)
            cycle_life = len(self.structured_summary) + 1

        return cycle_life


    @StructuringDecorators.must_be_structured
    def cycles_to_capacities(self, cycle_min=200, cycle_max=1800, cycle_interval=200):
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
                discharge_capacities[counter] = self.structured_summary.discharge_capacity.iloc[
                    cycle_index
                ]
            except IndexError:
                pass
            counter = counter + 1
        discharge_capacities.columns = np.core.defchararray.add(
            "cycle_", cycle_indices.astype(str)
        )
        return discharge_capacities


    @StructuringDecorators.must_be_structured
    def capacities_to_cycles(
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

    @property
    def paused_intervals(self):
        # a method to use get_max_paused_over_threshold
        return self.raw_data.groupby("cycle_index").apply(get_max_paused_over_threshold)

    @property
    def is_structured(self):
        required = [self.structured_summary, self.structured_data, self.diagnostic_summary, self.diagnostic_data]
        if any([df is not None for df in required]):
            return True
        else:
            return False


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
            "raw": path,
            "metadata": metadata_path if metadata else None
        }

        return cls(data, metadata, paths)


class MaccorDatapath(BEEPDatapath):

    @classmethod
    def from_file(cls, path):
        """
        Method for ingestion of Maccor format files.

        Args:
            filename (str): file path for maccor format file.
            include_eis (bool): whether to include the eis spectrum
                in the ingestion procedure.
            validate (bool): whether to validate on instantiation.
        """
        with open(path) as f:
            metadata_line = f.readline().strip()

        # Parse data
        data = pd.read_csv(path, delimiter="\t", skiprows=1)
        data.rename(str.lower, axis="columns", inplace=True)
        data = data.astype(MACCOR_CONFIG["data_types"])
        data.rename(MACCOR_CONFIG["data_columns"], axis="columns", inplace=True)
        data["charge_capacity"] = cls.quantity_sum(
            data, "capacity", "charge"
        )
        data["discharge_capacity"] = cls.quantity_sum(
            data, "capacity", "discharge"
        )
        data["charge_energy"] = cls.quantity_sum(data, "energy", "charge")
        data["discharge_energy"] = cls.quantity_sum(
            data, "energy", "discharge"
        )

        # Parse metadata - kinda hackish way to do it, but it works
        metadata = cls.parse_metadata(metadata_line)
        metadata = pd.DataFrame(metadata)
        _, channel_number = os.path.splitext(path)
        metadata["channel_id"] = int(channel_number.replace(".", ""))
        metadata.rename(str.lower, axis="columns", inplace=True)
        metadata.rename(MACCOR_CONFIG["metadata_fields"], axis="columns", inplace=True)
        # Note the to_dict, which scrubs numpy typing
        metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}


        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(cls.correct_timestamp)

        paths = {
            "raw": path,
            "metadata": path
        }

        return cls(data, metadata, paths=paths)


    def load_eis(self, path):
        # todo: ALEXTODO: move this to load_eis method or similar
        # # Check for EIS files
        # if include_eis:
        #     eis_pattern = ".*.".join(filename.rsplit(".", 1))
        #     all_eis_files = glob(eis_pattern)
        #     eis = EISpectrum.from_maccor_file(all_eis_files[0])
        # else:
        #     eis = None
        pass


    @staticmethod
    def quantity_sum(data, quantity, state_type):
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


    @staticmethod
    def parse_metadata(metadata_string):
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
        metadata_values = MaccorDatapath.split_string_by_fields(metadata_string,
                                                 metadata_fields)
        metadata = {
            k.replace(":", ""): [v.strip()]
            for k, v in zip(metadata_fields, metadata_values)
        }
        return metadata


    @staticmethod
    def correct_timestamp(x):
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
                pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"),
                                 is_dst=True)
                    .astimezone(utc)
                    .isoformat()
            )
        except ValueError:
            x = x + " 00:00:00"
            iso = (
                pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"),
                                 is_dst=True)
                    .astimezone(utc)
                    .isoformat()
            )
        return iso


    @staticmethod
    def split_string_by_fields(string, fields):
        """
        Helper function to split a string by a set of ordered strings,
        primarily used for Maccor metadata parsing.

        >>> MaccorDatapath.split_string_by_fields("first name: Joey  last name Montoya",
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

# based on get_interpolated_data
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

    df = dataframe.loc[:, dataframe.columns.intersection(columns)]
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
    interpolated_df = interpolated_df.merge(df, how="outer", on=field_name, sort=True)
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

    # test_arbin_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv"
    # test_maccor_path_w_diagnostics = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/xTESLADIAG_000020_CH71.071"
    #
    # from beep.structure import RawCyclerRun as rcrv1, \
    #     ProcessedCyclerRun as pcrv1
    #
    # # rcr = rcrv1.from_arbin_file(test_arbin_path)
    # # rcr.data.to_csv("BEEPDatapath_arbin_memloaded.csv")
    # # with open("tests/test_files/BEEPDatapath_arbin_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    #
    #
    # # test_maccor_paused_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000151_paused.052"
    # # rcr = rcrv1.from_maccor_file(test_maccor_paused_path, include_eis=False)
    # # rcr.data.to_csv("BEEPDatapath_maccor_paused_memloaded.csv")
    # # with open("tests/test_files/BEEPDatapath_maccor_paused_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    # test_maccor_paused_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000151_test.052"
    # rcr = rcrv1.from_maccor_file(test_maccor_paused_path, include_eis=False)
    # rcr.data.to_csv("tests/test_files/BEEPDatapath_maccor_timestamp_memloaded.csv")
    # with open("tests/test_files/BEEPDatapath_maccor_timestamp_metadata_memloaded.json", "w") as f:
    #     json.dump(rcr.metadata, f)
    #
    #
    # # rcr = rcrv1.from_maccor_file(filename=test_maccor_path_w_diagnostics, include_eis=False)
    # # rcr.data.to_csv("BEEPDatapath_maccor_w_diagnostic_memloaded.csv")
    # #
    # # with open("BEEPDatapath_maccor_with_diagnostic_metadata_memloaded.json", "w") as f:
    # #     json.dump(rcr.metadata, f)
    #
    #
    # print(rcr.metadata)
    #
    # print(rcr.data)



    # maccor = MaccorDatapath.from_file("/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PredictionDiagnostics_000109_tztest.010")
    # maccor.raw_data.to_csv("tests/test_files/BEEPDatapath_maccor_parameterized.csv")
    # with open("tests/test_files/BEEPDatapath_maccor_parameterized_metadata_memloaded.json", "w") as f:
    #     json.dump(maccor.metadata.raw, f)
    # maccor.structure()
    # print(maccor.get_cycle_life())

    # self.maccor_file_diagnostic_normal = os.path.join(
    #     TEST_FILE_DIR, "PreDiag_000287_000128short.092"
    # )
    # self.maccor_file_diagnostic_misplaced = os.path.join(
    #     TEST_FILE_DIR, "PreDiag_000412_00008Fshort.022"

    for d, f in {"diagnostic_normal":"PreDiag_000287_000128short.092", "diagnostic_misplaced":"PreDiag_000412_00008Fshort.022"}.items():
        maccor = MaccorDatapath.from_file(os.path.join("/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/", f))
        maccor.raw_data.to_csv(f"tests/test_files/BEEPDatapath_maccor_{d}_memloaded.csv")
        with open(f"tests/test_files/BEEPDatapath_maccor_{d}_metadata_memloaded.json", "w") as fi:
            json.dump(maccor.metadata.raw, fi)






    raise ValueError
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
