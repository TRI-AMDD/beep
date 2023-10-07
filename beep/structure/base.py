"""Base classes and assorted functions for beep structuring datapaths.
"""

import os
import abc
import json
import copy
import time

import pandas as pd
import numpy as np
from scipy import integrate
from monty.json import MSONable
from monty.io import zopen
from monty.serialization import dumpfn

from beep import tqdm
from beep.conversion_schemas import (
    STRUCTURE_DTYPES,
)

from beep import logger, VALIDATION_SCHEMA_DIR
from beep.structure.validate import SimpleValidator
from beep.structure.diagnostic import DiagnosticConfig


class BEEPDatapath(abc.ABC, MSONable):
    """The base class for all beep datapaths.

    BEEPDatapaths handle all BEEP structuring data, including

    - holding raw data
    - validating data
    - normal and diagnostic cycle interpolation
    - diagnostic extraction
    - automatic structuring
    - normal and diagnostic cycle summarization
    - determining cycle-capacity and capacity-cycle relationships
    - determining cycle lifetimes
    - saving and loading processed (structured) runs* from static json files
        - *including legacy json serialized with earlier versions of BEEP

    Note that BEEPDatapath does NOT handle featurization or ML modelling, only data organization and munging.


    BEEPDatapath is an abstract base class requiring a child class to implement the following methods:
    - from_file: Take raw cycler output files and convert them to a BEEPDatapath object. It should
        return a BEEPDatapath object with the correct data types (specified in __init__).


    Attributes:

        Important/Very common attributes:
            Each BEEPDatapath will have a maximum of 5 *very important* structured attributes dataframes:
            - structured_summary (pd.DataFrame): A summary of the cycles
            - structured_data (pd.DataFrame): The interpolated cycles
            - diagnostic_data (pd.DataFrame): The interpolated diagnostic cycles
            - diagnostic_summary (pd.DataFrame): A summary of diagnostic cycles
            - raw_data (pd.DataFrame): All raw data from the raw input file.
            - diagnostic (beep.structure.diagnostic.DiagnosticConfigBasic): A basic diagnostic configuration
                indicating where the diagnostic cycles are located.

        Less important attributes:
            - structuring_parameters: A dictionary of the structuring parameters used to structure
                this object.
            - metadata (BEEPDatapath.CyclerRunMetadata): An object holding all metadata.
            - paths (dict): A mapping of {descriptor: full_path or [paths]} for all files related to this datapath.
                This can include things like "raw", "metadata", "structured", as well as other paths (e.g., "eis").
            - schema: Validation schema used to validate the raw ingested data.

        Private:
            - _is_legacy (bool): Whether this file is loaded from a legacy ProcessedCyclerRun. Some operations are
                not supported for legacy-loaded processed/structured files.
            - _aggregation (dict): Specifies the pandas aggregation columns order for normal cycler data
            - _diag_aggregation (dict): Specifies the pandas aggregation columns order for diagnostic data
            - _summary_cols (list): The ordering of summary columns for normal cycler data.
            - _diag_summary_cols (list): The ordering of summary columns for diagnostic data.

    """

    IMPUTABLE_COLUMNS = ["temperature", "internal_resistance"]

    class StructuringDecorators:
        """
        Internal container class for decorators related to data structuring.
        """

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
            as some operations depend on data which is not in legacy serialized files.

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
        """
        Container class for cycler metadata. Does not change based on whether the instance is structured
        or unstructured.

        Attributes:
            raw (dict): The raw metadata dict, containing all input metadata.
            barcode (str): The barcode
            protocol (any, json serializable): The protocol used by the cycler for this run
            channel_id (int): Channel id number
        """

        def __init__(self, metadata_dict):
            """

            Args:
                metadata_dict (dict): A dictionary which can contain arbitrary metadata.
            """

            # Keep all the common metadata attr-accessible
            self.barcode = metadata_dict.get("barcode")
            self.protocol = metadata_dict.get("protocol")
            self.channel_id = metadata_dict.get("channel_id")

            # Extra metadata will always be in .raw
            self.raw = metadata_dict

        def __repr__(self):
            return str(self.raw)

        def __str__(self):
            return self.__repr__()

    def __init__(
            self,
            raw_data,
            metadata,
            paths=None,
            schema=None,
            impute_missing=True
    ):
        """

        Args:
            raw_data (pd.DataFrame): A pandas dataframe of raw data. Must contain the columns specified in
                summary_cols.
            metadata (dict): A dictionary for metadata. Should contain keys specified in CyclerRunMetadata, but does not
                have to.
            paths ({str: str, Pathlike}): Should contain "raw" and "metadata" keys, even if they are the same
                filepath.
            schema (str): the name of the validation schema file to use. Should be located in the
                VALIDATION_SCHEMA_DIR directory, or can alternatively be an absolute filepath.
            impute_missing (bool): Impute missing columns such as temperature and internal_resistance if they
                are not included in raw_data.
        """
        self.raw_data = raw_data

        # paths may include "raw", "metadata", and "structured", as well as others.
        if paths:
            for path_ref, path in paths.items():
                if path and not os.path.isabs(path):
                    raise ValueError(f"{path_ref}: '{path}' is not absolute! All paths must be absolute.")
            self.paths = paths
        else:
            self.paths = {"raw": None}

        if impute_missing:
            # support legacy operation
            if raw_data is not None:
                for col in self.IMPUTABLE_COLUMNS:
                    if col not in raw_data:
                        raw_data[col] = np.NaN

        if not schema:
            self.schema = schema
        else:
            if os.path.isabs(schema):
                abs_schema = schema
            else:
                abs_schema = os.path.join(VALIDATION_SCHEMA_DIR, schema)
            # TODO this dependence on a file path should be removed in the case of reloading an existing file
            # one solution would be to move this logic into the cycler subclasses and then pass the schema as a dict
            if os.path.exists(abs_schema):
                self.schema = abs_schema
            elif os.path.exists(os.path.join(VALIDATION_SCHEMA_DIR, os.path.split(schema)[-1])):
                self.schema = os.path.join(VALIDATION_SCHEMA_DIR, os.path.split(schema)[-1])
            else:
                raise FileNotFoundError(f"The schema file specified for validation could not be found: {schema}.")

        self.structured_summary = None
        self.structured_data = None
        self.diagnostic_data = None
        self.diagnostic_summary = None

        self._diagnostic = None

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
        self._diag_summary_cols.pop(5)  # internal_resistance
        self.structuring_parameters = {}
        self.indeterminate_step_default_charge = True

    @classmethod
    @abc.abstractmethod
    def from_file(cls, path, *args, **kwargs):
        """Go from a raw cycler output data file (or files) to a BEEPDatapath.

        Must be implemented for the BEEPDatapath child to be valid.

        Args:
            path (str, Pathlike): The path to the raw data file.

        Returns:
            (BEEPDatapath): A child class of BEEPDatapath.

        """
        raise NotImplementedError

    @StructuringDecorators.must_not_be_legacy
    def validate(self):
        """Validate the raw data.

        Returns:
            (bool) True if the raw data is valid, false otherwise.

        """

        if self.schema:
            v = SimpleValidator(schema_filename=self.schema)
        else:
            v = SimpleValidator()

        is_valid, reason = v.validate(self.raw_data)
        return is_valid, reason

    @classmethod
    def from_json_file(cls, filename):
        """Load a structured run previously saved to file.

        .json.gz files are supported.

        Loads a BEEPDatapath or (legacy) ProcessedCyclerRun structured object from json.

        Can be used in combination with files serialized with BEEPDatapath.to_json_file.

        Args:
            filename (str, Pathlike): a json file from a structured run, serialzed with to_json_file.

        Returns:
            None
        """
        with zopen(filename, "r") as f:
            d = json.load(f)

        # Add this structured file path to the paths dict
        paths = d.get("paths", {})
        paths["structured"] = os.path.abspath(filename)
        d["paths"] = paths

        return cls.from_dict(d)

    def to_json_file(self, filename, omit_raw=True):
        """Save a BEEPDatapath to disk as a json.

        .json.gz files are supported.

        Not named from_json to avoid conflict with MSONable.from_json(*)

        Args:
            filename (str, Pathlike): The filename to save the file to.
            omit_raw (bool): If True, saves only structured (NOT RAW) data.
                More efficient for saving/writing to disk.

        Returns:
            None
        """
        d = self.as_dict()
        if omit_raw:
            d.pop("raw_data")

        dumpfn(d, filename)

    @StructuringDecorators.must_not_be_legacy
    def as_dict(self):
        """Serialize a BEEPDatapath as a dictionary.

        Must not be loaded from legacy.

        Returns:
            (dict): corresponding to dictionary for serialization.

        """

        if not self.is_structured:
            summary = None
            cycles_interpolated = None
            diagnostic_summary = None
            diagnostic_interpolated = None
        else:
            summary = self.structured_summary.to_dict("list")
            cycles_interpolated = self.structured_data.to_dict("list")
            diagnostic_summary = self.diagnostic_summary.to_dict(
                "list") if self.diagnostic_summary is not None else None
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
            "diagnostic_interpolated": diagnostic_interpolated,

            # New diagnostic configuration
            "diagnostic": self.diagnostic.as_dict() if self.diagnostic else None,

            # Structuring parameters (mostly for provenance)
            "structuring_parameters": self.structuring_parameters,

            # Provence for validation
            "schema_path": self.schema,

        }

    @classmethod
    def from_dict(cls, d):
        """Create a BEEPDatapath object from a dictionary.

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
        schema = d.get("schema_path", None)

        # support legacy operations
        # support loads when raw_data not available
        if any([k not in d for k in ("raw_data", "metadata")]):
            raw_data = None
            metadata = {k: d.get(k) for k in ("barcode", "protocol", "channel_id")}
            is_legacy = True
        else:
            raw_data = pd.DataFrame(d["raw_data"])
            metadata = d.get("metadata")
            is_legacy = False

        datapath = cls(raw_data=raw_data, metadata=metadata, paths=paths, schema=schema)
        datapath._is_legacy = is_legacy

        datapath.structured_data = pd.DataFrame(d["cycles_interpolated"])
        datapath.structured_summary = pd.DataFrame(d["summary"])

        diagnostic_summary = d.get("diagnostic_summary")
        diagnostic_data = d.get("diagnostic_interpolated")
        datapath.diagnostic_summary = diagnostic_summary if diagnostic_summary is None else pd.DataFrame(
            diagnostic_summary)
        datapath.diagnostic_data = diagnostic_data if diagnostic_data is None else pd.DataFrame(diagnostic_data)

        datapath.structuring_parameters = d.get("structuring_parameters", {})

        diag = d.get("diagnostic", None)
        if diag:
            datapath.diagnostic = DiagnosticConfig.from_dict(diag)

        return datapath

    @property
    def semiunique_id(self):
        """
        An id that can identify the state of this datapath without complications
        associated with hashing dataframes.

        Returns:
            (str): A semiunique id for this datapath.

        """
        s = f"barcode:{self.metadata.barcode}-" \
            f"channel:{self.metadata.channel_id}-" \
            f"protocol:{self.metadata.protocol}-" \
            f"schema:{self.schema}-" \
            f"structured:{self.is_structured}-" \
            f"legacy:{self._is_legacy}"

        raw = self.paths.get("raw", None)
        structured = self.paths.get("structured", None)
        s += f"-raw_path:{raw}-structured_path:{structured}"
        return s

    @property
    def diagnostic(self):
        return self._diagnostic

    @diagnostic.setter
    def diagnostic(self, diagnostic):
        if not isinstance(diagnostic, (DiagnosticConfig, type(None))):
            raise TypeError("Diagnostic configuration must be "
                            "a DiagnosticConfig object.")
        self._diagnostic = diagnostic

    @diagnostic.deleter
    def diagnostic(self):
        del self._diagnostic

    @StructuringDecorators.must_not_be_legacy
    def structure(self,
                  v_range=None,
                  resolution=1000,
                  diagnostic_resolution=500,
                  nominal_capacity=1.1,
                  full_fast_charge=0.8,
                  charge_axis='charge_capacity',
                  discharge_axis='voltage',
                  exclude_cycles=None
                  ):
        """

        Args:
            v_range ([int, int]): range of voltages for cycle interpolation.
            resolution (int): resolution for cycle interpolation.
            diagnostic_resolution (int): number of datapoints per step for
                interpolating diagnostic cycles. For more granularity
                of diagnostic structuring, use the interpolate_diagnostic
                method.
            nominal_capacity (float): nominal capacity for summary stats.
            full_fast_charge (float): full fast charge for summary stats.
            charge_axis (str): Column to interpolate charge step
            discharge_axis (str): Column to interpolate discharge step
            exclude_cycles ([int]): List of cycle indices to exclude

        """

        if self.diagnostic:
            self.diagnostic_summary = self.summarize_diagnostic()
            self.diagnostic_data = self.interpolate_diagnostic_cycles(
                time_resolution=diagnostic_resolution,
                voltage_resolution=diagnostic_resolution
            )
        else:
            logger.info("Diagnostic not set; interpolating aging cycles only.")

        logger.info(
            f"Beginning structuring along charge axis '{charge_axis}' and discharge "
            f"axis '{discharge_axis}'."
        )
        self.structured_data = self.interpolate_cycles(
            v_range=v_range,
            resolution=resolution,
            charge_axis=charge_axis,
            discharge_axis=discharge_axis,
            exclude_cycles=exclude_cycles
        )

        self.structured_summary = self.summarize_cycles(
            nominal_capacity=nominal_capacity,
            full_fast_charge=full_fast_charge,
        )

        self.structuring_parameters = {
            "v_range": v_range,
            "resolution": resolution,
            "diagnostic_resolution": diagnostic_resolution,
            "nominal_capacity": nominal_capacity,
            "full_fast_charge": full_fast_charge,
            "charge_axis": charge_axis,
            "discharge_axis": discharge_axis
        }

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
        self.structuring_parameters = {}
        self.diagnostic = None
        logger.debug("Datapath structuring has been reset.")

    def interpolate_step(
            self,
            v_range,
            resolution,
            step_type="discharge",
            reg_cycles=None,
            axis="voltage",
            desc=None,
            exclude_cycles=None
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
            desc (str): Description to print to tqdm column.
            exclude_cycles ([int]): List of cycle indices to exclude

        Returns:
            pandas.DataFrame: DataFrame corresponding to interpolated values.
        """

        if not exclude_cycles:
            exclude_cycles = []

        if not desc and axis == "voltage":
            desc = \
                f"Interpolating {step_type} ({v_range[0]} - {v_range[1]})V " \
                f"({resolution} points)"
        elif not desc:
            desc = f"Interpolating {step_type} on axis {axis} ({resolution} points)"

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
        cycle_indices = sorted([c for c in cycle_indices if c in reg_cycles and c not in exclude_cycles])

        for cycle_index in tqdm(cycle_indices, desc=desc):
            cycle_df = self.raw_data.loc[self.raw_data["cycle_index"] == cycle_index]

            step_dfs = self.iterate_steps_in_cycle(cycle_df, step_type)

            for step_df in step_dfs:

                if step_df.size == 0 or step_df.shape[0] < 2:
                    continue
                if axis in ["charge_capacity", "discharge_capacity"]:
                    axis_range = [self.raw_data[axis].min(),
                                  self.raw_data[axis].max()]
                elif axis == "test_time":
                    axis_range = [step_df[axis].min(), step_df[axis].max()]
                elif axis == "voltage":
                    axis_range = v_range
                else:
                    raise ValueError(f"Axis {axis} not a valid step interpolation axis.")

                if len(step_df.step_index.unique()) > 1:
                    raise ValueError("Step DF has multiple step indices present!")

                step_index = step_df.step_index.iloc[0]

                step_df = interpolate_df(
                    step_df,
                    axis,
                    field_range=axis_range,
                    columns=incl_columns,
                    resolution=resolution,
                )

                step_df["cycle_index"] = cycle_index
                step_df["step_type"] = step_type
                step_df["step_index"] = step_index

                for c in ("step_type", "step_index"):
                    step_df[c] = step_df[c].astype("category")

                all_dfs.append(step_df)

        if not all_dfs:
            logger.warn(f"No steps found for cycle indices {cycle_indices} and step type {step_type}!")
            return pd.DataFrame()

        # Ignore the index to avoid issues with overlapping voltages
        result = pd.concat(all_dfs, ignore_index=True)

        # Cycle_index gets a little weird about typing, so round it here
        result.cycle_index = result.cycle_index.round()
        return result

    def interpolate_cycles(
            self,
            v_range=None,
            resolution=1000,
            charge_axis='charge_capacity',
            discharge_axis='voltage',
            exclude_cycles=None,
    ):
        """
        Gets interpolated cycles for both charge and discharge steps.

        Args:
            v_range ([float, float]): list of two floats that define
                the voltage interpolation range endpoints. Defaults to 2.8-3.5V.
            resolution (int): resolution of interpolated data.
            charge_axis (str): column to use for interpolation for charge
            discharge_axis (str): column to use for interpolation for discharge
            exclude_cycles ([int]): List of cycle indices to exclude


        Returns:
            (pandas.DataFrame): DataFrame corresponding to interpolated values.
        """
        if self.diagnostic:
            diag_mask = self.raw_data.cycle_index.isin(self.diagnostic.all_ix)
            reg_mask = ~diag_mask
        else:
            reg_mask = pd.Series([True] * self.raw_data.shape[0])

        reg_cycles = self.raw_data[reg_mask].cycle_index.unique()

        v_range = v_range or [2.8, 3.5]

        # If any regular cycle contains a waveform step, interpolate on test_time.
        if self.raw_data[reg_mask]. \
                groupby(["cycle_index", "step_index"]). \
                apply(step_is_waveform_dchg).any():
            discharge_axis = 'test_time'

        if self.raw_data[reg_mask]. \
                groupby(["cycle_index", "step_index"]). \
                apply(step_is_waveform_chg).any():
            charge_axis = 'test_time'

        interpolated_discharge = self.interpolate_step(
            v_range,
            resolution,
            step_type="discharge",
            reg_cycles=reg_cycles,
            axis=discharge_axis,
            exclude_cycles=exclude_cycles
        )
        interpolated_charge = self.interpolate_step(
            v_range,
            resolution,
            step_type="charge",
            reg_cycles=reg_cycles,
            axis=charge_axis,
            exclude_cycles=exclude_cycles
        )
        result = pd.concat(
            [interpolated_discharge, interpolated_charge], ignore_index=True
        )

        return self._cast_dtypes(result, "cycles_interpolated")

    # equivalent of legacy get_summary
    def summarize_cycles(
            self,
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
            (pandas.DataFrame): summary statistics by cycle.

        """
        if self.diagnostic:
            diag_mask = self.raw_data.cycle_index.isin(self.diagnostic.all_ix)
            reg_cycles = self.raw_data[~diag_mask].cycle_index.unique()
        else:
            reg_cycles = self.raw_data.cycle_index.unique()

        summary = self.raw_data.groupby("cycle_index").agg(self._aggregation)
        summary.columns = self._summary_cols

        summary = summary[summary.index.isin(reg_cycles)]
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
            pd.to_datetime(merged.date_time_iso_x, utc=True, errors="coerce"),
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

        # Add CV_time, CV_current, CV_capacity summary stats
        CV_time = []
        CV_current = []
        CV_capacity = []
        for cycle in summary.cycle_index:
            raw_cycle = self.raw_data.loc[self.raw_data.cycle_index == cycle]
            charge = raw_cycle.loc[raw_cycle.current > 0]
            CV = get_CV_segment_from_charge(charge)
            if CV.empty:
                logger.debug(f"Failed to extract CV segment for cycle {cycle}!")
                CV_time.append(np.nan)
                CV_current.append(np.nan)
                CV_capacity.append(np.nan)
            else:
                CV_time.append(get_CV_time(CV))
                CV_current.append(get_CV_current(CV))
                CV_capacity.append(get_CV_capacity(CV))
        summary["CV_time"] = CV_time
        summary["CV_current"] = CV_current
        summary["CV_capacity"] = CV_capacity

        summary = self._cast_dtypes(summary, "summary")

        # Avoid returning empty summary dataframe for single cycle raw_data
        if summary.shape[0] == 1:
            return summary

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
            self,
            time_resolution=1000,
            voltage_resolution=1000,
            v_delta_min=0.001,
            dv_fallback=None
    ):
        """
        Interpolates data by step (rather than basic chg/dchg) according to location and type of
        diagnostic cycles in the data

        Args:
            time_resolution (int): resolution of time-based interpolation
            voltage_resolution (int): resolution of voltage-based interpolation
            v_delta_min (float): minimum voltage delta for voltage based interpolation
            dv_fallback ((float, float)): Default fallback range for voltage-based interpolation
                if initial time/voltage interpolation is not within valid range. If not set,
                uses total voltage range of the step for interpolation.

        Returns:
            (pd.DataFrame) of interpolated diagnostic steps by step and cycle

        """

        if not self.diagnostic:
            raise ValueError("No DiagnosticConfig is set. Cannot interpolate diagnostic cycles.")
        diag_data = self.raw_data.loc[self.raw_data["cycle_index"].isin(self.diagnostic.all_ix)]
        # diag_types = [self.diagnostic.cycle_to_type[cix] for cix in diag_data.cycle_index.unique()]

        # Counter to ensure non-contiguous repeats of step_index
        # within same cycle_index are grouped separately
        diag_data.loc[:, "step_index_counter"] = 0

        for cycle_index in self.diagnostic.all_ix:
            indices = diag_data.loc[diag_data.cycle_index == cycle_index].index
            step_index_list = diag_data.step_index.loc[indices]
            shifted = step_index_list.ne(step_index_list.shift()).cumsum()
            diag_data.loc[indices, "step_index_counter"] = shifted

        group = diag_data.groupby(["cycle_index", "step_index", "step_index_counter"])
        incl_columns = [
            "current",
            "voltage",
            "charge_capacity",
            "discharge_capacity",
            "charge_energy",
            "discharge_energy",
            "internal_resistance",
            "temperature",
            "test_time",
        ]

        # todo: not sure this diag_dict thing is needed - alex
        diag_dict = {}
        for cycle in diag_data.cycle_index.unique():
            diag_dict.update({cycle: None})
            steps = diag_data[diag_data.cycle_index == cycle].step_index.unique()
            diag_dict[cycle] = list(steps)

        all_dfs = []
        for (cycle_index, step_index, step_index_counter), df in tqdm(group, desc="Interpolating diagnostic by step"):
            if len(df.index) < 2:
                logger.debug(f"Skipping cycle: {cycle_index}, step: {step_index_counter} with step "
                             f"type: {step_index} as there were < 2 data points.")
            step_dv = df.voltage.max() - df.voltage.min()
            dv = [df.voltage.min(), df.voltage.max()]
            if cycle_index in self.diagnostic.hppc_ix and step_dv >= v_delta_min:
                # Old way of using voltage resolution as an actual value of volts; default was 0.0005V
                # voltage_res = int(
                #     (df.voltage.max() - df.voltage.min()) / hppc_v_resolution
                # )
                new_df = interpolate_df(
                    df,
                    field_name="voltage",
                    field_range=dv,
                    columns=incl_columns,
                    resolution=voltage_resolution,
                )
            elif step_dv < v_delta_min:
                t_range_step = [df.test_time.min(), df.test_time.max()]
                new_df = interpolate_df(
                    df,
                    field_name="test_time",
                    field_range=t_range_step,
                    columns=incl_columns,
                    resolution=time_resolution,
                )
            else:
                if dv_fallback:
                    v_range = dv_fallback
                else:
                    v_range = dv
                new_df = interpolate_df(
                    df,
                    field_name="voltage",
                    field_range=v_range,
                    columns=incl_columns,
                    resolution=voltage_resolution,
                )
            new_df["cycle_index"] = cycle_index
            new_df["cycle_type"] = self.diagnostic.type_by_ix[cycle_index]
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
            if new_df.shape[0] < 2:
                logger.debug(f"Step number {step_index_counter} with step type {step_index} has less than "
                             f"2 interpolated points; refusing to append to interpolated data.")
            else:
                all_dfs.append(new_df)

        # Ignore the index to avoid issues with overlapping voltages
        result = pd.concat(all_dfs, ignore_index=True)
        result.sort_values(
            by=["cycle_index", "step_index_counter", "test_time"],
            axis=0,
            inplace=True
        )
        # Cycle_index gets a little weird about typing, so round it here
        result.cycle_index = result.cycle_index.round()
        result = self._cast_dtypes(result, "diagnostic_interpolated")

        return result

    def summarize_diagnostic(self):
        """
        Gets summary statistics for data according to location of
        diagnostic cycles in the data

        Returns:
            (DataFrame) of summary statistics by cycle

        """
        diag_summary = self.raw_data.groupby("cycle_index").agg(self._diag_aggregation)

        diag_summary.columns = self._diag_summary_cols

        diag_summary = diag_summary[diag_summary.index.isin(self.diagnostic.all_ix)]

        diag_summary["coulombic_efficiency"] = (
            diag_summary["discharge_capacity"] / diag_summary["charge_capacity"]
        )
        diag_summary["paused"] = self.raw_data.groupby("cycle_index").apply(
            get_max_paused_over_threshold
        )

        diag_summary.reset_index(drop=True, inplace=True)

        diag_summary["cycle_type"] = [
            self.diagnostic.type_by_ix[cix] for cix in diag_summary["cycle_index"]
        ]

        # Add CV_time, CV_current, and CV_capacity summary stats
        CV_time = []
        CV_current = []
        CV_capacity = []
        for cycle in diag_summary.cycle_index:
            raw_cycle = self.raw_data.loc[self.raw_data.cycle_index == cycle]

            # Charge is the very first step_index
            CCCV = raw_cycle.loc[raw_cycle.step_index == raw_cycle.step_index.min()]
            CV = get_CV_segment_from_charge(CCCV)
            if CV.empty:
                logger.debug(f"Failed to extract CV segment for diagnostic cycle {cycle}!")
            CV_time.append(get_CV_time(CV))
            CV_current.append(get_CV_current(CV))
            CV_capacity.append(get_CV_capacity(CV))

        diag_summary["CV_time"] = CV_time
        diag_summary["CV_current"] = CV_current
        diag_summary["CV_capacity"] = CV_capacity

        diag_summary = self._cast_dtypes(diag_summary, "diagnostic_summary")

        return diag_summary

    def iterate_steps_in_cycle(self, cycle_df, step_type):
        """
        For a given cycle df, return an iterable (or yield)
        individual dfs corresponding to step indices and charge step.

        For the simplest and easiest use, this means that within a
        single charge cycle, there will be one discharge and one charge
        step, each with an equal number of interpolated points.

        Args:
            cycle_df (pd.Dataframe): The dataframe corresponding to an
                entire cycle.
            step_type (str): "charge" or "discharge"

        Returns:
            (pd.Dataframe): The dataframe corresponding to a particular
                charge/discharge step. Used downstream for interpolation.

        """
        if step_type == "discharge":
            step_filter = step_is_dchg
        elif step_type == "charge":
            step_filter = step_is_chg
        else:
            raise ValueError(f"'{step_type}' is not a recognized step type")
        dfs_chgstate = cycle_df.groupby("step_index").filter(
            lambda ldf: step_filter(
                ldf,
                indeterminate_step_charge=self.indeterminate_step_default_charge
            )
        )
        for _, step_df in dfs_chgstate.groupby("step_index"):
            yield step_df

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

    @staticmethod
    def _cast_dtypes(result, structure_dtypes_key):
        """Cast data types of a result dataframe to those specified by the structuring config.

        Args:
            result (pd.DataFrame): The result to cast
            structure_dtypes_key: The required key, for structure dtype casting, as per STRUCTURE_DTYPES

        Returns:
            (pd.DataFrame): The result, cast to the correct datatypes.

        """
        available_dtypes = {}
        for field, dtype in STRUCTURE_DTYPES[structure_dtypes_key].items():
            if field in result.columns:
                # if not result[field].isna().all():
                available_dtypes[field] = dtype

        return result.astype(available_dtypes)


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
    # Remove duplicates
    interpolated_df = interpolated_df[~interpolated_df[field_name].duplicated()]
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
    return interpolated_df


def step_is_chg_state(step_df, chg, indeterminate_step_charge=True):
    """
    Helper function to determine whether a given dataframe corresponding
    to a single cycle_index's step is charging or discharging, only intended
    to be used with a dataframe for single step/cycle

    Args:
         step_df (pandas.DataFrame): dataframe to determine whether
            charging or discharging
         chg (bool): Charge state; True if charging, False if discharging
         indeterminate_step_charge (bool): Default to "charge" indication
            for steps where charge/discharge cannot be determined
            confidently. If false, indeterminate steps are considered
            discharge.

    Returns:
        (bool, None): True if step is the charge state specified,
            False if it is confidently not, or None if the
            charge state is indeterminate.
    """
    cap = step_df[["charge_capacity", "discharge_capacity"]]
    total_cap_diffs = cap.max() - cap.min()

    cdiff = cap.diff(axis=0)
    c_points_flagged = (cdiff / total_cap_diffs) > 0.99
    cdiff = cdiff[~c_points_flagged]

    avg_chg_delta = cdiff.mean(axis=0)["charge_capacity"]
    avg_dchg_delta = cdiff.mean(axis=0)["discharge_capacity"]

    if np.isnan(avg_chg_delta) and np.isnan(avg_dchg_delta):
        is_charging = None
    elif np.isnan(avg_chg_delta):
        is_charging = avg_dchg_delta > 0
    elif np.isnan(avg_dchg_delta):
        is_charging = avg_chg_delta > 0
    else:
        if avg_chg_delta > avg_dchg_delta and avg_chg_delta > 0:
            is_charging = True
        elif avg_chg_delta < avg_dchg_delta and avg_dchg_delta > 0:
            is_charging = False
        else:
            is_charging = None

    # If the is_charging cannot be determined, return false
    # regardless.
    if is_charging is None:
        return chg == indeterminate_step_charge
    else:
        return chg == is_charging


def step_is_dchg(step_df, **kwargs):
    return step_is_chg_state(step_df, False, **kwargs)


def step_is_chg(step_df, **kwargs):
    return step_is_chg_state(step_df, True, **kwargs)


def step_is_waveform(step_df, chg_filter):
    """
    Helper function for driving profiles to determine whether a given dataframe corresponding
    to a single cycle_index/step is a waveform discharge.

    Args:
         step_df (pandas.DataFrame): dataframe to determine whether waveform step is present
         chg_filter (func): Function for determining whether step is charging or not.


    Returns:
        (bool): True if step is waveform discharge.
    """

    # Check for waveform in maccor

    voltage_resolution = 3
    if len([col for col in step_df.columns if '_wf_' in col]):
        return (chg_filter(step_df)) & \
               ((step_df['_wf_chg_cap'].notna().any()) |
                (step_df['_wf_dis_cap'].notna().any()))
    elif not np.round(step_df.voltage, voltage_resolution).is_monotonic:
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
        (float): number of seconds that test was paused

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


def get_CV_segment_from_charge(charge, dt_tol=1, dVdt_tol=1e-5, dIdt_tol=1e-4):
    """
    Extracts the constant voltage segment from charge. Works for both CCCV or
    CC steps followed by a CV step.

    Args:
        charge (pd.DataFrame): charge dataframe for a single cycle
        dt_tol (float) : dt tolernace (minimum) for identifying CV
        dVdt_tol (float) : dVdt tolerance (maximum) for identifying CV
        dIdt_tol (float) : dVdt tolerance (minimum) for identifying CV 

    Returns:
        (pd.DataFrame): dataframe containing the CV segment

    """
    if charge.empty:
        return charge
    else:
        # Compute dI and dV
        dI = np.diff(charge.current)
        dV = np.diff(charge.voltage)
        dt = np.diff(charge.test_time)

        # Find the first index where the CV segment begins
        i = 0
        while i < len(dV) and (dt[i] < dt_tol or abs(dV[i]/dt[i]) > dVdt_tol or abs(dI[i]/dt[i]) < dIdt_tol):
            i = i+1

        # Filter for CV phase
        return charge.loc[charge.test_time >= charge.test_time.iat[i-1]]


def get_CV_time(CV):
    """
    Helper function to compute CV time.

    Args:
        CV (pd.DataFrame): CV segement of charge

    Returns:
        (float): length of the CV segment in seconds

    """
    if not CV.empty:
        return CV.test_time.iat[-1] - CV.test_time.iat[0]


def get_CV_current(CV):
    """
    Helper function to compute CV current.

    Args:
        CV (pd.DataFrame): CV segement of charge

    Returns:
        (float): current reached at the end of the CV segment

    """
    if not CV.empty:
        return CV.current.iat[-1]


def get_CV_capacity(CV):
    """
    Helper function to compute CV capacity.

    Args:
        CV (pd.DataFrame): CV segement of charge

    Returns:
        (float): charge capacity during the CV segment

    """
    if not CV.empty:
        return CV.charge_capacity.iat[-1] - CV.charge_capacity.iat[0]
