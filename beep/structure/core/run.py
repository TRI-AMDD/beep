import os
from typing import Optional, Union

import tqdm
import dask.dataframe as dd
import dask.bag as bag
import pandas as pd
import numpy as np

from monty.json import MSONable, MontyDecoder, jsanitize
from monty.serialization import loadfn, dumpfn
from dask.diagnostics import ProgressBar

from beep import logger
from beep.structure.diagnostic import DiagnosticConfig
from beep.structure.core.cycles_container import CyclesContainer
from beep.structure.core.util import (
    label_chg_state,
    get_max_paused_over_threshold,
    get_cv_stats,
    DFSelectorIndexError,
    CVStatsError,
    TQDM_STYLE_ARGS
)
from beep.structure.core.interpolate import interpolate_cycle, CONTAINER_CONFIG_DEFAULT
from beep.structure.core.validate import SimpleValidator

"""
Things assumed to make sure it all works correctly with checking:
    1. All cycles have a unique cycle index, and no cycles have more than one cycle index.
    2. All steps have a unique step index, and no steps have more than one step index.


# THINGS A RUN MUST IMPLEMENT
 - from_file (method)
 - conversion schema
    - must have 3 root keys: file_pattern, data_columns, data_types. 
        Optionally, can have metadata_fields
 - validation schema (class attribute)

"""


class Run(MSONable):
    """
    A Run object represents an entire cycler run as well as it's structured (interpolated)
    data. It is the top level object in the structured data hierarchy.

    A run object has its own config. This config mainly determines what columns
    will be kept and what data types they will possess.

    Args:
        raw_cycle_container (CyclesContainer): CyclesContainer object containing raw data
        structured_cycle_container (Optional[CyclesContainer], optional): CyclesContainer object containing structured data. Defaults to None.
        metadata (dict, optional): Dictionary of metadata. Defaults to None.
        schema (dict, optional): Dictionary to perform validation.
        paths (dict, optional): Dictionary of paths from which this object was derived. 
            Useful for keeping track of what Run file corresponds with what cycler run
            output file.
    """
    # Set the default datatypes for ingestion and raw data
    # to the same used for interpolated data (for simplicity)
    DEFAULT_DTYPES = CONTAINER_CONFIG_DEFAULT["dtypes"]

    # Basic LiFePO4 validation, as a backup/default
    DEFAULT_VALIDATION_SCHEMA = {
        'charge_capacity': {
            'schema': {
                'max': 2.0,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'cycle_index': {
            'schema': {
                'min': 0,
                'max_at_least': 1,
                'type': 'integer'
            },
            'type': 'list'
        },
        'discharge_capacity': {
            'schema': {
                'max': 2.0,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'temperature': {
            'schema': {
                'max': 80.0,
                'min': 20.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'test_time': {
            'schema': {
                'type': 'float'
            },
            'type': 'list'
        },
        'voltage': {
            'schema': {
                'max': 3.8,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        }
    }

    # Data types for all summaries (diag and regular)
    SUMMARY_DTYPES = {
        'cycle_index': 'int32',
        'discharge_capacity': 'float64',
        'charge_capacity': 'float64',
        'discharge_energy': 'float64',
        'charge_energy': 'float64',
        'dc_internal_resistance': 'float32',
        'temperature_maximum': 'float32',
        'temperature_average': 'float32',
        'temperature_minimum': 'float32',
        'date_time_iso': 'object',
        'energy_efficiency': 'float32',
        'charge_throughput': 'float32',
        'energy_throughput': 'float32',
        'charge_duration': 'float32',
        'time_temperature_integrated': 'float64',
        'paused': 'int32',
        'CV_time': 'float32',
        'CV_current': 'float32',
        'CV_capacity': 'float32',
        "coulombic_efficiency": "float64"
    }

    def __init__(
            self,
            raw_cycle_container: Optional[CyclesContainer] = None,
            structured_cycle_container: Optional[CyclesContainer] = None,
            diagnostic: Optional[DiagnosticConfig] = None,
            metadata: Optional[dict] = None,
            schema: Optional[dict] = None,
            paths: Optional[dict] = None,
            summary_regular: Optional[pd.DataFrame] = None,
            summary_diagnostic: Optional[pd.DataFrame] = None,
    ):
        self.raw = raw_cycle_container
        self.structured = structured_cycle_container
        if diagnostic:
            # This is needed because the setter for diagnostic
            self.diagnostic = diagnostic
        else:
            self._diagnostic = None
        self.paths = paths if paths else {}
        self.schema = schema if schema else self.DEFAULT_VALIDATION_SCHEMA
        self.metadata = metadata if metadata else {}
        self.summary_regular = summary_regular
        self.summary_diagnostic = summary_diagnostic
    
    def __repr__(self) -> str:
        has_raw = True if self.raw else False
        has_structured = True if self.structured else False
        has_diagnostic = True if self.diagnostic else False
        from_path = self.paths.get("raw", "unknown")
        return f"{self.__class__.__name__} (" \
            f"raw={has_raw}, structured={has_structured}, diagnostic={has_diagnostic})"\
            f" from {from_path}"

    def validate(self):
        """
        Validate the run object against the validation schema.
        If a validation schema is not passed to __init__, a default is used.
        """
        logger.warning("Validation requires loading entire df into memory!")
        validator = SimpleValidator(self.schema)
        is_valid, reason = validator.validate(self.raw.cycles.data)
        return is_valid, reason

    def structure(
            self,
            summarize_regular: bool = True,
            summarize_diagnostic: bool = True,
            summarize_regular_kwargs: Optional[dict] = None,
            summarize_diagnostic_kwargs: Optional[dict] = None,
    ):
        """
        Interpolate cycles and steps according to their configurations
        and generate summaries.

        After structure has been run, the Run.structured CyclesContainer
        and Run.summary_regular and Run.summary_diagnostic DataFrames
        should be accessible.

        Args:
            summarize_regular_kwargs (dict): Dictionary of kwargs to pass
                to summarize_regular.
            summarize_diagnostic_kwargs (dict): Dictionary of kwargs to pass\
                to summarize_diagnostic.

        Returns:
            None
        """
        pbar = ProgressBar(dt=1, width=10)
        pbar.register()
        cycles_interpolated = bag.from_sequence(
            bag.map(
                interpolate_cycle, 
                self.raw.cycles.items,
                # remaining kwargs are broadcast to all calls
                cconfig=self.raw.config
            ).remove(lambda xdf: xdf is None).compute()
        )
        cycles_interpolated = cycles_interpolated.\
            repartition(npartitions=cycles_interpolated.count().compute())
        self.structured = CyclesContainer(cycles_interpolated)
        pbar.unregister()

        if summarize_regular_kwargs is None:
            summarize_regular_kwargs = {}
        if summarize_diagnostic_kwargs is None:
            summarize_diagnostic_kwargs = {}

        if summarize_regular:
            self.summary_regular = self.summarize_regular(**summarize_regular_kwargs)

        if summarize_diagnostic:
            if self.diagnostic:
                self.summary_diagnostic = self.summarize_diagnostic(**summarize_diagnostic_kwargs)
            else:
                logger.debug("No diagnostic; cannot summarize diagnostic.")

    # Diagnostic config methods
    @property
    def diagnostic(self):
        return self._diagnostic

    @diagnostic.setter
    def diagnostic(self, diagnostic_config: DiagnosticConfig):
        if not isinstance(diagnostic_config, DiagnosticConfig):
            logger.warning(
                f"Diagnostic config passed does not inherit "
                "DiagnosticConfig, can cause downstream errors."
            )
        self._diagnostic = diagnostic_config
        for cycle in tqdm.tqdm(
            self.raw.cycles, 
            total=self.raw.cycles.items_length,
            desc="Updating cycle labels based on diagnostic config",
            **TQDM_STYLE_ARGS
        ):
            for step in cycle.steps:
                step.data["cycle_label"] = step.data["cycle_index"].apply(
                    lambda cix: diagnostic_config.type_by_ix.get(cix, "regular")
            )    

    @diagnostic.deleter
    def diagnostic(self):
        del self._diagnostic

    # Serialization methods
    # maybe do not need to do this for this class according to monty docstrings
    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a Run object from a dictionary.
        """
        d = {k: MontyDecoder().process_decoded(v) for k, v in d.items()}
        for k in ("summary_regular", "summary_diagnostic"):
            d[k] = pd.DataFrame(d[k]) if d[k] is not None else None
        return cls(**d)

    def as_dict(self):
        """
        Convert a Run object to a dictionary.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "raw_cycle_container": self.raw.as_dict(),
            "structured_cycle_container": self.structured.as_dict() if self.structured else None,
            "summary_regular": self.summary_regular.to_dict("list") if self.summary_regular is not None else None,
            "summary_diagnostic": self.summary_diagnostic.to_dict("list") if self.summary_diagnostic is not None else None,
            "diagnostic": self.diagnostic.as_dict() if self.diagnostic else None,
            "metadata": jsanitize(self.metadata),
            "schema": self.schema,
            "paths": self.paths
        }

    # Convenience methods for loading and saving
    @classmethod
    def load(cls, path: Union[str, os.PathLike]):
        """
        Load a Run object from a file or list of files.
        """
        return loadfn(path)
    
    def save(self, path: Union[str, os.PathLike]):
        """
        Save a Run object to a file.
        """
        dumpfn(self, path)

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        **kwargs
    ):
        """
        Convenience method to create an unstructured Run object from a raw dataframe.

        Assumes step_index is already in data.
        """
        raw = CyclesContainer.from_dataframe(
            df,
            diagnostic=kwargs.get("diagnostic", None),
            tqdm_desc_suffix="(raw)"
        )
        return cls(raw_cycle_container=raw, **kwargs)

    def summarize_regular(
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

        Note: Tries to avoid loading large dfs into memory at any given time
        by using dask. Small dfs are permissible.

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
        agg = {
            "cycle_index": "first",
            "discharge_capacity": "max",
            "charge_capacity": "max",
            "discharge_energy": "max",
            "charge_energy": "max",
            "internal_resistance": "last",
            "date_time_iso": "first",
            "test_time": "first"
        }

        # Aggregate and format a potentially large dataframe
        raw_lazy_df = self.raw.cycles["regular"].data_lazy
        summary = raw_lazy_df. \
            groupby("cycle_index").agg(agg). \
            rename(columns={"internal_resistance": "dc_internal_resistance"})
        summary = summary.compute()

        summary["energy_efficiency"] = \
                summary["discharge_energy"] / summary["charge_energy"]
        summary.loc[
            ~np.isfinite(summary["energy_efficiency"]), "energy_efficiency"
        ] = np.NaN
        # This code is designed to remove erroneous energy values
        for col in ["discharge_energy", "charge_energy"]:
            summary.loc[summary[col].abs() > error_threshold, col] = np.NaN
        summary["charge_throughput"] = summary.charge_capacity.cumsum()
        summary["energy_throughput"] = summary.charge_energy.cumsum()

        # Computing charge durations
        # This method for computing charge start and end times implicitly
        # assumes that a cycle starts with a charge step and is then followed
        # by discharge step.
        charge_start_time = raw_lazy_df. \
            groupby("cycle_index")["date_time_iso"]. \
            agg("first").compute().to_frame()
        charge_finish_time = raw_lazy_df \
            [raw_lazy_df.charge_capacity >= nominal_capacity * full_fast_charge]. \
            groupby("cycle_index")["date_time_iso"]. \
            agg("first").compute().to_frame()

        # Left merge, since some cells might not reach desired levels of
        # charge_capacity and will have NaN for charge duration
        merged = charge_start_time.merge(
            charge_finish_time, on="cycle_index", how="left"
        )

        # Charge duration stored in seconds -
        # note that date_time_iso is only ~1sec resolution
        time_diff = np.subtract(
            pd.to_datetime(merged.date_time_iso_y, utc=True, errors="coerce"),
            pd.to_datetime(merged.date_time_iso_x, utc=True, errors="coerce"),
        )
        summary["charge_duration"] = np.round(
            time_diff / np.timedelta64(1, "s"), 2)

        # Compute time-temeprature integral, if available
        if "temperature" in raw_lazy_df.columns:
            # Compute time since start of cycle in minutes. This comes handy
            # for featurizing time-temperature integral
            raw_lazy_df["time_since_cycle_start"] = \
                raw_lazy_df["date_time_iso"].apply(pd.to_datetime) - \
                raw_lazy_df.groupby("cycle_index")["date_time_iso"].transform("first")

            raw_lazy_df["time_since_cycle_start"] = \
                (raw_lazy_df["time_since_cycle_start"] / np.timedelta64(1, "s")) / 60

            # Group by cycle index and integrate time-temperature
            summary["time_temperature_integrated"] = raw_lazy_df.groupby(
                "cycle_index").apply(
                lambda g: np.integrate.trapz(
                    g.temperature,
                    x=g.time_since_cycle_start
                ).compute()
            )
            raw_lazy_df.drop(columns=["time_since_cycle_start"]).compute()

        # Determine if any of the cycles has been paused
        summary["paused"] = raw_lazy_df.groupby("cycle_index").apply(
            get_max_paused_over_threshold,
            meta=pd.Series([], dtype=float)).compute()

        # Find CV step data
        cv_data = []
        for cyc in self.raw.cycles["regular"]:
            cix = cyc.cycle_index
            try:
                cv = get_cv_stats(cyc.steps["charge"].data)
            except (CVStatsError, DFSelectorIndexError):
                logger.debug(f"Cannot extract CV charge segment for cycle {cix}!")
                continue
            cv_data.append(cv)
        cv_summary = pd.DataFrame(cv_data).set_index("cycle_index")
        summary = summary.merge(
            cv_summary,
            how="outer",
            right_index=True,
            left_index=True
        ).set_index("cycle_index")

        summary = summary.astype(
            {c: v for c, v in self.SUMMARY_DTYPES.items() if c in summary.columns}
        )

        # Avoid returning empty summary dataframe for single cycle raw_data
        if summary.shape[0] == 1:
            return summary

        # Ensure final cycle has actually been completed; if not, exclude it.
        last_cycle = raw_lazy_df.cycle_index.max().compute().item()
        last_voltages = self.raw.cycles[last_cycle].data.voltage
        min_voltage_ok = last_voltages.min() < cycle_complete_vmin
        max_voltage_ok = last_voltages.max() > cycle_complete_vmax
        dchg_ratio_ok = (summary.iloc[[-1]])["discharge_capacity"].iloc[0] \
            > cycle_complete_discharge_ratio \
            * ((summary.iloc[[-1]])["charge_capacity"].iloc[0])

        if all([min_voltage_ok, max_voltage_ok, dchg_ratio_ok]):
            return summary
        else:
            return summary.iloc[:-1]
        
    def summarize_diagnostic(
            self,
            cccv_step_location=0
    ):
        # todo: cccv_step_location is only one step
        # todo: and is not generalizable to more than one cccv step
        """
        Gets summary statistics for data according to location of
        diagnostic cycles in the data

        Args:
            cccv_step_location (int): Raw index of CCCV step in the
                diagnostic. Currently limited to one step.

        Returns:
            (DataFrame) of summary statistics by cycle
        """

        raw = self.raw.cycles[tuple(self.diagnostic.all_ix)]
        raw_lazy_df = raw.data_lazy
        
        agg = {
            "cycle_index": "first",
            "discharge_capacity": "max",
            "charge_capacity": "max",
            "discharge_energy": "max",
            "charge_energy": "max",
            "date_time_iso": "first",
            "test_time": "first"
        }
        
        summary = raw_lazy_df.groupby("cycle_index").agg(agg).compute()
        summary["coulombic_efficiency"] = (
            summary["discharge_capacity"] / summary["charge_capacity"]
        )
        summary["paused"] = raw_lazy_df.groupby("cycle_index").apply(
            get_max_paused_over_threshold,
            meta=pd.Series([], dtype=float)).compute()
        
        summary["cycle_type"] = [
            self.diagnostic.type_by_ix[cix] for cix in summary["cycle_index"]
        ]

        # Find CV step data
        cv_data = []
        for cyc in raw:
            cix = cyc.cycle_index
            try:
                cv = get_cv_stats(cyc.steps.by_raw_index(cccv_step_location).data)
            except (CVStatsError, DFSelectorIndexError):
                logger.debug(f"Cannot extract CV charge segment for cycle {cix}!")
                continue
            cv_data.append(cv)
        cv_summary = pd.DataFrame(cv_data).set_index("cycle_index")
        summary = summary.merge(
            cv_summary,
            how="outer",
            right_index=True,
            left_index=True
        ).set_index("cycle_index")
        return summary.astype(
            {c: v for c, v in self.SUMMARY_DTYPES.items() if c in summary.columns}
        )

    # Common, non-feature based on demand summaries for cycle life.
    def get_cycles_to_capacities(
        self, 
        cycle_min=200, 
        cycle_max=1800,
        cycle_interval=200
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
        
        if self.summary_regular is None:
            raise ValueError(
                "Summary has not been computed. Run.structure() "
                "or set Run.summary_regular manually. "
            )
        discharge_capacities = pd.DataFrame(
            np.zeros((1, int((cycle_max - cycle_min) / cycle_interval)))
        )
        counter = 0
        cycle_indices = np.arange(cycle_min, cycle_max, cycle_interval)
        for cycle_index in cycle_indices:
            try:
                discharge_capacities[counter] = \
                    self.summary_regular.discharge_capacity.iloc[cycle_index]
            except IndexError:
                pass
            counter = counter + 1

        discharge_capacities = discharge_capacities.transpose()
        discharge_capacities.columns = ["discharge_capacity"]
        discharge_capacities["cycle"] = cycle_indices
        return discharge_capacities

    def get_capacities_to_cycles(
            self,
            thresh_max_cap=0.98,
            thresh_min_cap=0.78,
            interval_cap=0.03
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
        cycles = cycles.transpose()
        cycles.columns = ["cycle"]
        cycles["discharge_capacity_fraction"] = threshold_list
        return cycles

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
        if self.summary_regular is None:
            raise ValueError(
                "Summary has not been computed. Run.structure() "
                "or set Run.summary_regular manually. "
            )

        # discharge_capacity has a spike and  then increases slightly between \
        # 1-40 cycles, so let us use take median of 1st 40 cycles for max.

        # If n_cycles <  n_cycles_cutoff, do not use median method

        if len(self.summary_regular) > n_cycles_cutoff:
            max_capacity = np.median(
                self.summary_regular.discharge_capacity.iloc[0:n_cycles_cutoff]
            )
        else:
            logger.warning(
                "Cycle max capacity could not be computed with median method! "
                "Falling back to nominal capacity of 1.1")
            max_capacity = 1.1

        # If capacity falls below 80% of initial capacity by end of run
        if (self.summary_regular.discharge_capacity.iloc[
                -1] / max_capacity) <= threshold:
            cycle_life = self.summary_regular[
                self.summary_regular.discharge_capacity < threshold * max_capacity
                ].index[0]
        else:
            # Some cells do not degrade below the threshold (low degradation rate)
            cycle_life = len(self.summary_regular) + 1
        return cycle_life
