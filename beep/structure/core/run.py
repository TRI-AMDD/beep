from typing import Optional, Iterable, Union

import dask.bag as bag
import pandas as pd
import copy
import tqdm

from monty.json import MSONable
from dask.diagnostics import ProgressBar

from beep import logger
from beep.structure.diagnostic import DiagnosticConfig
from beep.structure.core.cycles_container import CyclesContainer
from beep.structure.core.util import label_chg_state, TQDM_STYLE_ARGS
from beep.structure.core.interpolate import interpolate_cycle, CONTAINER_CONFIG_DEFAULT

"""
# ASSUMPTIONS

ASSUMES:
- there is a step_type column
- there is a cycle_label column
- also should REALLY include step_counter
- input field is monotonic??1

This is why we *need to use the structured data to instantiate it later on*! But in implementation it should be put into the `__init__` when determining each of these things.


DATA IS ONLY UPDATED IN ONE DIRECTION
STEPS/CYCLES ARE FORMED FROM THE BIG DF, BUT EDITING THE CYCLES/STEPS DOES NOT AFFECT THE BIG DF
RAW STEPS/CYCLES ARE USED TO FORM THE STRUCTURED BIG DF, BUT AFTER IT IS COLLATED, THEY ARE CONVERTED ONE WAY BACK TO STEPS/CYCLES


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

    def __init__(
            self,
            raw_cycle_container: CyclesContainer, 
            structured_cycle_container: Optional[CyclesContainer] = None,
            diagnostic_config: Optional[DiagnosticConfig] = None,
            metadata: Optional[dict] = None,
            schema: Optional[dict] = None,
            paths: Optional[dict] = None,
        ):
        self.raw = raw_cycle_container
        self.structured = structured_cycle_container
        self._diagnostic = diagnostic_config

        self.paths = paths if paths else {}


        # # paths may include "raw", "metadata", and "structured", as well as others.
        # if paths:
        #     for path_ref, path in paths.items():
        #         if path and not os.path.isabs(path):
        #             raise ValueError(f"{path_ref}: '{path}' is not absolute! All paths must be absolute.")
        #     self.paths = paths
        # else:
        #     self.paths = {"raw": None}

    
    def __repr__(self) -> str:
        has_raw = True if self.raw else False
        has_structured = True if self.structured else False
        has_diagnostic = True if self.diagnostic else False
        from_path = self.paths.get("raw", "unknown")
        return f"{self.__class__.__name__} (" \
            f"raw={has_raw}, structured={has_structured}, diagnostic={has_diagnostic})"\
            f" from {from_path}"

    def structure(self):
        pbar = ProgressBar(dt=1, width=10)
        pbar.register()
        cycles_interpolated = bag.from_sequence(
            bag.map(
                interpolate_cycle, 
                self.raw.cycles.items,

                # remaining kwargs are broadcast to all calls
                cconfig=self.raw.config
            ).compute()
        )
        cycles_interpolated.repartition(npartitions=cycles_interpolated.count().compute())
        cycles_interpolated.remove(lambda xdf: xdf is None)
        self.structured = CyclesContainer(cycles_interpolated)
        pbar.unregister()

    # Diagnostic config methods
    @property
    def diagnostic(self):
        return self._diagnostic

    @diagnostic.setter
    def diagnostic_set(self, diagnostic_config: DiagnosticConfig):

        if not isinstance(diagnostic_config, DiagnosticConfig):
            logger.warning(
                f"Diagnostic config passed does not inherit "
                "DiagnosticConfig, can cause downstream errors."
            )
        self._diagnostic = diagnostic_config


    # Serialization methods
    # maybe do not need to do this for this class according to monty docstrings
    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a Run object from a dictionary.
        """
        d.pop("@module")
        d.pop("@class")
        return cls(**d)

    def as_dict(self):
        """
        Convert a Run object to a dictionary.
        """

        #todo: need to incorportate:
        # paths
        # 
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "raw_cycle_container": self.raw.as_dict(),
            "structured_cycle_container": self.structured.as_dict() if self.structured else None,
            "diagnostic_config": self.diagnostic.as_dict() if self.diagnostic else None,
        }

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        diagnostic_config: Optional[DiagnosticConfig] = None,
        **kwargs
    ):
        """
        Convenience method to create a Run object from a dataframe.
        """
        # Assign a per-cycle step index counter
        df.loc[:, "step_counter"] = 0
        for cycle_index in tqdm.tqdm(
            df.cycle_index.unique(),
            desc="Assigning step counter",
            **TQDM_STYLE_ARGS
            ):
            indices = df.loc[df.cycle_index == cycle_index].index
            step_index_list = df.step_index.loc[indices]
            shifted = step_index_list.ne(step_index_list.shift()).cumsum()
            df.loc[indices, "step_counter"] = shifted - 1

        # Assign an absolute step index counter
        compounded_counter = df.step_counter.astype(str) + "-" + df.cycle_index.astype(str)
        absolute_shifted = compounded_counter.ne(compounded_counter.shift()).cumsum()
        df["step_counter_absolute"] = absolute_shifted - 1

        # Assign step label if not known
        if "step_label" not in df.columns:
            df["step_label"] = None
            for sca, df_sca in tqdm.tqdm(df.groupby("step_counter_absolute"),
                                         desc="Determining charge/discharge steps",
                                         **TQDM_STYLE_ARGS):
                indices = df_sca.index
                df.loc[indices, "step_label"] = label_chg_state(df_sca)

        # Assign cycle label from diagnostic config
        df["cycle_index"] = df["cycle_index"].astype(cls.DEFAULT_DTYPES["cycle_index"])
        df["cycle_label"] = "regular"
        if diagnostic_config:
            df["cycle_label"] = df["cycle_index"].apply(
                lambda cix: diagnostic_config.type_by_ix.get(cix, "regular")
            )    

        if "datum" not in df.columns:
            df["datum"] = df.index

        # Note this will not convert columns
        # not listed in the default dtypes
        df = df.astype(cls.DEFAULT_DTYPES)
        raw = CyclesContainer.from_dataframe(df, tqdm_desc_suffix="(raw)")
        return cls(raw, diagnostic_config=diagnostic_config, **kwargs)