"""Classes for representing sets of cycles for battery cycler data.
"""
from typing import Iterable, Union, Optional

import pandas as pd
from tqdm import tqdm
import dask
import dask.bag as bag
from dask.diagnostics import ProgressBar
import numpy as np
from monty.json import MSONable, MontyDecoder

from beep import logger
from beep.structure.diagnostic import DiagnosticConfig
from beep.structure.core.util import DFSelectorAggregator, aggregate_nicely, TQDM_STYLE_ARGS
from beep.structure.core.step import Step
from beep.structure.core.cycle import Cycle
from beep.structure.core.util import label_chg_state
from beep.structure.core.interpolate import CONTAINER_CONFIG_DEFAULT
from beep.structure.core.constants import TQDM_STRUCTURED_SUFFIX, TQDM_RAW_SUFFIX



class CyclesContainer(MSONable):
    """
    A container for many cycles. 

    Args:
        cycles (Iterable[Cycle]): An iterable of Cycle objects.
    
    Attributes:
        data (pd.DataFrame): All data for this container, organized in presentable format.
        cycles (DFSelectorAggregator): An object that allows for easy selection of cycles.
        config (dict): A dictionary of many-cycle level interpolation parameters.
    """

    DEFAULT_DTYPES = CONTAINER_CONFIG_DEFAULT["dtypes"]

    def __init__(
            self, 
            cycles: Union[Iterable[Cycle], DFSelectorAggregator],
    ):
        self.cycles = DFSelectorAggregator(
            items=cycles,
            index_field="cycle_index",
            slice_field="cycle_index",
            tuple_field="cycle_index",
            label_field="cycle_label",
        )

        self.config = {}

    def __repr__(self):
        # todo: could maybe include point count here as well,
        # todo: but computing .data is slow...
        n_items = self.cycles.items_length
        return f"{self.__class__.__name__}" \
            f"({n_items} cycles)"

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame,
        diagnostic: Optional[DiagnosticConfig] = None,
        step_cls: object = Step, 
        tqdm_desc_suffix: str = ""
    ):
        """
        Step index is a required column.

        Args:
            df:
            diagnostic:
            step_cls:
            tqdm_desc_suffix:

        Returns:

        """
        # df = dd.from_pandas(df)

        logger.info(f"Dataframe being read is {df.shape[0]} lines")

        df = df.reset_index(drop=True)

        # Assign a basic step counter
        if "step_counter" not in df.columns:
            df.loc[:, "step_counter"] = 0
            for cycle_index in tqdm(
                df.cycle_index.unique(),
                desc=f"Assigning step counter {tqdm_desc_suffix}",
                **TQDM_STYLE_ARGS
            ):
                indices = df.loc[df.cycle_index == cycle_index].index
                step_index_list = df.step_index.loc[indices]
                shifted = step_index_list.ne(step_index_list.shift()).cumsum()
                df.loc[indices, "step_counter"] = shifted - 1
        df.step_counter = df.step_counter.astype(int)

        # Assign an absolute step index counter
        if "step_counter_absolute" not in df.columns:
            compounded_counter = df.cycle_index.astype(str) + "-" + df.step_counter.astype(str)
            absolute_shifted = compounded_counter.ne(compounded_counter.shift()).cumsum()
            df["step_counter_absolute"] = absolute_shifted - 1
        df.step_counter_absolute = df.step_counter_absolute.astype(int)

        # Assign step label if not known
        if "step_label" not in df.columns:
            step_label_series = []
            df["step_label"] = None
            for sca, df_sca in tqdm(
                    df.groupby("step_counter_absolute"),
                    desc=f"Assigning compute for determining chg/dchg labels {tqdm_desc_suffix}",
                    **TQDM_STYLE_ARGS
            ):
                res = dask.delayed(
                    lambda dfl: np.repeat(
                        label_chg_state(dfl),
                        len(dfl.index)
                    )
                )(df_sca)
                step_label_series.append(res)

            pbar = ProgressBar(dt=1, width=10)
            pbar.register()
            step_label_series = dask.compute(*step_label_series)
            pbar.unregister()

            step_labels = np.concatenate(step_label_series)
            df["step_label"] = step_labels

        # Assign cycle label from diagnostic config
        df["cycle_index"] = df["cycle_index"].astype(cls.DEFAULT_DTYPES["cycle_index"])
        df["cycle_label"] = "regular"

        if "datum" not in df.columns:
            df["datum"] = df.index

        if diagnostic:
            df["cycle_label"] = df["cycle_index"].apply(
                lambda cix: diagnostic.type_by_ix.get(cix, "regular")
            )

        # Note this will not convert columns
        # not listed in the default dtypes
        dtypes = {c: dtype for c, dtype in cls.DEFAULT_DTYPES.items() if c in df.columns}
        df = df.astype(dtypes, errors="ignore")

        # todo: this could be done in parallel as well
        # generator comprehension to avoid loading all cycles into memory
        groups = df.groupby("cycle_index")
        cycles = (
            Cycle.from_dataframe(dfc, step_cls=step_cls) for _, dfc in
            tqdm(
                groups,
                desc=f"Organizing cycles and steps {tqdm_desc_suffix}",
                **TQDM_STYLE_ARGS
            )
        )

        # for i, dfc in df.groupby("cycle_index"):
        #     print(dfc)
        #     try:
        #         cyc = Cycle.from_dataframe(dfc, step_cls=step_cls)
        #     except ValueError:
        #         return dfc
        #     # raise ValueError

        return cls(bag.from_sequence(cycles, npartitions=len(groups)))

    @property
    def data(self):
        return aggregate_nicely([c.data for c in self.cycles])

    # Serialization methods required by monty
    # Note that using this will spike memory usage because we are 
    # iterating over a dask Bag
    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "cycles": [c.as_dict() for c in self.cycles],
            "config": self.config
        }

    @classmethod
    def from_dict(cls, d):
        cycles = bag.from_sequence(
            (Cycle.from_dict(c) for c in d["cycles"]),
            npartitions=len(d["cycles"])
        )
        c = cls(cycles)
        c.config = d["config"]
        return c