"""Classes for representing sets of cycles for battery cycler data.
"""
from typing import Iterable

import pandas as pd
from tqdm import tqdm
import dask.bag as bag
from monty.json import MSONable, MontyDecoder

from beep.structure.core.util import DFSelectorAggregator, aggregate_nicely, TQDM_STYLE_ARGS
from beep.structure.core.step import Step
from beep.structure.core.cycle import Cycle


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

    def __init__(
            self, 
            cycles: Iterable[Cycle],
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
        n_items = self.cycles.items_length
        return f"{self.__class__.__name__}" \
            f"({n_items} cycles, {self.data.shape[0]} points)"

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        step_cls: object = Step, 
        tqdm_desc_suffix: str = ""
        ):
        groups = df.groupby("cycle_index")
        # generator comprehension to avoid loading all cycles into memory
        cycles = (
            Cycle.from_dataframe(dfc, step_cls=step_cls) for _, dfc in
            tqdm(
                groups,
                desc=f"Organizing cycles and steps {tqdm_desc_suffix}",
                **TQDM_STYLE_ARGS
            )
        )
        # return cls(cycles)
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