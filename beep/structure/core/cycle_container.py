"""Classes for representing sets of cycles for battery cycler data.
"""
from typing import Iterable

from tqdm import tqdm
import dask.bag as bag

from beep.structure.core.util import DFSelectorAggregator, aggregate_nicely, TQDM_STYLE_ARGS
from beep.structure.core.step import Step
from beep.structure.core.cycle import Cycle

class CyclesContainer:
    """
    A container for many cycles. 

    Args:
        cycles (Iterable[Cycle]): An iterable of Cycle objects.
    
    Attributes:
        data (pd.DataFrame): All data for this container, organized in presentable format.
        cycles (DFSelectorAggregator): An object that allows for easy selection of cycles.
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

    @classmethod
    def from_dataframe(cls, df, step_cls=Step, tqdm_desc_suffix: str = ""):
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