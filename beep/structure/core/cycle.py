
"""Classes for representing individual cycles for battery cycler data.
"""
from typing import Iterable, Union

from beep.structure.core.step import Step, MultiStep
from beep.structure.core.util import DFSelectorAggregator, aggregate_nicely


class Cycle:
    """
    A persistent cycle object. A wrapper for many step objects.

    Args:
        steps (Iterable[Union[Step, MultiStep]]): An iterable of Step or MultiStep objects.

    Attributes:
        data (pd.DataFrame): All data for this cycle, organized in presentable format.
        steps (DFSelectorAggregator): An object that allows for easy selection of steps.
        config (dict): A dictionary of Cycle-level interpolation parameters.
            Note this config may override or change step-level configuration.
        uniques (tuple): A tuple of columns that must be unique for a cycle to be instantiated.
    """
    # Cycle level config ONLY
    # Config mode 1: Constant n point per step within a cycle. 
    # k steps within a cycle will result in n*k points per cycle.
    # Config for mode 2: Constant n points per step label within a cycle, 
    # regardless of k steps in cycle.
    # for $i \in S$ step labels, $n_i$ points per step label, will result in $\sum_i n_i$ points per cycle.
    # Note: for temporally disparate steps with the same steps label, strange behavior can occur. 
    CONFIG_DEFAULT = {
        "preaggregate_steps_by_step_label": False,
    }
    
    def __init__(
            self, 
            steps: Iterable[Union[Step, MultiStep]]
        ):

        self.steps = DFSelectorAggregator(
            items=steps,
            index_field="step_index",
            slice_field="step_index",
            tuple_field="step_index",
            label_field="step_label",
        )

        self.config = {}

        self.uniques = (
            "cycle_index",
            "cycle_label"
        )

        # Ensure cycle cannot be instantiated failing unique check
        for attr in self.uniques:
            getattr(self, attr)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.cycle_index} " \
            f"({self.cycle_label} cycle, {len(self.steps)} steps incl. " \
            f"{set([s.step_label for s in self.steps])}, {self.data.shape[0]} points)"

    def __getattr__(self, attr):
        if attr in self.__getattribute__("uniques"):
            uq = self.__getattribute__("data")[attr].unique()
            if len(uq) == 1:
                return uq[0]
            else:
                raise ValueError(f"Cycle check failed; '{attr}' has more than one unique value ({uq})!")
        else:
            return self.__getattribute__(attr)

    @classmethod
    def from_dataframe(cls, df, step_cls=Step):
        """
        Create a Cycle object from a dataframe of a cycle.
        """
        if step_cls is Step:
            steps = [step_cls(scdf) for _, scdf in df.groupby("step_counter")]
        elif step_cls is MultiStep:
            steps = [step_cls(scdf) for _, scdf in df.groupby("step_label")]
        else:
            raise ValueError(f"{cls.__name__} not implemented for step class of {step_cls.__name__}!")
        return cls(steps)

    @property
    def data(self):
        return aggregate_nicely([s.data for s in self.steps])