
"""Classes for representing individual cycles for battery cycler data.
"""
from typing import Iterable, Union
import json

from monty.json import MSONable, MontyDecoder


from beep import __version__
from beep.structure.core.step import Step, MultiStep
from beep.structure.core.util import DFSelectorAggregator, aggregate_nicely


class Cycle(MSONable):
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
        """
        Override getattr to allow for returning unique values for this Cycle.

        E.g., you can do `cycle.cycle_index` to get the cycle index because the cycle
        indices for all steps of this cycle will be the same.

        But you can NOT do `cycle.step_index` because there there are multiple steps
        with multiple step indices within a single cycle.
        """
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

    # Serialization methods required by monty
    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "@version": __version__,
            "steps": [s.as_dict() for s in self.steps],
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, d):
        dcdr = MontyDecoder()
        c = cls(
            [dcdr.process_decoded(s) for s in d["steps"]]
        )
        c.config = d["config"]
        return c