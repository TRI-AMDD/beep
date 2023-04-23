from typing import Optional, Iterable, Union

import dask.bag as db
import collections.abc
import warnings
import pprint
import dataclasses
import abc
import pandas as pd
from beep.structure import MaccorDatapath
import copy
import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
import dask
import tqdm

from dask.diagnostics import ProgressBar


from beep.structure.diagnostic import DiagnosticConfig

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


"""


# todo: step_index and other constant columns still get interpolated when using constant_n_points_per_step_label


# Defining all common structured datatypes
# This set of columns
DATA_COLUMN_DTYPES = {
    'test_time': 'float64',              # Total time of the test
    'cycle_index': 'int32',              # Index of the cycle
    'cycle_label': 'category',           # Label of the cycle - default="regular"
    'current': 'float32',                # Current
    'voltage': 'float32',                # Voltage
    'temperature': 'float32',            # Temperature of the cell
    'internal_resistance': 'float32',    # Internal resistance of the cell
    'charge_capacity': 'float32',        # Charge capacity of the cell
    'discharge_capacity': 'float32',     # Discharge capacity of the cell
    'charge_energy': 'float32',          # Charge energy of the cell
    'discharge_energy': 'float32',
    'step_index': 'int16',
    'step_counter': 'int32',
    'step_counter_absolute': 'int32',
    'step_label': 'category',
    'datum': 'int32',
    }


TQDM_STYLE_ARGS = {
    "ascii": ' ='
}

# Step level config ONLY
CONFIG_STEP_DEFAULT = {
    "field_name": "voltage",
    "field_range": None,
    "columns": list(DATA_COLUMN_DTYPES.keys()),
    "resolution": 1000,
    "exclude": False,
    "min_points": 2
}

# Cycle level config ONLY
CONFIG_CYCLE_DEFAULT = {
    # Config mode 1: Constant n point per step within a cycle. 
    # k steps within a cycle will result in n*k points per cycle.

    # Config for mode 2: Constant n points per step label within a cycle, 
    # regardless of k steps in cycle.
    # for $i \in S$ step labels, $n_i$ points per step label, will result in $\sum_i n_i$ points per cycle.
    # Note: for temporally disparate steps with the same steps label, strange behavior can occur. 
    "preaggregate_steps_by_step_label": False,
}


class DFSelectorAggregator:
    """
    Class for aggregating the following from lists of Cycles/Steps:
        - individual indices
        - tuples of indices
        - slices of indices
        - labels on pre-defined dataframe fields

    ... into either a list of items (to be iterated) or a data frame (which is aggregated)
    """
    def __init__(
            self,
            items,
            index_field,
            tuple_field,
            slice_field,
            label_field
    ):
        self.index_field = index_field
        self.tuple_field = tuple_field
        self.label_field = label_field
        self.slice_field = slice_field
        self.items = items

    def __getitem__(self, indexer):
        if isinstance(indexer, int):
            item_selection = [i for i in self.items if getattr(i, self.index_field) == indexer]
        elif isinstance(indexer, tuple):
            item_selection = [i for i in self.items if getattr(i, self.tuple_field) in indexer]
        elif isinstance(indexer, slice):
            indexer = tuple(range(*indexer.indices(len(self.items))))
            item_selection = [i for i in self.items if getattr(i, self.slice_field) in indexer]
        elif isinstance(indexer, str):
            item_selection = [i for i in self.items if getattr(i, self.label_field) == indexer]
        else:
            raise TypeError(
                f"No indexing scheme for {self.__class__.__name__} available for type {type(indexer)}")

        if len(item_selection) == 1:
            return item_selection[0]
        else:
            return DFSelectorAggregator(
                item_selection,
                self.index_field,
                self.tuple_field,
                self.slice_field,
                self.label_field
            )

    def __iter__(self):
        for item in self.items:
            yield item

    def __len__(self):
        return len(self.items)

    def __getattr__(self, attr):
        if attr == "steps" and all([isinstance(item, Cycle) for item in self]):
            # If we are trying to get steps for a selection of cycles,
            # We need to get all those steps' data and aggregate them
            steps = []
            for c in self:
                for s in c.steps:
                    steps.append(s)
            return DFSelectorAggregator(
                steps,
                index_field="step_index",
                slice_field="step_index",
                tuple_field="step_index",
                label_field="step_label"
            )
        
        # Return a single items attribute, e.g. a config for a single step or cycle.
        elif len(self.__getattribute__("items")) == 1:
            return getattr(self.__getattribute__("items")[0], attr)
        else:
            # default behavior
            return self.__getattribute__(attr)

    def __setattr__(self, key, value):
        if key == "config":
            for item in self.items:
                item.config = value
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return self.items.__repr__()

    @property
    def data(self):
        return aggregate_nicely([obj.data for obj in self.items])


class Step:
    """
    A persistent step object, basically a wrapper around a step's dataframe.
    Requires several columns to only have one unique value.
    This is where all ground truth data is kept.
    """

    def __init__(self, df_step: pd.DataFrame):
        self.data = df_step
        self.config = {}

        self.uniques = (
            "step_counter_absolute", 
            "step_counter", 
            "step_index", 
            "step_label",
            "cycle_index",
            "cycle_label"
            )

        # Ensure step cannot be instantiated while failing uniques check
        for attr in self.uniques:
            getattr(self, attr)

    def __repr__(self):
        return f"{self.__class__.__name__} " \
               f"(cycle_index={self.cycle_index}, "\
               f"step_counter={self.step_counter}, " \
               f"step_index={self.step_index}, " \
               f"step_label={self.step_label}, " \
               f"{self.data.shape[0]} points)"
    
    def __getattr__(self, attr):
        if attr in self.__getattribute__("uniques"):
            uq = self.__getattribute__("data")[attr].unique()
            if len(uq) == 1:
                return uq[0]
            else:
                raise ValueError(f"Step check failed; '{attr}' has more than one unique value ({uq})!")
        else:
            return self.__getattribute__(attr)


class MultiStep(Step):
    """
    A persistent object holding multiple steps, that should generally
    act as a step, but can hold data containing multiple unique values
    for columns that would generally only have one unique value.

    However, a MultiStep still has some fields that must contain only one unique value, 
    so it will throw an error if there are multiple values for these fields.
    """

    # TODO: Step indexing does not work on multisteps!

    def __init__(self, df_multistep: pd.DataFrame):
        self.mandatory_uniques = (
            "step_label",
            "cycle_index",
            "cycle_label"
        )

        super().__init__(df_multistep)

        # Ensure multistep cannot be instantiated while failing mandatory uniques
        for attr in self.mandatory_uniques:
            getattr(self, attr)
        

    def __getattr__(self, attr):
        if attr in self.__getattribute__("uniques"):
            l = list(self.__getattribute__("data")[attr].unique())
            if attr in self.__getattribute__("mandatory_uniques"):
                if len(l) != 1:
                    raise ValueError(f"MultiStep check failed; '{attr}' does not have unique value ({l})!")
                else:
                    return l[0]
            else:
                return l
        else:
            return super().__getattr__(attr)


class Cycle:
    """
    A persistent cycle object. A wrapper for many step objects.
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
        return f"{self.__class__.__name__} {self.cycle_index} ({self.cycle_label} cycle, {len(self.steps)} steps incl. {set([s.step_label for s in self.steps])}, {self.data.shape[0]} points)"

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


class CyclesContainer:
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
        cycles = [
            Cycle.from_dataframe(dfc, step_cls=step_cls) for _, dfc in
            tqdm.tqdm(
                df.groupby("cycle_index"),
                desc=f"Organizing cycles and steps {tqdm_desc_suffix}",
                **TQDM_STYLE_ARGS
            )
        ]
        return cls(cycles)

    @property
    def data(self):
        return aggregate_nicely([c.data for c in self.cycles])


class Run:
    """
    An entire cycler run.
    A persistent object to replace BEEPDatapath.
    """

    def __init__(
            self,
            raw_cycle_container: CyclesContainer, 
            structured_cycle_container: Optional[CyclesContainer] = None,
            diagnostic_config: Optional[DiagnosticConfig] = None,
        ):
        self.raw = raw_cycle_container
        self.structured = structured_cycle_container
        self.diagnostic = diagnostic_config

    def structure(self, num_workers: Optional[int] = None):
        ProgressBar().register()
        cycle_bag = db.from_sequence(self.raw.cycles, npartitions=len(self.raw.cycles))
        cycles_interpolated = db.map(interpolate_cycle, cycle_bag).compute(num_workers=num_workers)
        self.structured = CyclesContainer(
            [c for c in cycles_interpolated if c is not None]
        )

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame, 
        diagnostic_config: Optional[DiagnosticConfig] = None
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
        df["cycle_index"] = df["cycle_index"].astype(DATA_COLUMN_DTYPES["cycle_index"])
        df["cycle_label"] = "regular"
        if diagnostic_config:
            df["cycle_label"] = df["cycle_index"].apply(
                lambda cix: diagnostic_config.type_by_ix.get(cix, "regular")
            )    

        df = df.astype(DATA_COLUMN_DTYPES)
        raw = CyclesContainer.from_dataframe(df, tqdm_desc_suffix="(raw)")
        return cls(raw, diagnostic_config=diagnostic_config)


def interpolate_cycle(cycle: Cycle) -> Cycle:
    config = copy.deepcopy(CONFIG_CYCLE_DEFAULT).update(cycle.config)

    preaggregate = config["preaggregate_steps_by_step_label"]

    if preaggregate:
        # Create new "Steps" based on multiple steps grouped by their step labels
        steps = []
        for step_label, df in cycle.data.groupby("step_label"):
            new_step = MultiStep(df)

            # NOTE: we assume that if pre-aggregating, we can use the first matching
            # step (acc. to label) to obtain the config for the multistep.
            first_matching_step = [step for step in cycle.steps if step.step_label == step_label][0]
            new_step.config = copy.deepcopy(first_matching_step.config)
            steps.append(new_step)
        constant_columns = ["cycle_index", "step_label", "cycle_label"]
        step_cls = MultiStep

    # constant number of points per step
    else:
        steps = cycle.steps
        constant_columns = ["step_index", "step_counter",
                            "step_counter_absolute", "cycle_index",
                            "step_label", "cycle_label"]
        step_cls = Step

    interpolated_steps = []
    for step in steps:
        dataframe = step.data

        # todo: need to implement some sort of field range checking, as it
        # is really sneaky when some fields have no range (because it only looks at first and last, not min-max)
        # i.e., something like "Exclude bad interpolation" and then checks length of interpolated df

        sconfig = copy.deepcopy(CONFIG_STEP_DEFAULT).update(step.config)

        resolution = sconfig["resolution"]
        field_name = sconfig["field_name"]
        field_range = sconfig["field_range"]
        min_points = sconfig["min_points"]
        columns = sconfig["columns"]
        exclude = sconfig["exclude"]

        if dataframe.shape[0] <= min_points or exclude:
            continue

        # TODO: Fix this unneeded dropping when actually merging in
        droppables = [c for c in dataframe.columns if c.startswith("_")] + ["date_time", "date_time_iso"]
        dataframe = dataframe.drop(columns=droppables)

        # at this point we assume all the values are unique
        cc = [c for c in constant_columns if c in dataframe.columns]
        constant_column_vals = {k: dataframe[k].unique()[0] for k in cc}
        dataframe = dataframe.drop(columns=cc)

        columns = columns or dataframe.columns
        columns = list(set(columns) | {field_name})

        df = dataframe.loc[:, dataframe.columns.intersection(columns)]
        field_range = field_range or [df[field_name].iloc[0],
                                      df[field_name].iloc[-1]]
        
        # If interpolating on datetime, change column to datetime series and
        # use date_range to create interpolating vector
        if field_name == "date_time_iso":
            df["date_time_iso"] = pd.to_datetime(df["date_time_iso"])
            interp_x = pd.date_range(start=df[field_name].iloc[0],
                                     end=df[field_name].iloc[-1],
                                     periods=resolution)
        else:
            interp_x = np.linspace(*field_range, resolution)
        interpolated_df = pd.DataFrame(
            {field_name: interp_x, "interpolated": True})

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
        interpolated_df = interpolated_df.drop(
            ["interpolated_x", "interpolated_y"],
            axis=1)
        interpolated_df = interpolated_df.reset_index()

        # Remove duplicates
        interpolated_df = interpolated_df[~interpolated_df[field_name].duplicated()]
        for k, v in constant_column_vals.items():
            interpolated_df[k] = v

        column_types = {k: v for k, v in DATA_COLUMN_DTYPES.items() if k in interpolated_df.columns}
        interpolated_df = interpolated_df.astype(column_types)


        # Skip interpolated dfs that weren't actually interpolated
        # in case min_points does not catch something where the resulting df is like 2 points
        # or where the field is so narrow the interpolated points are the same as the original points
        if interpolated_df.shape[0] < resolution:
            continue

        # Create step classes, incl. config
        step_interpolated = step_cls(interpolated_df)
        step_interpolated.config = copy.deepcopy(step.config)
        interpolated_steps.append(step_interpolated)
    
    # cycdf = pd.concat(interpolated_dfs) if interpolated_dfs else pd.DataFrame()

    # Create a new cycle instance, incl. config
    # Ignore cycles where no steps were interpolated.
    if not interpolated_steps:
        return None
    else:
        cycle_interpolated = Cycle(interpolated_steps)
        cycle_interpolated.config = copy.deepcopy(cycle.config)
        return cycle_interpolated


def label_chg_state(step_df: pd.DataFrame, indeterminate_label: str = "unknown") -> dict:
    """
    Helper function to determine whether a given dataframe corresponding
    to a single cycle_index's step is charging or discharging, only intended
    to be used with a dataframe for single step/cycle

    Args:
         step_df (pandas.DataFrame): dataframe to determine whether
            charging or discharging
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
    return {True: "charge", False: "discharge", None: indeterminate_label}[is_charging]


def aggregate_nicely(iterable: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate a bunch of dataframes (i.e., from steps) into a single dataframe 
    that looks presentable.
    """
    if iterable:
        return pd.concat(iterable).sort_values(by="test_time").reset_index(drop=True)
    else:
        return pd.DataFrame()
