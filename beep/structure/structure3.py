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

"""


# todo: step_index and other constant columns still get interpolated when using constant_n_points_per_step_label

TQDM_STYLE_ARGS = {
    "ascii": ' ='
}

CONFIG_STEP_DEFAULT = {
    "field_name": "voltage",
    "field_range": None,
    "columns": None,
    "resolution": 1000,
    "exclude": False,
    "min_points": 2
}

CONFIG_CYCLE_DEFAULT = {
    "constant_n_points_per_step": True,
    "constant_n_points_per_step_label": False,
    "config_per_step_label":
        {
            "charge": copy.deepcopy(CONFIG_STEP_DEFAULT),
            "discharge": copy.deepcopy(CONFIG_STEP_DEFAULT)
        }
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
    A persistent step object.
    This is where all ground truth data is kept.
    """

    def __init__(self, df_step, lenient=True):
        self.data = df_step
        self.step_counter_absolute = None
        self.step_counter = None
        self.step_index = None
        self.step_label = None
        self.cycle_index = None
        self.config = {}

        chklist = ("step_counter_absolute", "step_counter", "step_index", "cycle_index", "step_label")

        for chk in chklist:
            unique = self.data[chk].unique()
            if len(unique) != 1:
                if lenient:
                    warnings.warn(f"Unique values for {chk} of {self.__class__.__name__} were {unique}! Assigning {self.__class__.__name__} attribute '{chk}' to None.")
                    setattr(self, chk, None)
                else:
                    raise ValueError(
                        f"Step check failed; '{chk}' has more than one unique value (lenient={lenient})!\n{unique}"
                    )
            else:
                setattr(self, chk, unique[0])

    def __repr__(self):
        return f"{self.__class__.__name__} " \
               f"(cycle_index={self.cycle_index}, "\
               f"step_counter={self.step_counter}, " \
               f"step_index={self.step_index}, " \
               f"step_label={self.step_label}, " \
               f"{self.data.shape[0]} points)"


class Cycle:
    """
    A persistent cycle object.
    """

    def __init__(self, df_cyc, lenient=True):
        steps = [Step(scdf, lenient=lenient) for _, scdf in df_cyc.groupby("step_counter")]
        self.steps = DFSelectorAggregator(
            items=steps,
            index_field="step_index",
            slice_field="step_index",
            tuple_field="step_index",
            label_field="step_label",
        )
        self.config = {}

        unique = self.data.cycle_index.unique()
        if len(unique) != 1:
            raise ValueError(
                f"Cycle check failed; 'cycle_index' has more than one unique value!\n{unique}"
            )
        else:
            self.cycle_index = unique

    def __repr__(self):
        return f"{self.__class__.__name__} {self.cycle_index}({len(self.steps)} steps, {self.data.shape[0]} points)"

    @property
    def data(self):
        return aggregate_nicely([s.data for s in self.steps])


class Run:
    """
    An entire cycler run.
    A persistent object to replace BEEPDatapath.
    """
    class CyclesContainer:
        def __init__(self, df, tqdm_desc_suffix=None, lenient=True):
            cycles = [Cycle(dfc, lenient=lenient) for _, dfc in
                      tqdm.tqdm(df.groupby("cycle_index"),
                                desc=f"Organizing cycles and steps {tqdm_desc_suffix}",
                                **TQDM_STYLE_ARGS)]
            # groups = df.groupby("cycle_index")
            # cycle_bag = db.from_sequence(groups, npartitions=len(groups))
            # cycles = db.map(Cycle.__init__, cycle_bag).compute()
            self.cycles = DFSelectorAggregator(
                items=cycles,
                index_field="cycle_index",
                slice_field="cycle_index",
                tuple_field="cycle_index",
                label_field="cycle_label",
            )
        @property
        def data(self):
            return aggregate_nicely([c.data for c in self.cycles])

    def __init__(self, df):
        df2 = copy.deepcopy(df)

        # Assign a per-cycle step index counter
        df2.loc[:, "step_counter"] = 0
        for cycle_index in tqdm.tqdm(df2.cycle_index.unique(),
                                     desc="Assigning step counter",
                                     **TQDM_STYLE_ARGS):
            indices = df2.loc[df2.cycle_index == cycle_index].index
            step_index_list = df2.step_index.loc[indices]
            shifted = step_index_list.ne(step_index_list.shift()).cumsum()
            df2.loc[indices, "step_counter"] = shifted - 1

        # Assign an absolute step index counter
        compounded_counter = df2.step_counter.astype(str) + "-" + df2.cycle_index.astype(str)
        absolute_shifted = compounded_counter.ne(compounded_counter.shift()).cumsum()
        df2["step_counter_absolute"] = absolute_shifted - 1

        # Assign step label if not known
        if "step_label" not in df2.columns:
            df2["step_label"] = None
            for sca, df_sca in tqdm.tqdm(df2.groupby("step_counter_absolute"),
                                         desc="Determining charge/discharge steps",
                                         **TQDM_STYLE_ARGS):
                indices = df_sca.index
                df2.loc[indices, "step_label"] = label_chg_state(df_sca)

        # Assign cycle label
        self.raw = self.CyclesContainer(df2, tqdm_desc_suffix="(raw)", lenient=False)
        self.structured = None

    def structure(self):
        ProgressBar().register()
        cycle_bag = db.from_sequence(self.raw.cycles, npartitions=len(self.raw.cycles))
        dfs = db.map(interpolate_cycle, cycle_bag).compute()
        dfs = [df for df in dfs if not df.empty]
        self.structured = self.CyclesContainer(pd.concat(dfs),
                                               tqdm_desc_suffix="(structured)")


def interpolate_cycle(cycle):
    config = update_nested(
        copy.deepcopy(CONFIG_CYCLE_DEFAULT),
        cycle.config
    )

    if sum([config["constant_n_points_per_step"],
            config["constant_n_points_per_step_label"]]) != 1:
        raise ValueError(
            f"A constant number of points must either be specified per step "
            f"or per chgstate for a given cycle, but the following was passed\n{config}")

    # constant number of points per step
    if config["constant_n_points_per_step"]:
        steps = cycle.steps
        constant_columns = ["step_index", "step_counter",
                            "step_counter_absolute", "cycle_index",
                            "step_label", "cycle_label"]

    # else, constant number of points per chgstate (step_label)
    else:
        # Create new "Steps" based on multiple steps grouped by their step labels
        steps = []
        for step_label, df in cycle.data.groupby("step_label"):
            step = Step(df, lenient=True)
            step.config = config["config_per_step_label"].get(step_label, {"exclude": True})

            if step_label not in config["config_per_step_label"]:
                warnings.warn(
                    f"Step label '{step_label}' for step-label based interpolation was "
                    f"not found in the config! Excluding from interpolation. Config was:\n{config}"
                )

            steps.append(step)
        constant_columns = ["cycle_index", "step_label", "cycle_label"]

    interpolated_dfs = []
    for step in steps:
        dataframe = step.data

        # todo: need to implement some sort of field range checking, as it
        # is really sneaky when some fields have no range (because it only looks at first and last, not min-max)
        # i.e., something like "Exclude bad interpolation" and then checks length of interpolated df

        sconfig = update_nested(
            copy.deepcopy(CONFIG_STEP_DEFAULT),
            step.config
        )

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
        interpolated_dfs.append(interpolated_df)

    return pd.concat(interpolated_dfs) if interpolated_dfs else pd.DataFrame()


def update_nested(d, u):
    """
    Graciously taken from
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def label_chg_state(step_df, indeterminate_label="unknown"):
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


def aggregate_nicely(iterable):
    if iterable:
        return pd.concat(iterable).sort_values(by="test_time").reset_index()
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    from beep.structure import MaccorDatapath
    # from beep.structure.structure3 import Run
    filename = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092"
    md = MaccorDatapath.from_file(filename)

    run = Run(md.raw_data)

    run.structure()


    # Now structure based on a constant number of points for each charge/discharge cycle
    for cyc in run.raw.cycles:
        cyc.config = {
            "constant_n_points_per_step": False,
            "constant_n_points_per_step_label": True,
            "config_per_step_label": {
                "charge": {"resolution": 1000},
                "discharge": {"resolution": 1000},
                "unknown": {"exclude": True}
            }
        }

    run.structure()


