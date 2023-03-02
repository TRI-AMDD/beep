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

    return {True: "charge", False: "discharge", None: indeterminate_label}[
        is_charging]


def interpolate_df(cycle):
    config = update_nested(
        copy.deepcopy(CONFIG_CYCLE_DEFAULT),
        cycle.config
    )

    pprint.pprint(config)

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
            step.config = config["config_per_step_label"].get(step_label,
                                                              {"exclude": True})

            print(step_label, step.config)

            if step_label not in config["config_per_step_label"]:
                warnings.warn(
                    f"Step label '{step_label}' for step-label based interpolation was not found in the config! Excluding from interpolation. Config was:\n{config}")

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
        droppables = [c for c in dataframe.columns if c.startswith("_")] + [
            "date_time", "date_time_iso"]
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
        interpolated_df = interpolated_df[
            ~interpolated_df[field_name].duplicated()]
        for k, v in constant_column_vals.items():
            interpolated_df[k] = v
        interpolated_dfs.append(interpolated_df)

    return pd.concat(interpolated_dfs) if interpolated_dfs else pd.DataFrame()


class Run(CyclerDataView):
    """
    An entire cycler run.
    A persistent object to replace BEEPDatapath.
    """

    class CyclesContainer:
        def __init__(self, df, tqdm_desc_suffix=None, lenient=True):
            self.cycles = [Cycle(dfc, lenient=lenient) for _, dfc in
                           tqdm.tqdm(df.groupby("cycle_index"),
                                     desc=f"Organizing cycles and steps {tqdm_desc_suffix}",
                                     ascii=' =')]
            self.steps = []
            for c in self.cycles:
                self.steps += c.steps

        @property
        def data(self):
            return pd.concat([c.data for c in self.cycles])

    def __init__(self, df):
        df2 = copy.deepcopy(df)

        # Assign a per-cycle step index counter
        df2.loc[:, "step_counter"] = 0
        for cycle_index in tqdm.tqdm(df2.cycle_index.unique(),
                                     desc="Assigning step counter"):
            indices = df2.loc[df2.cycle_index == cycle_index].index
            step_index_list = df2.step_index.loc[indices]
            shifted = step_index_list.ne(step_index_list.shift()).cumsum()
            df2.loc[indices, "step_counter"] = shifted - 1

        # Assign an absolute step index counter
        compounded_counter = df2.step_counter.astype(
            str) + df2.cycle_index.astype(str)
        absolute_shifted = compounded_counter.ne(
            compounded_counter.shift()).cumsum()
        df2["step_counter_absolute"] = absolute_shifted - 1

        # Assign step label if not known
        if "step_label" not in df2.columns:
            df2["step_label"] = None
            for sca, df_sca in tqdm.tqdm(df2.groupby("step_counter_absolute"),
                                         desc="Determining charge/discharge steps"):
                indices = df_sca.index
                df2.loc[indices, "step_label"] = label_chg_state(df_sca)

        # Assign cycle label
        self.raw = self.CyclesContainer(df2, tqdm_desc_suffix="(raw)",
                                        lenient=False)
        self.structured = None

    def structure(self):
        cycle_bag = db.from_sequence(self.raw.cycles,
                                     npartitions=len(self.raw.cycles))
        dfs = db.map(interpolate_df, cycle_bag).compute()
        dfs = [df for df in dfs if not df.empty]
        self.structured = self.CyclesContainer(pd.concat(dfs),
                                               tqdm_desc_suffix="(structured)",
                                               lenient=True)


class Cycle:
    """
    A persistent cycle object.
    """

    def __init__(self, df_cyc, lenient=True):
        self.steps = [Step(scdf, lenient=lenient) for _, scdf in
                      df_cyc.groupby("step_counter")]
        self.config = {}

        unique = self.data.cycle_index.unique()
        if len(unique) != 1:
            raise ValueError(
                f"Cycle check failed; 'cycle_index' has more than one unique value!\n{unique}")
        else:
            self.cycle_index = unique

    @property
    def data(self):
        return pd.concat([s.data for s in self.steps])


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

        if lenient:
            chklist = ("cycle_index",)
        else:
            chklist = ("step_counter_absolute", "step_counter", "step_index",
                       "cycle_index", "step_label")

        for chk in chklist:
            unique = self.data[chk].unique()
            if len(unique) != 1:
                raise ValueError(
                    f"Step check failed; '{chk}' has more than one unique value (lenient={lenient})!\n{unique}")
            else:
                setattr(self, chk, unique[0])


if __name__ == "__main__":
    filename = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/PreDiag_000287_000128.092"
    md = MaccorDatapath.from_file(filename)