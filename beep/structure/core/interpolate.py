"""Classes, functions, and default configurations for interpolating data.
"""
import copy
from typing import Union, Dict

import pandas as pd
import numpy as np

from beep.structure.core.step import Step, MultiStep
from beep.structure.core.cycle import Cycle

# currently not used, but should be
MINIMAL_COLUMNS = [
    "test_time",
    "cycle_index",
    "cycle_label",
    "step_index",
    "step_label",
    "step_counter",
    "step_counter_absolute",
    "datum"
]

# CycleContainer level config ONLY
CONTAINER_CONFIG_DEFAULT = {
    "dtypes": {
        'test_time': 'float64',              # Total time of the test
        'date_time_iso': 'datetime64',       # ISO datetime
        'cycle_index': 'int32',              # Index of the cycle
        'cycle_label': 'category',           # Label of the cycle - default="regular"
        'current': 'float32',                # Current
        'voltage': 'float32',                # Voltage
        'temperature': 'float32',            # Temperature of the cell
        'internal_resistance': 'float32',    # Internal resistance of the cell
        'charge_capacity': 'float32',        # Charge capacity of the cell
        'discharge_capacity': 'float32',     # Discharge capacity of the cell
        'charge_energy': 'float32',          # Charge energy of the cell
        'discharge_energy': 'float32',       # Discharge energy of the cell
        'step_index': 'int16',               # Index of the step (i.e., type), according to the cycler output
        'step_counter': 'int32',             # Counter of the step within cycle, according to the cycler
        'step_counter_absolute': 'int32',    # BEEP-determined step counter across all cycles
        'step_label': 'category',            # Label of the step - default is automatically determined
        'datum': 'int32',                    # Data point, an index.
    },
    # If retain is None, all columns are kept, including nonstandard
    # columns. If retain is a list, only columns in the list are kept.
    "retain": None 
}

# Cycle level config ONLY
# Config mode 1: Constant n point per step within a cycle. 
# k steps within a cycle will result in n*k points per cycle.
# Config for mode 2: Constant n points per step label within a cycle, 
# regardless of k steps in cycle.
# for $i \in S$ step labels, $n_i$ points per step label, will result in $\sum_i n_i$ points per cycle.
# Note: for temporally disparate steps with the same steps label, strange behavior can occur. 
CYCLE_CONFIG_DEFAULT = {
    "preaggregate_steps_by_step_label": False,
}

# Step level config ONLY
# For a "columns" value of None, ALL columns will be interpolated except
# for those known to be constant (e.g., cycle label)
STEP_CONFIG_DEFAULT = {
    "field_name": "voltage",
    "field_range": None,
    "columns": None,
    "resolution": 1000,
    "exclude": False,
    "min_points": 2
}

def interpolate_cycle(
        cycle: Cycle,
        cconfig: dict = None
    ) -> Union[Cycle, None]:
    """
    Interpolate a Cycle object and return it's interpolated version.

    Exactly how the cycle (and it's steps) are interpolated depends on
    the config of the cycle and the configs of the steps.

    Args:
        cycle (Cycle): Cycle object to interpolate
        cconfig (dict): Cycle container level config, including which columns
            to keep. If None, default config is used.
    
    Returns:
        Union[Cycle, None]: Interpolated Cycle object, or None if interpolation failed.
    """
    container_config = copy.deepcopy(CONTAINER_CONFIG_DEFAULT)
    container_config.update(cconfig)
    dtypes = container_config["dtypes"]
    retain = container_config["retain"]

    config = copy.deepcopy(CYCLE_CONFIG_DEFAULT)
    config.update(cycle.config)

    preaggregate = config["preaggregate_steps_by_step_label"]
    if preaggregate:
        # Create new "Steps" based on multiple steps grouped by their step labels
        steps = []
        for step_label, df in cycle.data.groupby("step_label"):

            if not df.empty:
                new_step = MultiStep(df)

                # NOTE: we assume that if pre-aggregating, we can use the first matching
                # step (acc. to label) to obtain the config for the multistep.
                first_matching_step = [
                    step for step in cycle.steps if step.step_label == step_label
                ][0]
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

        if retain:
            dataframe = dataframe[retain]

        # todo: need to implement some sort of field range checking, as it
        # is really sneaky when some fields have no range (because it only looks at first and last, not min-max)
        # i.e., something like "Exclude bad interpolation" and then checks length of interpolated df

        sconfig = copy.deepcopy(STEP_CONFIG_DEFAULT)
        sconfig.update(step.config)

        resolution = sconfig["resolution"]
        field_name = sconfig["field_name"]
        field_range = sconfig["field_range"]
        min_points = sconfig["min_points"]
        columns = sconfig["columns"]
        exclude = sconfig["exclude"]

        if dataframe.shape[0] <= min_points or exclude:
            continue

        # TODO: Fix this unneeded dropping when actually merging in
        date_droppables = [f for f in ("date_time", "date_time_iso") if f != field_name]
        droppables = [c for c in dataframe.columns if c.startswith("_")] + date_droppables
        droppables += ["datum"]
        dataframe = dataframe.drop(columns=[d for d in droppables if d in dataframe.columns])

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

        column_types = {k: v for k, v in dtypes.items() if k in interpolated_df.columns}
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
    
    # Create a new cycle instance, incl. config
    # Ignore cycles where no steps were interpolated.
    if not interpolated_steps:
        return None
    else:
        cycle_interpolated = Cycle(interpolated_steps)
        cycle_interpolated.config = copy.deepcopy(cycle.config)
        return cycle_interpolated