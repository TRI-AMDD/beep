import time
from typing import Iterable

import dask.bag as bag
import dask.dataframe as ddf
from dask import delayed

import numpy as np
import pandas as pd

# Common args for TQDM that will determine appearance.
TQDM_STYLE_ARGS = {
    "ascii": ' #'
}


class DFSelectorAggregator:
    """
    Class for aggregating the following from lists of Cycles/Steps:
        - individual indices
        - tuples (or other iterables) of indices
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

        if isinstance(self.items, bag.Bag):
            self.items_length = self.items.count().compute()
        else:
            self.items_length = len(self.items)

    def __getitem__(self, indexer):
        if isinstance(indexer, int):
            item_selection = [i for i in self.items if getattr(i, self.index_field) == indexer]
        elif isinstance(indexer, (tuple, set, list, frozenset)):
            item_selection = [i for i in self.items if getattr(i, self.tuple_field) in indexer]
        elif isinstance(indexer, slice):
            indexer = tuple(range(*indexer.indices(self.items_length)))
            item_selection = [i for i in self.items if getattr(i, self.slice_field) in indexer]
        elif isinstance(indexer, str):
            item_selection = [i for i in self.items if getattr(i, self.label_field) == indexer]
        else:
            raise TypeError(
                f"No indexing scheme for {self.__class__.__name__} available for type {type(indexer)}")

        if len(item_selection) == 1:
            return item_selection[0]
        elif len(item_selection) == 0:
            raise DFSelectorIndexError(f"No items found for indexer: {indexer}")
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

    def __getattr__(self, attr):
        if attr == "steps" and all([hasattr(item, "steps") for item in self]):
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
        elif self.__getattribute__("items_length") == 1:
            # allowing for dask bag to be the items requires goofy comprehension
            it = [i for i in self.__getattribute__("items")]
            return getattr(it[0], attr)
        elif self.__getattribute__("items_length") == 0:
            return getattr([], attr)
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
    
    def __len__(self):
        # Necessary since dask bag has no length
        return self.items_length

    def by_raw_index(self, ix):
        """
        Get a single item by its raw index in the list of items.
        Indices can be negative as well.

        """
        # todo: this ordering is not guaranteed by dask.bag
        # todo: and therefore it is subject to breakage in the future
        if ix < 0:
            ix = self.items_length + ix
        for i, item in enumerate(self.items):
            if i == ix:
                return item
        else:
            raise DFSelectorIndexError(f"No item found at index {ix}")

    @property
    def data(self):
        """
        Get a pandas dataframe in memory from items in the
        collection.

        Returns:
            pd.DataFrame
        """
        return aggregate_nicely([obj.data for obj in self.items])

    @property
    def data_lazy(self):
        """
        Get a dask (lazy) dataframe object from items in the
        collection. You must compute the dask object at some point
        in order to use it.

        Returns:
            dask.dataframe.DataFrame
        """
        # todo: this .data call can maybe be .data_lazy itself
        return ddf.from_delayed(
            [delayed(lambda s: s.data)(step) for step in self.items]
        )
    

def aggregate_nicely(iterable: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate a bunch of dataframes (i.e., from steps) into a single dataframe 
    that looks presentable.

    Args:
        iterable (Iterable[pd.DataFrame]): An iterable of dataframes to aggregate
    """
    if iterable:
        return pd.concat(iterable).sort_values(by="test_time").reset_index(drop=True)
    else:
        return pd.DataFrame()
    

def label_chg_state(
        step_df: pd.DataFrame, 
        indeterminate_label: str = "unknown"
    ) -> str:
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


def get_max_paused_over_threshold(
        group: pd.DataFrame,
        paused_threshold: int = 3600
) -> float:
    """
    Evaluate a raw cycling dataframe to determine if there is a pause in cycling.
    The method looks at the time difference between each row and if this value
    exceeds a threshold, it returns that length of time in seconds. Otherwise, it
    returns 0

    Args:
        group (pd.DataFrame): cycling dataframe with date_time_iso column
        paused_threshold (int): gap in seconds to classify as a pause in cycling

    Returns:
        (float): number of seconds that test was paused

    """
    date_time_objs = pd.to_datetime(group["date_time_iso"])
    date_time_float = [
        time.mktime(t.timetuple()) if t is not pd.NaT else float("nan")
        for t in date_time_objs
    ]
    date_time_float = pd.Series(date_time_float, dtype=float)
    if date_time_float.diff().max() > paused_threshold:
        max_paused_duration = date_time_float.diff().max()
    else:
        max_paused_duration = 0
    return max_paused_duration


def get_cv_stats(charge, **kwargs):
    """
    Extract all constant voltage statistics from a dataframe
    of a charge step (or steps).

    Args:
        charge (pd.DataFrame): Dataframe of charge step.
        **kwargs: Keyword arguments to pass to get_cv_segment_from_charg

    Returns:

    """
    cv = get_cv_segment_from_charge(charge, **kwargs)
    if cv.empty:
        raise CVStatsError("No CV segment found in charge dataframe.")
    else:
        cv_stats = pd.Series(
            data=[
                charge.cycle_index.unique()[0],
                cv.test_time.iat[-1] - cv.test_time.iat[0],
                cv.current.iat[-1],
                cv.charge_capacity.iat[-1] - cv.charge_capacity.iat[0]
            ],
            index=[
                "cycle_index",
                "cv_time",
                "cv_current",
                "cv_capacity"
            ]
        )
        return cv_stats


def get_cv_segment_from_charge(charge, dt_tol=1, dvdt_tol=1e-5, didt_tol=1e-4):
    """
    Extracts the constant voltage segment from charge. Works for both cccv or
    CC steps followed by a cv step.

    Args:
        charge (pd.DataFrame): charge dataframe for a single cycle
        dt_tol (float) : dt tolerance (minimum) for identifying cv
        dvdt_tol (float) : dvdt tolerance (maximum) for identifying cv
        didt_tol (float) : dvdt tolerance (minimum) for identifying cv 

    Returns:
        (pd.DataFrame): dataframe containing the cv segment

    """
    if charge.empty:
        return charge
    else:
        # Compute di and dv
        di = np.diff(charge.current)
        dv = np.diff(charge.voltage)
        dt = np.diff(charge.test_time)

        # Find the first index where the cv segment begins
        i = 0
        while i < len(dv) and (dt[i] < dt_tol or abs(dv[i]/dt[i]) > dvdt_tol or abs(di[i]/dt[i]) < didt_tol):
            i = i+1

        # Filter for cv phase
        return charge.loc[charge.test_time >= charge.test_time.iat[i-1]]


class DFSelectorIndexError(BaseException):
    """
    Raised when DFSelectorAggregator is unable to find a suitable dataframe
    """
    pass


class CVStatsError(BaseException):
    """Raised when problems are found in CV segments."""
    pass
