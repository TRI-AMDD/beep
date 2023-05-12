from typing import Iterable

import dask.bag as bag

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

        if isinstance(self.items, bag.Bag):
            self.items_length = self.items.count().compute()
        else:
            self.items_length = len(self.items)

    def __getitem__(self, indexer):
        if isinstance(indexer, int):
            item_selection = [i for i in self.items if getattr(i, self.index_field) == indexer]
        elif isinstance(indexer, tuple):
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
            raise IndexError(f"No items found for indexer: {indexer}")
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

    @property
    def data(self):
        return aggregate_nicely([obj.data for obj in self.items])
    

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
    ) -> dict:
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

