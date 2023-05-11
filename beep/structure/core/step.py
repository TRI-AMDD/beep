"""Classes for representing individual cycler steps (fundamental parts of a cycle).
"""
import pandas as pd


class Step:
    """
    A persistent step object, basically a wrapper around a step's dataframe.
    Requires several columns to only have one unique value.
    This is where all ground truth data is kept.

    Args:
        df_step (pd.DataFrame): A dataframe containing a single step's data.

    Attributes:
        data (pd.DataFrame): All data for this step, organized in presentable format.
        config (dict): A dictionary of Step-level interpolation parameters.
            This config may be overriden by cycle-level configuration.
        uniques (tuple): A tuple of columns that must be unique for a step to be instantiated.
    """
    # Step level config ONLY
    # For a "columns" value of None, ALL columns will be interpolated except
    # for those known to be constant (e.g., cycle label)
    CONFIG_DEFAULT = {
        "field_name": "voltage",
        "field_range": None,
        "columns": None,
        "resolution": 1000,
        "exclude": False,
        "min_points": 2
    }

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

    Args:
        df_multistep (pd.DataFrame): A dataframe containing one or multiple step's data
            which you would like to 'represent' as a single step for interpolation
            purposes. 

    Attributes:
        data (pd.DataFrame): All data for this series of steps, organized in presentable format.
        mandatory_uniques (tuple): A tuple of columns that must be unique for a 
            multistep to be instantiated.

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