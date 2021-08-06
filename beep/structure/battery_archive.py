"""Classes and functions for handling Battery Archive cycler data.

"""
import os
from datetime import datetime

import pytz
import pandas as pd
import numpy as np

from beep.structure.base import BEEPDatapath


class BatteryArchiveDatapath(BEEPDatapath):
    """A datapath for Battery Archive cycler data.


    One file is required - a *timeseries.csv data file
    from the Battery Archive. No metadata is supported.

    Attributes:
        All from BEEPDatapath
    """

    # Mapping of raw data file columns to BEEP columns
    COLUMN_MAPPING = {
        "test_time (s)": "test_time",
        "cycle_index": "cycle_index",
        "current (a)": "current",
        "voltage (v)": "voltage",
        "charge_capacity (ah)": "charge_capacity",
        "discharge_capacity (ah)": "discharge_capacity",
        "charge_energy (wh)": "charge_energy",
        "discharge_energy (wh)": "discharge_energy",
        "cell_temperature (c)": "temperature",
        "date_time": "date_time"
    }

    # Columns to ignore
    COLUMNS_IGNORE = ["environment_temperature (c)"]

    # Mapping of data types for BEEP columns
    DATA_TYPES = {
        "test_time": "float64",
        "cycle_index": "int32",
        "current": "float32",
        "voltage": "float32",
        "charge_capacity": "float64",
        "discharge_capacity": "float64",
        "charge_energy": "float64",
        "discharge_energy": "float64",
        "temperature": "float32",
        "date_time": "float32",
    }

    FILE_PATTERN = ".*timeseries\\.csv"

    @classmethod
    def from_file(cls, path):
        """Load a Battery Archive cycler file from raw file.

        Step indices and times are not given, so they are reverse engineered
        based on current. Three main steps are assigned for each cycle:

        1. Rest
        2. Charge
        3. Discharge

        Args:
            path (str, Pathlike): Path to the raw data csv.

        Returns:
            (ArbinDatapath)
        """
        df = pd.read_csv(path)
        df.rename(str.lower, axis="columns", inplace=True)
        df.drop(columns=[c for c in cls.COLUMNS_IGNORE if c in df.columns], inplace=True)
        df["step_index"] = 0

        df["step_index"] = df["current (a)"].apply(decide_step_index)

        step_change_ix = np.where(np.diff(df["step_index"], prepend=np.nan))[0]

        # get list of start times to subtract
        start_times = df["test_time (s)"].loc[step_change_ix]
        arrays = [None] * len(step_change_ix)

        final_ix = [df.shape[0] - 1]
        step_change_ix_buffered = np.append(step_change_ix[1:], final_ix)

        for i, scix in enumerate(step_change_ix_buffered):
            prev_scix = step_change_ix[i]
            n_repeats = scix - prev_scix
            arrays[i] = np.repeat(start_times.loc[prev_scix], n_repeats)

        # include the final data point, which is not a real step change
        subtractor = np.concatenate(arrays, axis=0)
        subtractor = np.append(subtractor, [subtractor[-1]])

        df["step_time"] = df["test_time (s)"] - subtractor

        df.rename(columns=cls.COLUMN_MAPPING, inplace=True)
        dtfmt = '%Y-%m-%d %H:%M:%S.%f'
        # convert date time string to
        dts = df["date_time"].apply(lambda x: datetime.strptime(x, dtfmt))

        df["date_time"] = dts.apply(lambda x: x.timestamp())
        df["date_time_iso"] = dts.apply(lambda x: x.replace(tzinfo=pytz.UTC).isoformat())

        for column, dtype in cls.DATA_TYPES.items():
            if column in df:
                if not df[column].isnull().values.any():
                    df[column] = df[column].astype(dtype)

        paths = {
            "raw": os.path.abspath(path),
            "metadata": None
        }

        # there is no metadata given in the BA files
        metadata = {}

        return cls(df, metadata, paths)


def decide_step_index(i):
    """
    Decide a step index based on current values.
    Args:
        i (float): Current value

    Returns:
        (int): Step index
    """
    if np.abs(i) < 1e-6:
        return 1
    elif i < 0:
        return 3
    else:
        return 2
