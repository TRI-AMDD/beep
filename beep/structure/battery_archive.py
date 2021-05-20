"""Classes and functions for handling Arbin battery cycler data.

"""
import os
from datetime import datetime

import pytz
import pandas as pd
import numpy as np
import tqdm
import time

from beep import logger
from beep.structure.base import BEEPDatapath
from multiprocessing import Pool, cpu_count



class BatteryArchiveDatapath(BEEPDatapath):
    """A datapath for Arbin cycler data.

    Arbin cycler data contains two files:

    - Raw data: A raw CSV
    - Metadata: Typically the filename of the raw data + "_Metadata" at the end.

    The metadata file is optional but strongly recommended.

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
        "test_time": "float32",
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

        for i, scix in tqdm.tqdm(enumerate(step_change_ix_buffered)):
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
            "raw": path,
            "metadata": None

        }

        # there is no metadata given in the BA files
        metadata = {}

        return cls(df, metadata, paths)


def decide_step_index(i):
    if np.abs(i) < 1e-6:
        return 1
    elif i < 0:
        return 3
    else:
        return 2


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # ba_cycle_file = "/Users/ardunn/alex/tri/code/beep/alex_scripts/extra_df_files_chirru/SNL_18650_LFP_15C_0-100_0.5-1C_a_cycle_data.csv"
    ba_ts_file = "/Users/ardunn/alex/tri/code/beep/alex_scripts/extra_data_files_chirru/SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv"
    import matplotlib.pyplot as plt

    t0 = time.time()
    bad = BatteryArchiveDatapath.from_file(ba_ts_file)
    t1 = time.time()

    print(f"time taken: {t1 - t0}")

    print(bad.raw_data)


    # relevant_columns = ["test_time (s)", "current (a)", "cycle_index", "step_index", "step_time"]
    # print(df[relevant_columns].loc[:20])
    #
    # print(df[relevant_columns].tail(20))
    #
    # print(df[relevant_columns].loc[660:680])
    #
    # raise ValueError
    # df = df.loc[:]
    #
    # # plt.plot(df["test_time (s)"], df["current (a)"])
    # plt.plot(df["test_time (s)"], df["step_index"])
    #
    # plt.show()
    # print(df)