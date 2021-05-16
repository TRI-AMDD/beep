"""Classes and functions for handling Arbin battery cycler data.

"""
import os
from datetime import datetime

import pytz
import pandas as pd
import numpy as np

from beep.conversion_schemas import ARBIN_CONFIG
from beep import logger
from beep.structure.base import BEEPDatapath




class BatteryArchiveDatapath(BEEPDatapath):
    """A datapath for Arbin cycler data.

    Arbin cycler data contains two files:

    - Raw data: A raw CSV
    - Metadata: Typically the filename of the raw data + "_Metadata" at the end.

    The metadata file is optional but strongly recommended.

    Attributes:
        All from BEEPDatapath
    """

    @classmethod
    def from_file(cls, path, metadata_path=None):
        """Load an Arbin file to a datapath.

        Args:
            path (str, Pathlike): Path to the raw data csv.

        Returns:
            (ArbinDatapath)
        """
        data = pd.read_csv(path)
        data.rename(str.lower, axis="columns", inplace=True)

        cycles = df["cycle_index"].unique()

        step = 1
        tol = 1e-6
        for c in cycles:
            df_cyc = df[df["cycle_index"] == c]
            df_cyc["step_index"]
            df_has_chg_state = df_cyc[df_cyc["current (a)"].abs() > tol]

            sign = df_has_chg_state["current (a)"].map(np.sign)
            sign_changes = sign.diff(periods=1).fillna(0)

            # ix where step change occurs is one after sign is changed
            step_change_ix = df_has_chg_state[sign_changes == 2] + 1


            is_chg_pre_stepchange = df_has_chg_state.loc[step_change_ix - 10:step_change_ix].mean() > 0.0
            is_chg_post_stepchange = df_has_chg_state.loc[step_change_ix + 1: step_change_ix + 11].mean() > 0.0

            if is_chg_pre_stepchange and not is_chg_post_stepchange:
                df_cyc.loc[step_change_ix:] = step
                df_cyc.loc[0:step_change_ix] = step + 1

            df2 = df.loc[diff2[diff2 != 0].index]
            idx = np.where(abs(df1.value.values) < abs(df2.value.values),
                           df1.index.values, df2.index.values)
            df.loc[idx]






        # print(data[(data["current (a)"] > 0.0) & (data["charge_energy (wh)"] != 0)].shape)

        print(data)

        raise ValueError

        # for column, dtype in ARBIN_CONFIG["data_types"].items():
        #     if column in data:
        #         if not data[column].isnull().values.any():
        #             data[column] = data[column].astype(dtype)
        #
        # data.rename(ARBIN_CONFIG["data_columns"], axis="columns", inplace=True)
        #
        # metadata_path = metadata_path if metadata_path else path.replace(".csv",
        #                                                                  "_Metadata.csv")
        #
        # if os.path.exists(metadata_path):
        #     metadata = pd.read_csv(metadata_path)
        #     metadata.rename(str.lower, axis="columns", inplace=True)
        #     metadata.rename(ARBIN_CONFIG["metadata_fields"], axis="columns",
        #                     inplace=True)
        #     # Note the to_dict, which scrubs numpy typing
        #     metadata = {col: item[0] for col, item in
        #                 metadata.to_dict("list").items()}
        # else:
        #     logger.warning(f"No associated metadata file for Arbin: "
        #                    f"'{metadata_path}'. No metadata loaded.")
        #     metadata = {}
        #
        # # standardizing time format
        # data["date_time_iso"] = data["date_time"].apply(
        #     lambda x: datetime.utcfromtimestamp(x).replace(
        #         tzinfo=pytz.UTC).isoformat()
        # )
        #
        # paths = {
        #     "raw": path,
        #     "metadata": metadata_path if metadata else None
        # }
        #
        # return cls(data, metadata, paths)


if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    ba_cycle_file = "/Users/ardunn/alex/tri/code/beep/alex_scripts/extra_data_files_chirru/SNL_18650_LFP_15C_0-100_0.5-1C_a_cycle_data.csv"
    ba_ts_file = "/Users/ardunn/alex/tri/code/beep/alex_scripts/extra_data_files_chirru/SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv"
    import matplotlib.pyplot as plt



    # df = pd.read_csv(ba_cycle_file)
    #
    # print(df)
    #
    # df = pd.read_csv(ba_ts_file)
    #
    # print(df)



    from beep.structure.arbin import ArbinDatapath


    # ad = ArbinDatapath.from_file("/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/2017-08-14_8C-5per_3_47C_CH44.csv")


    # df = ad.raw_data
    # plt.plot(df["cycle_index"], df["step_index"])
    #
    # print("arbin")
    # print(ad.raw_data)
    # print(ad.raw_data["step_index"].isna().all())
    #
    #
    #
    # print("ba")
    # bad = BatteryArchiveDatapath.from_file(ba_ts_file)




    df = pd.read_csv(ba_ts_file)
    df.rename(str.lower, axis="columns", inplace=True)
    df = df[df["cycle_index"] == 1.0]


    print(df)


    # plt.plot(df["test_time (s)"], df["charge_capacity (ah)"], label="charge")
    # plt.plot(df["test_time (s)"], df["discharge_capacity (ah)"], label="discharge")
    # plt.legend()


    plt.plot(df["test_time (s)"], df["current (a)"])
    plt.plot(df["test_time (s)"], df["voltage (v)"])

    plt.show()

