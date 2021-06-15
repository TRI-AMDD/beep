"""Classes and functions for handling Arbin battery cycler data.

"""
import os
from datetime import datetime

import pytz
import pandas as pd

from beep.conversion_schemas import ARBIN_CONFIG
from beep import logger
from beep.structure.base import BEEPDatapath


class ArbinDatapath(BEEPDatapath):
    """A datapath for Arbin cycler data.

    Arbin cycler data contains two files:

    - Raw data: A raw CSV
    - Metadata: Typically the filename of the raw data + "_Metadata" at the end.

    The metadata file is optional but strongly recommended.

    Attributes:
        All from BEEPDatapath
    """


    def validate(self):
        """
        Validator for large, cyclic dataframes coming from Arbin.
        Requires a valid Cycle_Index column of type int.
        Designed for performance - will stop at the first encounter of issues.

        Args:
            df (pandas.DataFrame): Arbin output as DataFrame.
            schema (str): Path to the validation schema. Defaults to arbin for now.
        Returns:
            bool: True if validated with out errors. If validation fails, errors
                are listed at ValidatorBeep.errors.
        """

        try:
            schema = loadfn(schema)
            self.arbin_schema = schema
        except Exception as e:
            warnings.warn("Arbin schema could not be found: {}".format(e))

        df = df.rename(str.lower, axis="columns")

        # Validation cycle index data and cast to int
        if not self._prevalidate_nonnull_column(df, "cycle_index"):
            return False
        df.cycle_index = df.cycle_index.astype(int, copy=False)

        # Validation starts here
        self.schema = self.arbin_schema

        for cycle_index, cycle_df in tqdm(df.groupby("cycle_index")):
            cycle_dict = cycle_df.replace({np.nan, "None"}).to_dict(orient="list")
            result = self.validate(cycle_dict)
            if not result:
                return False
        return True

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

        for column, dtype in ARBIN_CONFIG["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(ARBIN_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata_path = metadata_path if metadata_path else path.replace(".csv",
                                                                         "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(ARBIN_CONFIG["metadata_fields"], axis="columns",
                            inplace=True)
            # Note the to_dict, which scrubs numpy typing
            metadata = {col: item[0] for col, item in
                        metadata.to_dict("list").items()}
        else:
            logger.warning(f"No associated metadata file for Arbin: "
                           f"'{metadata_path}'. No metadata loaded.")
            metadata = {}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(
            lambda x: datetime.utcfromtimestamp(x).replace(
                tzinfo=pytz.UTC).isoformat()
        )

        paths = {
            "raw": path,
            "metadata": metadata_path if metadata else None
        }

        return cls(data, metadata, paths)
