"""Classes and functions for handling Arbin battery cycler data.

"""
import os
from datetime import datetime

import pytz
from monty.serialization import loadfn
import pandas as pd

from beep.conversion_schemas import ARBIN_CONFIG
from beep import logger, VALIDATION_SCHEMA_DIR
from beep.structure.base import BEEPDatapath
from beep.structure.validate import PROJECT_SCHEMA


class ArbinDatapath(BEEPDatapath):
    """A datapath for Arbin cycler data.

    Arbin cycler data contains two files:

    - Raw data: A raw CSV
    - Metadata: Typically the filename of the raw data + "_Metadata" at the end.

    The metadata file is optional but strongly recommended.

    Attributes:
        All from BEEPDatapath
    """

    conversion_config = ARBIN_CONFIG

    @classmethod
    def from_file(cls, path, metadata_path=None):
        """Load an Arbin file to a datapath.

        Args:
            path (str, Pathlike): Path to the raw data csv.
            metadata_path (str, None): Path to metadata file, if it
                cannot be inferred from the path of the raw file.

        Returns:
            (ArbinDatapath)
        """
        data = pd.read_csv(path, index_col=0)
        data.rename(str.lower, axis="columns", inplace=True)

        for column, dtype in cls.conversion_config["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(cls.conversion_config["data_columns"], axis="columns", inplace=True)

        metadata_path = metadata_path if metadata_path else path.replace(".csv",
                                                                         "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(cls.conversion_config["metadata_fields"], axis="columns",
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

        # Set schema from filename, if possible; otherwise, use default arbin schema
        project_schema = loadfn(PROJECT_SCHEMA)
        name = os.path.basename(path)
        special_schema_filename = project_schema.get(name.split("_")[0], {}).get("arbin")

        if special_schema_filename:
            schema = os.path.join(VALIDATION_SCHEMA_DIR, special_schema_filename)
        else:
            schema = os.path.join(VALIDATION_SCHEMA_DIR, "schema-arbin-lfp.yaml")

        return cls(data, metadata, paths=paths, schema=schema)
