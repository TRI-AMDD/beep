import abc
import json
import re
from datetime import datetime

import pandas as pd
import numpy as np
import os
import pytz
import time
from scipy import integrate
import itertools
import hashlib
from dataclasses import dataclass

from monty.json import MSONable
from docopt import docopt
from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir
from glob import glob
from beep import tqdm

from beep import StringIO, MODULE_DIR
from beep.validate import ValidatorBeep, BeepValidationError
from beep.collate import add_suffix_to_filename
from beep.conversion_schemas import (
    ARBIN_CONFIG,
    MACCOR_CONFIG,
    FastCharge_CONFIG,
    xTesladiag_CONFIG,
    INDIGO_CONFIG,
    NEWARE_CONFIG,
    BIOLOGIC_CONFIG,
    STRUCTURE_DTYPES,
)

from beep.utils import WorkflowOutputs, parameters_lookup
from beep import logger, __version__

@dataclass
class BeepData:
    raw: pd.DataFrame
    structured: pd.DataFrame



class BeepDatapath:

    def __init__(self, raw_data, metadata, paths):
        self.raw_data = raw_data
        self.metadata = metadata
        self.paths = paths

    @classmethod
    @abc.abstractmethod
    def from_file(cls, path):
        raise NotImplementedError

    def validate(self):
        pass

    def structure(self):
        pass




class ArbinDatapath(BeepDatapath):


    # todo: include metadata file path as optional arg
    @classmethod
    def from_file(cls, path, metadata_path=None):
        """
        Creates RawCyclerRun from an Arbin data file.

        Args:
            path (str): file path to data file
            validate (bool): True if data is to be validated.

        Returns:
            beep.structure.RawCyclerRun
        """
        data = pd.read_csv(path)
        data.rename(str.lower, axis="columns", inplace=True)

        for column, dtype in ARBIN_CONFIG["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(ARBIN_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata_path = metadata_path if metadata_path else path.replace(".csv", "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(ARBIN_CONFIG["metadata_fields"], axis="columns", inplace=True)
            # Note the to_dict, which scrubs numpy typing
            metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}
        else:
            logger.warning(f"No associated metadata file for Arbin: "
                           f"'{metadata_path}'. No metadata loaded.")
            metadata = {}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(
            lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.UTC).isoformat()
        )

        return cls(data, metadata, {})



if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    test_arbin_path = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv"

    from beep.structure import RawCyclerRun as rcrv1

    rcr = rcrv1.from_arbin_file(test_arbin_path)

    print(rcr.data)

    ad = ArbinDatapath.from_file(test_arbin_path)

    print(ad.raw_data)