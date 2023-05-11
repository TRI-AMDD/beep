"""Classes and functions for handling Arbin battery cycler data.

"""
import os
from datetime import datetime

import pytz
from monty.serialization import loadfn
import pandas as pd

from beep.structure.base import BEEPDatapath
from beep import logger


class ArbinRun(BEEPDatapath):
    """A datapath for Arbin cycler data.

    Arbin cycler data contains two files:

    - Raw data: A raw CSV
    - Metadata: Typically the filename of the raw data + "_Metadata" at the end.

    The metadata file is optional but strongly recommended.

    Attributes:
        All from BEEPDatapath
    """
    CONVERSION_CONFIG = {
        'file_pattern': ".*CH.*\\.csv",
        'metadata_fields': {
            'test_id': 'test_id',
            'device_id': 'device_id',
            'iv_ch_id': 'channel_id',
            'first_start_datetime': 'start_datetime',
            'schedule_file_name': 'protocol',
            'item_id': 'barcode',
            'resumed_times': '_resumed_times',
            'last_end_datetime': '_last_end_datetime',
            'databases': '_databases',
            'grade_id': '_grade_id',
            'has_aux': '_has_aux',
            'has_special': '_has_special',
            'schedule_version': '_schedule_version',
            'log_aux_data_flag': '_log_aux_data_flag',
            'log_special_data_flag': '_log_special_data_flag',
            'rowstate': '_rowstate',
            'canconfig_filename': '_canconfig_filename',
            'm_ncanconfigmd5': '_m_ncanconfigmd5',
            'value': '_value',
            'value2': '_value2'
        },
        'data_columns': {
            'data_point': 'data_point',
            'test_time': 'test_time',
            'datetime': 'date_time',
            'step_time': 'step_time',
            'step_index': 'step_index',
            'cycle_index': 'cycle_index',
            'current': 'current',
            'voltage': 'voltage',
            'charge_capacity': 'charge_capacity',
            'discharge_capacity': 'discharge_capacity',
            'charge_energy': 'charge_energy',
            'discharge_energy': 'discharge_energy',
            'dv/dt': '_dv/dt',
            'internal_resistance': 'internal_resistance',
            'temperature': 'temperature'
        },
        'data_types': {
            'data_point': 'int32',
            'test_time': 'float64',
            'datetime': 'float32',
            'step_time': 'float32',
            'step_index': 'int16',
            'cycle_index': 'int32',
            'current': 'float32',
            'voltage': 'float32',
            'charge_capacity': 'float64',
            'discharge_capacity': 'float64',
            'charge_energy': 'float64',
            'discharge_energy': 'float64',
            'dv/dt': 'float32',
            'internal_resistance': 'float32',
            'temperature': 'float32'
        }
    }

    # Default validation schema is based on operating parameters
    # of lithium iron phosphate cells
    VALIDATION_SCHEMA = {
        'charge_capacity': {
            'schema': {
                'max': 2.0,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'cycle_index': {
            'schema': {
                'min': 0,
                'max_at_least': 1,
                'type': 'integer'
            },
            'type': 'list'
        },
        'discharge_capacity': {
            'schema': {
                'max': 2.0,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'temperature': {
            'schema': {
                'max': 80.0,
                'min': 20.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'test_time': {
            'schema': {
                'type': 'float'
            },
            'type': 'list'
        },
        'voltage': {
            'schema': {
                'max': 3.8,
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        }
    }

    @classmethod
    def from_file(
        cls, 
        path, 
        metadata_path=None, 
        validation_schema=None, 
        extra_metadata=None
    ):
        """Load an Arbin file to a Run object.

        Args:
            path (str, Pathlike): Path to the raw data csv.
            metadata_path (str, None): Path to metadata file, if it
                cannot be inferred from the path of the raw file.
            validation_schema (dict): Validation schema as a dictionary. If none
                is passed, the default will be used.
            extra_metadata (dict): Extra metadata to add to the metadata file.
                For example, if a custom validation dictionary was passed,
                you can include the filename is was determined from, etc. 

        Returns:
            (ArbinDatapath)
        """
        data = pd.read_csv(path, index_col=0)
        data.rename(str.lower, axis="columns", inplace=True)

        for column, dtype in cls.CONVERSION_CONFIG["data_types"].items():
            if column in data:
                if not data[column].isnull().values.any():
                    data[column] = data[column].astype(dtype)

        data.rename(cls.CONVERSION_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata_path = metadata_path if metadata_path else path.replace(".csv",
                                                                         "_Metadata.csv")

        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            metadata.rename(str.lower, axis="columns", inplace=True)
            metadata.rename(cls.CONVERSION_CONFIG["metadata_fields"], axis="columns",
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

        if not validation_schema:
            validation_schema = cls.VALIDATION_SCHEMA

        if extra_metadata:
            metadata.update(extra_metadata)

        return cls(
            data, 
            metadata, 
            paths=paths, 
            schema=validation_schema
        )