import os
from datetime import datetime

import pandas as pd
import numpy as np
import pytz

from beep import logger
from beep.structure.core.run import Run


class MaccorRun(Run):
    """Datapath for ingesting and structuring Maccor battery cycler data.

    Attributes:
        - all attributes inherited from BEEPDatapath
        - eis [MaccorEIS]: List of MaccorEIS objects, each representing an electrochemical
            impedance spectrum complete with data and metadata.
    """

    CONVERSION_CONFIG = {
        'file_pattern': ".*\\d{5,6}.*\\d{3}",
        'charge_state_code': 'C',
        'discharge_state_code': 'D',
        'end_step_code_min': 128,
        'end_step_code_max': 255,
        'include_eis': False,
        'metadata_fields': {
            "today's date": '_today_datetime',
            'date of test': 'start_datetime',
            'filename': 'filename',
            'procedure': 'protocol',
            'comment/barcode': 'barcode'
        },
        'data_columns': {
            'rec#': 'data_point',
            'cyc#': 'cycle_index',
            'step': 'step_code',
            'test (sec)': 'test_time',
            'step (sec)': 'step_time',
            'amp-hr': '_capacity',
            'watt-hr': '_energy',
            'amps': 'current',
            'volts': 'voltage',
            'state': '_state',
            'es': '_ending_status',
            'dpt time': 'date_time',
            'acimp/ohms': 'ac_impedence',
            'dcir/ohms': 'internal_resistance',
            'wf chg cap': '_wf_chg_cap',
            'wf dis cap': '_wf_dis_cap',
            'wf chg e': '_wf_chg_e',
            'wf dis e': '_wf_dis_e',
            'range': '_range',
            'var1': '_var1',
            'var2': '_var2',
            'var3': '_var3',
            'var4': '_var4',
            'var5': '_var5',
            'var6': '_var6',
            'var7': '_var7',
            'var8': '_var8',
            'var9': '_var9',
            'var10': '_var10',
            'var11': '_var11',
            'var12': '_var12',
            'var13': '_var13',
            'var14': '_var14',
            'var15': '_var15'
        },
        'data_types': {
            'rec#': 'int32',
            'cyc#': 'int32',
            'step': 'int16',
            'test (sec)': 'float64',
            'step (sec)': 'float32',
            'amp-hr': 'float64',
            'watt-hr': 'float64',
            'amps': 'float32',
            'volts': 'float32',
            'state': 'category',
            'es': 'category',
            'acimp/ohms': 'float32',
            'dcir/ohms': 'float32',
            'wf chg cap': 'float32',
            'wf dis cap': 'float32',
            'wf chg e': 'float32',
            'wf dis e': 'float32',
            'range': 'uint8',
            'var1': 'float16',
            'var2': 'float16',
            'var3': 'float16',
            'var4': 'float16',
            'var5': 'float16',
            'var6': 'float16',
            'var7': 'float16',
            'var8': 'float16',
            'var9': 'float16',
            'var10': 'float16',
            'var11': 'float16',
            'var12': 'float16',
            'var13': 'float16',
            'var14': 'float16',
            'var15': 'float16',
            'dpt time': 'str'
        }
    }

    # based on maccor_2170
    VALIDATION_SCHEMA = {
        'data_point': {
            'schema': {
                'type': 'integer'
            },
            'type': 'list'
        },
        'cycle_index': {
            'schema': {
                'min': 0,
                'min_is_below': 2,
                'max_at_least': 1,
                'type': 'integer',
                'monotonic': 'increasing'
            },
            'type': 'list'
        },
        '_state': {
            'schema': {
                'type': 'string'
            },
            'type': 'list'
        },
        'voltage': {
            'schema': {
                'max': 5.0,
                'min': 0,
                'type': 'float'
            },
            'type': 'list'
        },
        'current': {
            'schema': {
                'max': 15.0,
                'min': -15.0,
                'min_is_below': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'test_time': {
            'schema': {
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'step_time': {
            'schema': {
                'min': 0.0,
                'type': 'float'
            },
            'type': 'list'
        },
        'date_time': {
            'schema': {
                'type': 'string'
            },
            'type': 'list'
        }
    }

    @classmethod
    def from_file(cls, path, **kwargs):
        """Create a MaccorDatapath file from a Maccor cycler run raw file.

        Args:
            path (str, Pathlike): file path for maccor file.
            kwargs: Keyword args to pass to Run init.

        Returns:
            (MaccorRun)
        """
        with open(path) as f:
            metadata_line = f.readline().strip()

        # Parse data
        data = pd.read_csv(path, delimiter="\t", skiprows=1)
        data.rename(str.lower, axis="columns", inplace=True)
        data = data.astype(cls.CONVERSION_CONFIG["data_types"])
        data.rename(cls.CONVERSION_CONFIG["data_columns"], axis="columns", inplace=True)

        # Needed for validating correctly
        data["_state"] = data["_state"].astype(str)

        data["charge_capacity"] = cls.quantity_sum(
            data, "capacity", "charge"
        )
        data["discharge_capacity"] = cls.quantity_sum(
            data, "capacity", "discharge"
        )
        data["charge_energy"] = cls.quantity_sum(data, "energy", "charge")
        data["discharge_energy"] = cls.quantity_sum(
            data, "energy", "discharge"
        )

        # Parse metadata - kinda hackish way to do it, but it works
        metadata = cls.parse_metadata(metadata_line)
        metadata = pd.DataFrame(metadata)

        try:
            _, channel_number = os.path.splitext(path)
            metadata["channel_id"] = int(channel_number.replace(".", ""))
        except ValueError:
            logger.warning("Could not infer channel number from path name!")
            metadata["channnel_id"] = None

        metadata.rename(str.lower, axis="columns", inplace=True)
        metadata.rename(cls.CONVERSION_CONFIG["metadata_fields"], axis="columns", inplace=True)
        # Note the to_dict, which scrubs numpy typing
        metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(cls.correct_timestamp)

        paths = {
            "raw": path,
            "metadata": path
        }

        # If a custom schema is not passed, use the default
        if "schema" not in kwargs:
            kwargs["schema"] = cls.VALIDATION_SCHEMA

        return cls.from_dataframe(
            data, 
            metadata=metadata, 
            paths=paths, 
            **kwargs
        )

    @classmethod
    def quantity_sum(cls, data, quantity, state_type):
        """Computes non-decreasing capacity or energy (either charge or discharge)
        through multiple steps of a single cycle and resets capacity at the
        start of each new cycle. Input Maccor data resets to zero at each step.

        Args:
            data (pd.DataFrame): maccor data.
            quantity (str): capacity or energy.
            state_type (str): charge or discharge.

        Returns:
            Series: summed quantities.

        """
        state_code = cls.CONVERSION_CONFIG["{}_state_code".format(state_type)]
        quantity_agg = data['_' + quantity].where(data["_state"] == state_code, other=0, axis=0)

        # If a waveform step is present, maccor initializes waveform-specific quantities
        # that are to be used in place of '_capacity' and '_energy'

        if data['_wf_chg_cap'].notna().sum():
            if (state_type, quantity) == ('discharge', 'capacity'):
                quantity_agg = data['_wf_dis_cap'].where(data['_wf_dis_cap'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('charge', 'capacity'):
                quantity_agg = data['_wf_chg_cap'].where(data['_wf_chg_cap'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('discharge', 'energy'):
                quantity_agg = data['_wf_dis_e'].where(data['_wf_dis_e'].notna(), other=quantity_agg, axis=0)
            elif (state_type, quantity) == ('charge', 'energy'):
                quantity_agg = data['_wf_chg_e'].where(data['_wf_chg_e'].notna(), other=quantity_agg, axis=0)
            else:
                pass

        end_step = data["_ending_status"].apply(
            lambda x: cls.CONVERSION_CONFIG["end_step_code_min"] <= x <= cls.CONVERSION_CONFIG["end_step_code_max"]
        )
        # For waveform discharges, maccor seems to trigger ending_status within a step multiple times
        # As a fix, compute the actual step change using diff() on step_code and set end_step to be
        # a logical AND(step_change, end_step)
        is_step_change = data['step_code'].diff(periods=-1).fillna(value=0) != 0
        end_step_inds = end_step.index[np.logical_and(list(end_step), list(is_step_change))]
        # If no end steps, quantity not reset, return it without modifying
        if end_step_inds.size == 0:
            return quantity_agg

        # Initialize accumulator and beginning step slice index
        cycle_sum = 0.0
        begin_step_ind = quantity_agg.index[0] + 1
        for end_step_ind in end_step_inds:
            # Detect whether cycle changed and reset accumulator if so
            if (
                data.loc[begin_step_ind - 1, "cycle_index"]
                != data.loc[begin_step_ind, "cycle_index"]
            ):
                cycle_sum = 0.0

            # Add accumulator to current reset step
            quantity_agg[begin_step_ind:end_step_ind + 1] += cycle_sum

            # Update accumulator
            cycle_sum = quantity_agg[end_step_ind]

            # Set new step slice initial index
            begin_step_ind = end_step_ind + 1

        # Update any dangling step without an end
        last_index = quantity_agg.index[-1]
        if end_step_inds[-1] < last_index:
            quantity_agg[begin_step_ind:] += cycle_sum
        return quantity_agg

    @staticmethod
    def parse_metadata(metadata_string):
        """Parses maccor metadata string, which is annoyingly inconsistent.
        Basically just splits the string by a set of fields and creates
        a dictionary of pairs of fields and values with colons scrubbed
        from fields.

        Args:
            metadata_string (str): string corresponding to maccor metadata.

        Returns:
            dict: dictionary of metadata fields and values.

        """
        metadata_fields = [
            "Today's Date",
            "Date of Test:",
            "Filename:",
            "Procedure:",
            "Comment/Barcode:",
        ]
        metadata_values = MaccorRun.split_string_by_fields(metadata_string,
                                                                metadata_fields)
        metadata = {
            k.replace(":", ""): [v.strip()]
            for k, v in zip(metadata_fields, metadata_values)
        }
        return metadata

    @staticmethod
    def correct_timestamp(x):
        """Helper function with exception handling for cases where the
        maccor cycler mis-prints the datetime stamp for the row. This
        happens when data is being recorded rapidly as the date switches over
        ie. between 10/21/2019 23:59:59 and 10/22/2019 00:00:00.

        Args:
            x (str): The datetime string for maccor in format '%m/%d/%Y %H:%M:%S'

        Returns:
            datetime.Datetime: Datetime object in iso format (daylight savings aware)

        """
        pacific = pytz.timezone("US/Pacific")
        utc = pytz.timezone("UTC")
        try:
            iso = (
                pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"),
                                 is_dst=True)
                .astimezone(utc)
                .isoformat()
            )
        except ValueError:
            x = x + " 00:00:00"
            iso = (
                pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"),
                                 is_dst=True)
                .astimezone(utc)
                .isoformat()
            )
        return iso

    @staticmethod
    def split_string_by_fields(string, fields):
        """Helper function to split a string by a set of ordered strings,
        primarily used for Maccor metadata parsing.

        >>> MaccorRun.split_string_by_fields("first name: Joey  last name Montoya",
        >>>                       ["first name:", "last name"])
        ["Joey", "Montoya"]

        Args:
            string (str): string input to be split
            fields (list): list of fields to split input string by.

        Returns:
            list: substrings corresponding to the split input strings.

        """
        # A bit brittle, there's probably something more clever with recursion
        substrings = []
        init, leftovers = string.split(fields[0])
        for field in fields[1:]:
            init, leftovers = leftovers.split(field)
            substrings.append(init)
        substrings.append(leftovers)
        return substrings
