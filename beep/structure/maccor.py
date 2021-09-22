"""Classes and functions for handling Maccor battery cycler data.
"""
import os
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np
import pytz
from monty.serialization import loadfn

from beep import tqdm, StringIO, VALIDATION_SCHEMA_DIR
from beep.conversion_schemas import MACCOR_CONFIG
from beep.structure.base_eis import BEEPDatapathWithEIS, EIS
from beep.structure.validate import PROJECT_SCHEMA


class MaccorDatapath(BEEPDatapathWithEIS):
    """Datapath for ingesting and structuring Maccor battery cycler data.

    Attributes:
        - all attributes inherited from BEEPDatapath
        - eis [MaccorEIS]: List of MaccorEIS objects, each representing an electrochemical
            impedance spectrum complete with data and metadata.
    """

    class MaccorEIS(EIS):
        """Class representing a single EIS run from Maccor data.
        """

        @classmethod
        def from_file(cls, filename):
            """Create a single MaccorEIS object from a raw EIS output file.

            Args:
                filename (str, Pathlike): file path to data.

            Returns:
                (MaccorEIS): EIS object representation of data.
            """
            with open(filename) as f:
                lines = f.readlines()
            # Parse freq sweep, method, and output filename
            freq_sweep = lines[1].split("Frequency Sweep:")[1].strip()
            freq_sweep = freq_sweep.replace("Circut", "Circuit")
            method = lines[2]
            filename = lines[3]

            # Parse start datetime and reformat in isoformat
            start = lines[6].split("Start Date:")[1].strip()
            date, time = start.split("Start Time:")
            start = ",".join([date.strip(), time.strip()])
            start = datetime.strptime(start, "%A, %B %d, %Y,%H:%M")
            start = start.isoformat()

            line_8 = lines[8].split()

            # Construct metadata dictionary
            metadata = {
                "frequency_sweep": freq_sweep,
                "method": method,
                "filename": filename,
                "start": start,
                "line_8": line_8,
            }

            data = "\n".join(lines[10:])
            data = pd.read_csv(StringIO(data), delimiter="\t")
            return cls(data=data, metadata=metadata)

    @classmethod
    def from_file(cls, path):
        """Create a MaccorDatapath file from a Maccor cycler run raw file.

        Args:
            path (str, Pathlike): file path for maccor file.

        Returns:
            (MaccorDatapath)
        """
        with open(path) as f:
            metadata_line = f.readline().strip()

        # Parse data
        data = pd.read_csv(path, delimiter="\t", skiprows=1)
        data.rename(str.lower, axis="columns", inplace=True)
        data = data.astype(MACCOR_CONFIG["data_types"])
        data.rename(MACCOR_CONFIG["data_columns"], axis="columns", inplace=True)

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
        _, channel_number = os.path.splitext(path)
        metadata["channel_id"] = int(channel_number.replace(".", ""))
        metadata.rename(str.lower, axis="columns", inplace=True)
        metadata.rename(MACCOR_CONFIG["metadata_fields"], axis="columns", inplace=True)
        # Note the to_dict, which scrubs numpy typing
        metadata = {col: item[0] for col, item in metadata.to_dict("list").items()}

        # standardizing time format
        data["date_time_iso"] = data["date_time"].apply(cls.correct_timestamp)

        paths = {
            "raw": path,
            "metadata": path
        }

        # Set schema from filename, if possible; otherwise, use default maccor
        project_schema = loadfn(PROJECT_SCHEMA)
        name = os.path.basename(path)
        special_schema_filename = project_schema.get(name.split("_")[0], {}).get("maccor")

        if special_schema_filename:
            schema = os.path.join(VALIDATION_SCHEMA_DIR, special_schema_filename)
        else:
            schema = os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml")

        return cls(data, metadata, paths=paths, schema=schema)

    def load_eis(self, paths=None):
        """Load eis from specified paths to EIS files, or automatically detect them from
        the directory where the raw maccor data file is located.

        This method sets MaccorDatapath.eis to a list MaccorEIS objects.
        This method also updates MaccorDatapath.paths to reflect EIS paths.

        Args:
            paths((str, Pathlike) or None): Paths to Maccor EIS files. If None, will automatically
                scan the directory where the raw MaccorDatapath file was located and choose all
                files matching the Maccor EIS RegEx as EIS files.

        Returns:
            None
        """

        # Automatically find EIS from directory

        if not paths:
            # todo: replace with logging
            print(f"Looking in directory {self.paths['raw']} for EIS runs, as no paths specified to `load_eis`.")
            eis_pattern = ".*.".join(self.paths["raw"].rsplit(".", 1))
            paths = glob(eis_pattern)

        eis_runs = []
        for path in tqdm.tqdm(paths, desc="Loading EIS files..."):
            eis = self.MaccorEIS.from_file(path)
            eis_runs.append(eis)

        self.eis = eis_runs
        self.paths["eis"] = paths

        # todo: add logging for if no paths added

    @staticmethod
    def quantity_sum(data, quantity, state_type):
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
        state_code = MACCOR_CONFIG["{}_state_code".format(state_type)]
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
            lambda x: MACCOR_CONFIG["end_step_code_min"] <= x <= MACCOR_CONFIG["end_step_code_max"]
        )
        # For waveform discharges, maccor seems to trigger ending_status within a step multiple times
        # As a fix, compute the actual step change using diff() on step_index and set end_step to be
        # a logical AND(step_change, end_step)
        is_step_change = data['step_index'].diff(periods=-1).fillna(value=0) != 0
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
        metadata_values = MaccorDatapath.split_string_by_fields(metadata_string,
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

        >>> MaccorDatapath.split_string_by_fields("first name: Joey  last name Montoya",
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
