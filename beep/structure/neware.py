"""Classes and functions for handling Neware battery cycler data.

"""

import numpy as np
import pandas as pd
from monty.tempfile import ScratchDir

from beep.structure.base import BEEPDatapath
from beep.structure.maccor import MaccorDatapath
from beep.conversion_schemas import NEWARE_CONFIG


class NewareDatapath(BEEPDatapath):
    """A BEEPDatapath for ingesting and structuring Neware data files.
    """

    @classmethod
    def from_file(cls, filename):
        """Create a NewareDatapath from a raw Neware cycler file.

        Args:
            filename (str, Pathlike): file path for neware file.

        Returns:
            (NewareDatapath)
        """
        ir_column_name = '"DCIR(O)"'
        with open(filename, encoding="ISO-8859-1") as input:
            with ScratchDir("."):
                cycle_header = input.readline().replace("\t", "")
                cycle_file = open("cycle_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = cycle_header.encode("ascii", "ignore")
                cycle_header = encoded_string.decode()
                cycle_file.write(cycle_header)

                step_header = input.readline().replace("\t", "")
                ir_index = step_header.split(",").index(ir_column_name)
                step_file = open("step_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = step_header.encode("ascii", "ignore")
                step_header = encoded_string.decode()
                step_file.write(step_header)

                record_header = input.readline().replace("\t", "")
                record_header = record_header.split(",")
                record_header[0] = cycle_header.split(",")[0]
                record_header[1] = step_header.split(",")[1]
                record_header[22] = ir_column_name
                record_header = ",".join(record_header)
                record_file = open("record_file.csv", "a", encoding="ISO-8859-1")
                encoded_string = record_header.encode("ascii", "ignore")
                record_header = encoded_string.decode()
                record_file.write(record_header)

                # Read file line by line and write to the appropriate file
                cycle_number = 0
                step_number = 0
                for row, line in enumerate(input):
                    if line[:2] == r',"':
                        step_file.write(line)
                        step_number = line.split(",")[1]
                        ir_value = line.split(",")[ir_index]
                    elif line[:2] == r",,":
                        line_list = line.split(",")
                        line_list[0] = cycle_number
                        line_list[1] = step_number
                        line_list[22] = ir_value
                        line = ",".join(line_list)
                        record_file.write(line)
                    else:
                        cycle_file.write(line)
                        cycle_number = line.split(",")[0]
                record_file.close()
                step_file.close()
                cycle_file.close()

                # Read in the data and convert the column values to MKS units
                data = pd.read_csv(
                    "record_file.csv", sep=",", skiprows=0, encoding="ISO-8859-1"
                )
                data = data.loc[:, ~data.columns.str.contains("Unnamed")]
                data["Time(h:min:s.ms)"] = data["Time(h:min:s.ms)"].apply(
                    cls.step_time
                )
                data["Current(mA)"] = data["Current(mA)"] / 1000
                data["Capacitance_Chg(mAh)"] = data["Capacitance_Chg(mAh)"] / 1000
                data["Capacitance_DChg(mAh)"] = data["Capacitance_DChg(mAh)"] / 1000
                data["Engy_Chg(mWh)"] = data["Engy_Chg(mWh)"] / 1000
                data["Engy_DChg(mWh)"] = data["Engy_DChg(mWh)"] / 1000

                # Deal with missing data in the internal resistance
                data["DCIR(O)"] = data["DCIR(O)"].apply(
                    lambda x: np.nan if x == "\t-" else x
                )
                data["DCIR(O)"] = data["DCIR(O)"].fillna(method="ffill")
                data["DCIR(O)"] = data["DCIR(O)"].fillna(method="bfill")

        data["test_time"] = (
            data["Time(h:min:s.ms)"]
            .diff()
            .fillna(0)
            .apply(lambda x: 0 if x < 0 else x)
            .cumsum()
        )
        # print(data.columns)
        # print(NEWARE_CONFIG["data_types"])
        data = data.astype(NEWARE_CONFIG["data_types"])

        data.rename(NEWARE_CONFIG["data_columns"], axis="columns", inplace=True)
        data["date_time"] = data["date_time"].apply(lambda x: x.replace("\t", ""))
        data["date_time_iso"] = data["date_time"].apply(MaccorDatapath.correct_timestamp)

        metadata = dict()
        path = filename

        paths = {
            "raw": path,
            "metadata": path
        }

        return cls(data, metadata, paths)

    @staticmethod
    def step_time(x):
        """Helper function to convert the step time format from Neware h:min:s.ms into
        decimal seconds

        Args:
            x (str): The datetime string for neware in format 'h:min:s.ms'

        Returns:
            float: The time in seconds
        """
        time_list = x.split(":")
        time = (
            3600 * float(time_list[-3]) + 60 * float(time_list[-2]) + float(time_list[-1])
        )
        return time
