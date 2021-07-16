"""Classes and functions for handling BioLogic battery cyclers.
"""

import hashlib
from datetime import datetime

import pandas as pd

from beep.structure.base import BEEPDatapath
from beep.conversion_schemas import BIOLOGIC_CONFIG


class BiologicDatapath(BEEPDatapath):
    """Datapath for ingesting and structuring BioLogic cycler data.
    """

    @classmethod
    def from_file(cls, path):
        """Creates a BEEPDatapath from a raw BioLogic battery cycler output file.

        Args:
            path (str): file path to data file

        Returns:
            BiologicDatapath
        """

        header_line = 1  # specific to file layout
        data_starts_line = 2  # specific to file layout
        column_map = BIOLOGIC_CONFIG["data_columns"]

        raw = dict()
        i = 0
        with open(path, "rb") as f:

            empty_lines = 0  # used to find the end of the data entries.
            while empty_lines < 2:  # finding 2 empty lines in a row => EOF
                line = f.readline()
                i += 1

                if i == header_line:
                    header = str(line.strip())[2:-1]
                    columns = header.split(";")
                    for c in columns:
                        raw[c] = list()

                if i >= data_starts_line:
                    line = line.strip().decode()
                    if len(line) == 0:
                        empty_lines += 1
                        continue
                    items = line.split(";")
                    for ci in range(len(items)):
                        column_name = columns[ci]
                        data_type = column_map.get(column_name, dict()).get(
                            "data_type", str
                        )
                        scale = column_map.get(column_name, dict()).get("scale", 1.0)
                        item = items[ci]
                        if data_type == "int":
                            item = int(float(item))
                        if data_type == "float":
                            item = float(item) * scale
                        raw[column_name].append(item)

        data = dict()
        for column_name in column_map.keys():
            data[column_map[column_name]["beep_name"]] = raw[column_name]
        data["data_point"] = list(range(1, len(raw["cycle number"]) + 1))

        data = pd.DataFrame(data)
        data.loc[data.step_index % 2 == 0, "charge_capacity"] = abs(
            data.charge_capacity
        )
        data.loc[data.step_index % 2 == 1, "charge_capacity"] = 0
        data.loc[data.step_index % 2 == 1, "discharge_capacity"] = abs(
            data.discharge_capacity
        )
        data.loc[data.step_index % 2 == 0, "discharge_capacity"] = 0

        metadata_path = path.replace(".csv", ".mpl")
        metadata = cls.parse_metadata(metadata_path)
        metadata["filename"] = path
        metadata["_today_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        paths = {
            "raw": path,
            "metadata": metadata_path
        }

        return cls(data, metadata, paths)

    @staticmethod
    def parse_metadata(metadata_path):
        """Extracts BioLogic metadata from metadata file.

        Iterates through a biologic metadata file and extracts the meta fields:
            - cell_id
            - barcode
            - protocol

        Args:
            metadata_path (str): path to metadata file

        Returns:
            (dict): dictionary with metadata fields
        """

        flag_cell_id = False
        flag_barcode = False
        flag_protocol = False
        flag_protocol_start = False
        protocol_text = ""
        max_lines = 10000

        metadata = dict()

        with open(metadata_path, "rb") as f:

            i = 0
            while True:
                line = f.readline()
                line = str(line)
                if line.startswith("b'"):
                    line = line[2:]
                if line.endswith("'"):
                    line = line[:-1]
                if line.endswith("\\r\\n"):
                    line = line[:-4]

                if line.strip().split(" : ")[0] == "Run on channel":
                    channel_id = line.strip().split(' : ')[1]
                    channel_id = channel_id.split(' ')[0]
                    channel_id = int(str(ord(channel_id[0]) - 64) + channel_id[1:])
                    metadata["channel_id"] = channel_id
                    flag_cell_id = True

                if "Comments" in line.strip().split(" : ")[0]:
                    barcode = line.strip().split(" : ")[-1]
                    metadata["barcode"] = barcode
                    flag_barcode = True

                if "Cycle Definition" in line.strip().split(" : ")[0]:
                    protocol_text += line + "\n"
                    flag_protocol_start = True

                if flag_protocol_start:
                    if line == "":
                        flag_protocol = True
                        protocol = hashlib.md5(protocol_text.encode()).digest()
                        metadata["protocol"] = protocol
                    else:
                        protocol_text += line + "\n"

                done = flag_cell_id and flag_barcode and flag_protocol
                if done:
                    break

                i += 1
                if i > max_lines:
                    break
        return metadata
