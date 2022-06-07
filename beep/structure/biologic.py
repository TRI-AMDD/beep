"""Classes and functions for handling BioLogic battery cyclers.
"""

import hashlib
import os.path
import json
from datetime import datetime
import pytz

import pandas as pd

from beep.structure.base import BEEPDatapath
from beep.conversion_schemas import BIOLOGIC_CONFIG


class BiologicDatapath(BEEPDatapath):
    """Datapath for ingesting and structuring BioLogic cycler data.
    """

    @staticmethod
    def _get_file_type(path, search_lines=200):
        """
        Checks on the file and gets the right separators and header length.
        Should enable seemless processing of either manual exported file (.txt) or
        automatically exported file (.csv)

        Args:
            path (str): file path to data file
            search_lines (int): number of lines to read when looking for header line

        Returns:
            sep (str): separation character between fields
            encoding (str): file encoding to expect based on extension
            header_starts_line (int): line number of the header line
            data_starts_line (int): line number where data values start
        """
        if os.path.splitext(path)[1] == ".csv":
            sep = ";"
            encoding = "utf-8"
        elif os.path.splitext(path)[1] == ".txt":
            sep = "\t"
            encoding = "iso-8859-1"
        else:
            raise TypeError("Unable to determine separator character from file extension")

        with open(path, "rb") as f:
            i = 1
            header_starts_line = None
            while header_starts_line is None:
                line = f.readline()
                if b'Ecell/V' in line and b'Variables' not in line:
                    header_starts_line = i
                    data_starts_line = i + 1
                i += 1
                if i > search_lines:
                    raise LookupError("Unable to find the header line in first {} lines of file".format(search_lines))

        return sep, encoding, header_starts_line, data_starts_line

    @classmethod
    def from_file(cls, path, mapping_file=None):
        """Creates a BEEPDatapath from a raw BioLogic battery cycler output file.

        Args:
            path (str): file path to data file
            mapping_file (str): file path to mapping file containing step transitions where cycle_index should
                increment

        Returns:
            BiologicDatapath
        """

        sep, encoding, header_line, data_starts_line = cls._get_file_type(path)
        column_map = BIOLOGIC_CONFIG["data_columns"]

        raw = dict()
        i = 0
        with open(path, "rb") as f:

            empty_lines = 0  # used to find the end of the data entries.
            while empty_lines < 2:  # finding 2 empty lines in a row => EOF
                line = f.readline()
                i += 1
                if i == header_line:
                    header = str(line.decode(encoding=encoding))
                    columns = header.split(sep)
                    for c in columns:
                        raw[c] = list()
                if i >= data_starts_line:
                    line = line.decode(encoding=encoding)
                    if len(line) == 0:
                        empty_lines += 1
                        continue
                    items = line.split(sep)
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

        if "cycle_index" not in columns and not mapping_file:
            raw["cycle_index"] = [int(float(i)) for i in raw["cycle number"]]
        elif "cycle_index" not in columns and mapping_file:
            if "Loop" in raw.keys():
                raw["cycle_index"] = get_cycle_index(raw["Ns"], mapping_file, loop_list=raw["Loop"])
            else:
                raw["cycle_index"] = get_cycle_index(raw["Ns"], mapping_file)

        data = dict()
        for column_name in column_map.keys():
            data[column_map[column_name]["beep_name"]] = raw[column_name]
        data["data_point"] = list(range(1, len(raw["cycle number"]) + 1))

        data = pd.DataFrame(data)
        metadata_path = path.replace(".csv", ".mpl")
        metadata = cls.parse_metadata(metadata_path)
        metadata["filename"] = path
        metadata["_today_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # standardizing time format
        pacific = pytz.timezone("US/Pacific")
        utc = pytz.timezone("UTC")
        data["date_time_iso"] = data["date_time"].apply(
            lambda x: pacific.localize(datetime.strptime(x, "%m/%d/%Y %H:%M:%S.%f"),
                                       is_dst=True).astimezone(utc).isoformat()
        )
        data["test_time"] = data["date_time"].apply(
            lambda x: (datetime.strptime(x, "%m/%d/%Y %H:%M:%S.%f") -
                       datetime.strptime(data["date_time"].iloc[0], "%m/%d/%Y %H:%M:%S.%f")).total_seconds()
        )
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


def get_cycle_index(ns_list, serialized_transition_fp, loop_list=None):
    """
    Processes CSV files generated from several biologic techniques
    and creates a new set of CSVs with an additional "cycle_index" column.

    Args:
        df (pandas.DataFrame): data frame of biologic file
        serialized_transition_fp (path): path to mapping file containing step transitions where
            cycle index should increment

    Returns:
        df (pandas.DataFrame): data frame of biologic file with cycle_index and Tech Num added

    """

    serializer = CycleTransitionRulesSerializer()
    cycle_num = 1

    with open(serialized_transition_fp, "r") as f:
        data = f.read()
        cycle_transition_rules = serializer.parse_json(data)

    cycle_num += cycle_transition_rules.adv_cycle_on_start

    prev_seq_num = int(ns_list[0])
    if loop_list:
        prev_loop_num = int(loop_list[0])
    cycle_nums = []
    tech_nums = []
    # TODO speed up by reducing logic and use list comprehension
    if loop_list:
        for indx, ns in enumerate(ns_list):
            seq_num = int(ns)
            loop_num = int(loop_list[indx])
            # a transition may occur because of a loop technique or a loop seq,
            # it is possible to double count cycle advances if we don't handle them separately
            if loop_num != prev_loop_num:
                cycle_num += cycle_transition_rules.adv_cycle_on_tech_loop
            elif seq_num != prev_seq_num:
                transition = (prev_seq_num, seq_num)
                cycle_num += cycle_transition_rules.adv_cycle_seq_transitions.get(
                    transition, 0
                )
            prev_loop_num = loop_num
            prev_seq_num = seq_num
            cycle_nums.append(cycle_num)
            tech_nums.append(cycle_transition_rules.tech_num)
    else:
        for indx, ns in enumerate(ns_list):
            seq_num = int(ns)
            if seq_num != prev_seq_num:
                transition = (prev_seq_num, seq_num)
                cycle_num += cycle_transition_rules.adv_cycle_seq_transitions.get(
                    transition, 0
                )
            prev_seq_num = seq_num
            cycle_nums.append(cycle_num)
            tech_nums.append(cycle_transition_rules.tech_num)

    return cycle_nums


class CycleTransitionRules:
    def __init__(
        self,
        tech_num,
        tech_does_loop,
        adv_cycle_on_start,
        adv_cycle_on_tech_loop,
        adv_cycle_seq_transitions,
        debug_adv_cycle_on_step_transitions={},
    ):
        self.tech_num = tech_num
        self.tech_does_loop = tech_does_loop
        self.adv_cycle_on_start = adv_cycle_on_start
        self.adv_cycle_on_tech_loop = adv_cycle_on_tech_loop
        self.adv_cycle_seq_transitions = adv_cycle_seq_transitions
        self.debug_adv_cycle_on_step_transitions = debug_adv_cycle_on_step_transitions

    def __repr__(self):
        return (
            "{\n"
            + "  tech_num: {},\n".format(self.tech_num)
            + "  tech_does_loop: {},\n".format(self.tech_does_loop)
            + "  adv_cycle_on_start: {},\n".format(self.adv_cycle_on_start)
            + "  adv_cycle_on_tech_loop: {},\n".format(self.adv_cycle_on_tech_loop)
            + "  adv_cycle_seq_transitions: {},\n".format(
                self.adv_cycle_seq_transitions
            )
            + "  debug_adv_cycle_on_step_transitions: {},\n".format(
                self.debug_adv_cycle_on_step_transitions
            )
            + "}\n"
        )


class CycleTransitionRulesSerializer:
    def json(self, cycle_transition_rules, indent=2):
        parseable_adv_cycle_seq_transitions = []
        for (
            s,
            t,
        ), adv_cycle_count in cycle_transition_rules.adv_cycle_seq_transitions.items():
            parseable_adv_cycle_seq_transitions.append(
                {
                    "source": s,
                    "target": t,
                    "adv_cycle_count": adv_cycle_count,
                }
            )

        parseable_debug_adv_cycle_on_step_transitions = []
        for (
            s,
            t,
        ), adv_cycle_count in (
            cycle_transition_rules.debug_adv_cycle_on_step_transitions.items()
        ):
            parseable_debug_adv_cycle_on_step_transitions.append(
                {
                    "source": s,
                    "target": t,
                    "adv_cycle_count": adv_cycle_count,
                }
            )

        obj = {
            "tech_num": cycle_transition_rules.tech_num,
            "tech_does_loop": cycle_transition_rules.tech_does_loop,
            "adv_cycle_on_start": cycle_transition_rules.adv_cycle_on_start,
            "adv_cycle_on_tech_loop": cycle_transition_rules.adv_cycle_on_tech_loop,
            # these are (int, int) -> int maps, tuples cannot be represented in json
            "adv_cycle_seq_transitions": parseable_adv_cycle_seq_transitions,
            "debug_adv_cycle_on_step_transitions": parseable_debug_adv_cycle_on_step_transitions,
        }

        return json.dumps(obj, indent=indent)

    def parse_json(self, serialized):
        data = json.loads(serialized)

        tech_num = data["tech_num"]
        tech_does_loop = data["tech_does_loop"]
        adv_cycle_on_start = data["adv_cycle_on_start"]
        adv_cycle_on_tech_loop = data["adv_cycle_on_tech_loop"]

        adv_cycle_seq_transitions = {}
        for d in data["adv_cycle_seq_transitions"]:
            transition = (d["source"], d["target"])
            adv_cycle_seq_transitions[transition] = d["adv_cycle_count"]

        debug_adv_cycle_on_step_transitions = {}
        for d in data["debug_adv_cycle_on_step_transitions"]:
            transition = (d["source"], d["target"])
            debug_adv_cycle_on_step_transitions[transition] = d["adv_cycle_count"]

        return CycleTransitionRules(
            tech_num,
            tech_does_loop,
            adv_cycle_on_start,
            adv_cycle_on_tech_loop,
            adv_cycle_seq_transitions,
            debug_adv_cycle_on_step_transitions,
        )
