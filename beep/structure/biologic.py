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
            raise NotImplementedError("Missing cycle index and step mapping file")

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


def add_cycle_nums_to_file(
    technique_csv_file_paths,
    technique_serialized_transition_rules_file_paths,
    technique_csv_out_file_paths,
):
    """
    Processes CSV files generated from several biologic techniques
    and creates a new set of CSVs with an additional "cycle_index" column.

    accepts
      - technique_csv_file_paths: list of file paths to Biologic CSVs
      - technique_serialized_transition_rules_file_paths: list of file paths to serialized CycleTransitionRules
      - technique_csv_out_file_paths: list of filepaths to write new data to

    side-effects
       - writes a new CSV file for every entry in csv_and_transition_rules_file_paths

    invariants
        - all arguments must be of the same length
        - the i-th entry form a logical tuple
        - technique files appear in the order in which they were created
          e.g. technique 1, then technique 2 etc.

    example call:
    add_cycle_nums_to_csvs(
        [
            os.path.join(MY_DIR, "protocol1_2a_technique_1.csv"),
            os.path.join(MY_DIR, "protocol1_2a_technique_2.csv"),
        ],
        [
            os.path.join(MY_DIR, "protocol1_technique_1_transition_rules.json"),
            os.path.join(MY_DIR, "protocol1_technique_2_transition_rules.json"),
        ],
        [
            os.path.join(MY_DIR, "protocol1_2a_technique_1_processed.csv"),
            os.path.join(MY_DIR, "protocol1_2a_technique_2_processed.csv"),
        ]
    )
    """
    assert len(technique_csv_file_paths) == len(technique_csv_out_file_paths)
    assert len(technique_csv_file_paths) == len(
        technique_serialized_transition_rules_file_paths
    )

    technique_conversion_filepaths = zip(
        technique_csv_file_paths,
        technique_serialized_transition_rules_file_paths,
        technique_csv_out_file_paths,
    )

    serializer = CycleTransitionRulesSerializer()
    cycle_num = 1
    for csv_fp, serialized_transition_fp, csv_out_fp in technique_conversion_filepaths:
        with open(serialized_transition_fp, "r") as f:
            data = f.read()
            cycle_transition_rules = serializer.parse_json(data)

        df = pd.read_csv(csv_fp, sep=";")

        cycle_num += cycle_transition_rules.adv_cycle_on_start

        prev_seq_num = int(df.iloc[0]["Ns"])
        prev_loop_num = int(df.iloc[0]["Loop"])
        cycle_nums = []
        tech_nums = []
        for _, row in df.iterrows():
            seq_num = int(row["Ns"])
            loop_num = int(row["Loop"])

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

        df["cycle_index"] = cycle_nums
        df["Tech Num"] = tech_nums
        df.to_csv(csv_out_fp, sep=";")


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
