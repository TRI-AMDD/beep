# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line/batch interface for structuring many battery cycler runs.

"""

import re

from monty.serialization import loadfn

from beep.conversion_schemas import (
    FastCharge_CONFIG,
    xTesladiag_CONFIG,
    ARBIN_CONFIG,
    MACCOR_CONFIG,
    INDIGO_CONFIG,
    BIOLOGIC_CONFIG,
    NEWARE_CONFIG
)
from beep.structure.arbin import ArbinDatapath
from beep.structure.maccor import MaccorDatapath
from beep.structure.neware import NewareDatapath
from beep.structure.indigo import IndigoDatapath
from beep.structure.biologic import BiologicDatapath
from beep.structure.battery_archive import BatteryArchiveDatapath
from beep.structure.base import BEEPDatapath


def auto_load(filename):
    """Load any supported raw battery cycler file to the correct Datapath automatically.

    Matches raw file patterns to the correct datapath and returns the datapath object.

    Example:
        auto_load("2017-05-09_test-TC-contact_CH33.csv")

        >>> <ArbinDatapath object>

        auto_load("PreDiag_000287_000128short.092")

        >>> <MaccorDatapath object>

    Args:
        filename (str, Pathlike): string corresponding to battery cycler file filename.

    Returns:
        (beep.structure.base.BEEPDatapath): The datapath child class corresponding to this file.

    """
    if re.match(ARBIN_CONFIG["file_pattern"], filename) or re.match(FastCharge_CONFIG["file_pattern"], filename):
        return ArbinDatapath.from_file(filename)
    elif re.match(MACCOR_CONFIG["file_pattern"], filename) or re.match(xTesladiag_CONFIG["file_pattern"], filename):
        return MaccorDatapath.from_file(filename)
    elif re.match(INDIGO_CONFIG["file_pattern"], filename):
        return IndigoDatapath.from_file(filename)
    elif re.match(BIOLOGIC_CONFIG["file_pattern"], filename):
        return BiologicDatapath.from_file(filename)
    elif re.match(NEWARE_CONFIG["file_pattern"], filename):
        return NewareDatapath.from_file(filename)
    elif re.match(BatteryArchiveDatapath.FILE_PATTERN, filename):
        return BatteryArchiveDatapath.from_file(filename)
    else:
        raise ValueError("{} does not match any known file pattern".format(filename))


def auto_load_processed(path):
    """Load processed BEEP .json files regardless of their class.

    Enables loadfn capability for legacy BEEP files, since calling
    loadfn on legacy files will return dictionaries instead of objects
    or will outright fail.

    Examples:

        auto_load_processed("maccor_file_structured.json")

        >>> <MaccorDatapath object>

        auto_load_processed("neware_file_structured.json")

        >>> <NewareDatapath object>

        auto_load_processed("old_Biologic_ProcessedCyclerRun_legacy.json")

        >>> <BiologicDatapath object>

    Args:
        path (str, Pathlike): Path to the structured json file.

    Returns:
        (BEEPDatapath)

    """

    processed_obj = loadfn(path)

    if not isinstance(processed_obj, BEEPDatapath):
        if isinstance(processed_obj, dict):
            # default to Arbin file for BEEPDatapath purposes
            return ArbinDatapath.from_json_file(path)
        else:
            raise TypeError(f"Unknown type for legacy processed json! `{type(processed_obj)}`")
    else:

        # Processed object must have path manually appended
        # as loadfn cannot trigger .from_json_file directly
        processed_obj.paths["structured"] = path
        return processed_obj
