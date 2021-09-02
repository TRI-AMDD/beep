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

Usage:
    structure [INPUT_JSON]

Options:
    -h --help       Show this screen
    --version       Show version


The `structure` script will run the data structuring on specified filenames corresponding
to validated raw cycler files.  It places the structured datafiles in `/data-share/structure`.

The input json must contain the following fields:
* `file_list` - a list of full path filenames which have been processed
* `validity` - a list of boolean validation results, e. g. `[True, True, False]`

The output json contains the following fields:

* `invalid_file_list` - a list of invalid files according to the validity
* `file_list` - a list of files which have been structured into processed_cycler_runs

Example:
```angular2
$ structure '{"validity": [false, false, true],
             file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
                          "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv",
                          "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"]}''
{"invalid_file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
                       "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv"],
 "file_list": ["/data-share/structure/FastCharge_2_CH29_structure.json"]}
```
"""

import re

from docopt import docopt
from monty.serialization import loadfn

from beep import logger, __version__
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
        return processed_obj


def main():
    """Main function of this module, takes in arguments of an input
    and output filename and uses the input file to create a
    structured data output for analysis/ML processing.
    """
    logger.info("starting", extra=SERVICE_CONFIG)
    logger.info("Running version=%s", __version__, extra=SERVICE_CONFIG)
    try:
        args = docopt(__doc__)
        input_json = args["INPUT_JSON"]
        print(process_file_list_from_json(input_json))
    except Exception as e:
        logger.error(str(e), extra=SERVICE_CONFIG)
        raise e
    logger.info("finish", extra=SERVICE_CONFIG)
    return None


if __name__ == "__main__":
    main()
