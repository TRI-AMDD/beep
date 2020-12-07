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
""" Parsing and conversion of maccor procedure files to arbin schedule files"""

import os
import re
import copy
import xmltodict
from beep.protocol import PROTOCOL_SCHEMA_DIR
from monty.serialization import loadfn
from collections import OrderedDict
from pydash import get, unset

# magic number for biologic
END_SEQ_NUM = 9999

class MaccorToBiologicMb:
    """
    Collection of methods to convert maccor protocol files to biologic modulo bat protocol files
    Differences in underlying hardware and software makes this impossible in some cases
    or forces workarounds that may fail. Certain work-arounds may be attempted and are
    documented under the conversion function.
    """

    def __init__(self):
        BIOLOGIC_SCHEMA = loadfn(
            os.path.join(PROTOCOL_SCHEMA_DIR, "biologic_mb_schema.yaml")
        )
        schema = OrderedDict(BIOLOGIC_SCHEMA)
        self.blank_seq = OrderedDict(schema["blank_seq"])


    def _get_decimal_sig_figs(self, val_str):
        match_p10 = re.search("(e|E)([-+]?[0-9]+)", val_str)
        p10 = 0 if match_p10 is None else int(match_p10.groups()[1])

        match_sig_figs = re.search("\\.([0-9]*[1-9])", val_str)
        explicit_sig_figs = 0 if match_sig_figs is None else len(match_sig_figs.groups(1)[0])

        return explicit_sig_figs - p10

    def _convert_volts(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)
        if num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mV"
        else:
            return "{:.3f}".format(num), "V"

    def _convert_amps(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)

        if num < 1e-9 or decimal_sig_figs > 12:
            return "{:.3f}".format(num * 1e12), "pA"
        if num < 1e-6 or decimal_sig_figs > 9:
            return "{:.3f}".format(num * 1e9), "nA"
        elif num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}A"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mA"
        else:
            return "{:.3f}".format(num), "A"
            
    def _convert_watts(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)

        if num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}W"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mW"
        else:
            return "{:.3f}".format(num), "W"

    def _convert_ohms(self, val_str):
        decimal_sig_figs = self._get_decimal_sig_figs(val_str)
        num = float(val_str)
        print(decimal_sig_figs)

        if num < 1e-3 or decimal_sig_figs > 6:
            return "{:.3f}".format(num * 1e6), "\N{Micro Sign}Ohms"
        elif num < 1 or decimal_sig_figs > 3:
            return "{:.3f}".format(num * 1e3), "mOhms"
        elif num < 1e3 or decimal_sig_figs > 0:
            return "{:.3f}".format(num), "Ohms"
        elif num < 1e6 or decimal_sig_figs > -3:
            return "{:.3f}".format(num * 1e-3), "kOhms"
        else:
            return "{:.3f}".format(num * 1e-6), "MOhms"