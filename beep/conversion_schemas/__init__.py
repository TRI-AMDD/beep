#  Copyright (c) 2019 Toyota Research Institute

import os

from beep import CONVERSION_SCHEMA_DIR
from monty.serialization import loadfn


ARBIN_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "arbin_conversion.yaml"))
MACCOR_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "maccor_conversion.yaml"))
FastCharge_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "FastCharge_conversion.yaml"))
xTesladiag_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "xTESLADIAG_conversion.yaml"))
INDIGO_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "indigo_conversion.yaml"))
BIOLOGIC_CONFIG = loadfn(os.path.join(CONVERSION_SCHEMA_DIR, "biologic_conversion.yaml"))

ALL_CONFIGS = [ARBIN_CONFIG, MACCOR_CONFIG]
