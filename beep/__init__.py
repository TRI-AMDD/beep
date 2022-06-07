#  Copyright (c) 2019 Toyota Research Institute
"""
Top-level module for beep.  Put anything that should
be available in the beep namespace here.
"""
import os
import logging
import sys
from functools import partial
import numpy as np
import watchtower
import time
from botocore.exceptions import NoCredentialsError
from tqdm import tqdm as _tqdm

try:
    from io import StringIO
except (ImportError, ModuleNotFoundError):
    from StringIO import StringIO

# Versioning.  The python code version is frequently tagged
# with a commit hash from the repo, which is supplied via
# an environment variable by the integration build procedure
__version__ = "2022.5.2.14"
VERSION_TAG = os.environ.get("BEEP_VERSION_TAG")
if VERSION_TAG is not None:
    __version__ = "-".join([__version__, VERSION_TAG])

# Custom tqdm with optional turnoff from env
tqdm = partial(_tqdm, disable=bool(os.environ.get("TQDM_OFF")))


JSON_LOG_FMT = {"fmt": (
    '{"time": "%(asctime)s", "level": "%(levelname)s", '
    '"process": "%(process)d", '
    '"module": "%(module)s", "func": "%(funcName)s", '
    '"msg": "%(message)s"}'
)
}

HUMAN_LOG_FMT = {
    "fmt": "%(asctime)s %(levelname)-8s %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S"
}


# Common locations
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
CONVERSION_SCHEMA_DIR = os.path.join(MODULE_DIR, "conversion_schemas")
FEATURES_DIR = os.path.join(MODULE_DIR, "features")
VALIDATION_SCHEMA_DIR = os.path.join(MODULE_DIR, "validation_schemas")
PROTOCOL_PARAMETERS_DIR = os.path.join(MODULE_DIR, "protocol_parameters")
MODEL_DIR = os.path.join(MODULE_DIR, "models")


# All environment variables
BEEP_S3_CACHE_KEY = "BEEP_S3_CACHE"

# Get S3 cache location from env or use default in repo
S3_CACHE = os.environ.get(BEEP_S3_CACHE_KEY, os.path.join(MODULE_DIR, "..", "s3_cache"))

# Logging configuration
LOG_DIR = os.path.join(MODULE_DIR, "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
np.set_printoptions(precision=3)
# clear previous loggers/handlers/filters, if exists
logger = logging.getLogger("beep")
logger.handlers = []
logger.filters = []
formatter_stdout = logging.Formatter(**HUMAN_LOG_FMT)
formatter_jsonl = logging.Formatter(**JSON_LOG_FMT)

# Stdout log will always be enabled
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(formatter_stdout)
logger.addHandler(hdlr)

logger.setLevel("DEBUG")
logger.propagate = False
