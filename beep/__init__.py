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

from .config import CONFIG

# Versioning.  The python code version is frequently tagged
# with a commit hash from the repo, which is supplied via
# an environment variable by the integration build procedure
__version__ = "2021.6.3.21"
VERSION_TAG = os.environ.get("BEEP_VERSION_TAG")
if VERSION_TAG is not None:
    __version__ = "-".join([__version__, VERSION_TAG])

# Custom tqdm with optional turnoff from env
tqdm = partial(_tqdm, disable=bool(os.environ.get("TQDM_OFF")))


# All environment variables
BEEP_ENV_KEY = "BEEP_ENV"
BEEP_PARAMETERS_KEY = "BEEP_PARAMETERS_DIR"
BEEP_S3_CACHE_KEY = "BEEP_S3_CACHE"
BEEP_ENV = os.environ.get(BEEP_ENV_KEY, "local")
BEEP_PARAMETERS_DIR = os.environ.get(BEEP_PARAMETERS_KEY, "")
# Get S3 cache location from env or use default in repo
S3_CACHE = os.environ.get("BEEP_S3_CACHE", os.path.join(MODULE_DIR, "..", "s3_cache"))


# Common locations
MODULE_DIR = os.path.dirname(__file__)
CONVERSION_SCHEMA_DIR = os.path.join(MODULE_DIR, "conversion_schemas")
VALIDATION_SCHEMA_DIR = os.path.join(MODULE_DIR, "validation_schemas")
MODEL_DIR = os.path.join(MODULE_DIR, "models")


# Logging configuration
LOG_DIR = os.path.join(MODULE_DIR, "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
np.set_printoptions(precision=3)
# clear previous loggers/handlers/filters, if exists
logger = logging.getLogger(BEEP_ENV + "/beep")
logger.handlers = []
logger.filters = []
formatter = logging.Formatter(config)

# output and format
if "CloudWatch" in config[ENVIRONMENT]["logging"]["streams"]:
    if ENVIRONMENT == "stage":
        for _ in range(MAX_RETRIES):
            try:
                hdlr = watchtower.CloudWatchLogHandler(log_group="/stage/beep/services")
            except NoCredentialsError:
                time.sleep(10)
                continue
            else:
                break
        else:
            raise NoCredentialsError
    else:
        hdlr = watchtower.CloudWatchLogHandler(log_group="Worker")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
if "stdout" in config[ENVIRONMENT]["logging"]["streams"]:
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
if "file" in config[ENVIRONMENT]["logging"]["streams"]:
    log_file = os.path.join(MODULE_DIR, "Testing_logger.log")
    hdlr = logging.FileHandler(log_file, "a")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

logger.setLevel("DEBUG")
logger.propagate = False
