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
from tqdm import tqdm as _tqdm

from .config import config

try:
    from io import StringIO
except (ImportError, ModuleNotFoundError):
    from StringIO import StringIO

# Versioning.  The python code version is frequently tagged
# with a commit hash from the repo, which is supplied via
# an environment variable by the integration build procedure
__version__ = "2020.4.5"
VERSION_TAG = os.environ.get("BEEP_VERSION_TAG")
if VERSION_TAG is not None:
    __version__ = '-'.join([__version__, VERSION_TAG])

# Custom tqdm with optional turnoff from env
tqdm = partial(_tqdm, disable=bool(os.environ.get("TQDM_OFF")))

ENV_VAR = 'BEEP_ENV'

# environment
ENVIRONMENT = os.getenv(ENV_VAR)
if ENVIRONMENT is None or ENVIRONMENT not in config.keys():
    raise ValueError(f'Environment variable {ENV_VAR} must be set and be one '
                     + f'of the following: {", ".join(list(config.keys()))}. '
                     + f'Found: {ENVIRONMENT}')

MODULE_DIR = os.path.dirname(__file__)
CONVERSION_SCHEMA_DIR = os.path.join(MODULE_DIR, "conversion_schemas")
VALIDATION_SCHEMA_DIR = os.path.join(MODULE_DIR, "validation_schemas")
PROCEDURE_TEMPLATE_DIR = os.path.join(MODULE_DIR, "procedure_templates")
MODEL_DIR = os.path.join(MODULE_DIR, "models")

LOG_DIR = os.path.join(MODULE_DIR, "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Get S3 cache location from env or use default in repo
S3_CACHE = os.environ.get("BEEP_S3_CACHE",
                          os.path.join(MODULE_DIR, "..", "s3_cache"))

# logging
np.set_printoptions(precision=3)

# service (logging)
container = config[ENVIRONMENT]['logging']['container']

# initialize logger and clear previous handlers and filters, if exist
logger = logging.getLogger(ENVIRONMENT + "/beep")
logger.handlers = []
logger.filters = []

fmt_str = '{"time": "%(asctime)s", "level": "%(levelname)s", ' \
          '"service": "%(service)s", "process": "%(process)d", ' \
          '"module": "%(module)s", "func": "%(funcName)s", ' \
          '"msg": "%(message)s"}'
formatter = logging.Formatter(fmt_str)

# output and format
if 'CloudWatch' in config[ENVIRONMENT]['logging']['streams']:
    if ENVIRONMENT == "stage":
        hdlr = watchtower.CloudWatchLogHandler(log_group='/stage/beep/services')
    else:
        hdlr = watchtower.CloudWatchLogHandler(log_group='Worker')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
if 'stdout' in config[ENVIRONMENT]['logging']['streams']:
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
if 'file' in config[ENVIRONMENT]['logging']['streams']:
    log_file = os.path.join(MODULE_DIR, "Testing_logger.log")
    hdlr = logging.FileHandler(log_file, 'a')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

logger.setLevel('DEBUG')
logger.propagate = False