import os
import warnings
from .events import Logger, KinesisEvents
from .splice import MaccorSplice


def warn_os():
    """Helper function to issue warning to CLI invocations on windows"""
    if os.name == "nt":
        warnings.warn("Command-line scripts are currently "
                      "unsupported with direct json input "
                      "on Windows, use with caution")

