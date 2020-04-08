import os
from .events import Logger, KinesisEvents
from .splice import MaccorSplice


def os_format(json_string):
    """
    Helper function to format json string into something
    that can be parsed on the command line.  For nt (windows)
    systems, uses enclosing double quotes and escaped quotes,
    for POSIX systems uses enclosing single quotes and
    no escape characters.

    Args:
        json_string (str): json string to be formatted
    """
    if os.name == "nt":
        return "\"{}\"".format(json_string.replace("\"", "\\\""))
    else:
        return "'{}'".format(json_string)
