import os
import hashlib
import json
import datetime
import re
from collections import OrderedDict
from .events import Logger, KinesisEvents
from .splice import MaccorSplice
from pydash import get, set_with, unset, merge


class DashOrderedDict(OrderedDict):
    """
    Nested data structure with pydash enabled
    getters and setters.  Nested values can
    be set using dot notation, e. g.

    >>> dod = DashOrderedDict()
    >>> dod.set('key1.key2', 5)
    >>> print(dod['key1']['key2'])
    >>> 5
    """

    def set(self, string, value):
        set_with(self, string, value, lambda x: OrderedDict())

    def get(self, string):
        return get(self, string)

    def unset(self, string):
        unset(self, string)

    def merge(self, obj):
        merge(self, obj)

    def __str__(self):
        return "{}:\n{}".format(self.__class__.__name__, json.dumps(self, indent=4))

    def __repr__(self):
        return self.__str__()


def hash_file(filename):
    """
    Utility function to hash a file

    Args:
        filename (str): name fo file to hash
    """
    with open(filename, "rb") as f:
        chunk = f.read()
    return hashlib.md5(chunk).digest()


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
        return '"{}"'.format(json_string.replace('"', '\\"'))
    else:
        return "'{}'".format(json_string)


def get_new_version(current_ver):
    """
    Helper function to increment semantic versioning by date

    Args:
        current_ver (str): string-formatted date as YYYY.MM.DD,
            or YYYY.MM.DD-post{NUMBER} for post-versions released
            on the same day

    Returns:
        (str) new version string

    """
    today = datetime.datetime.today().strftime("%Y.%-m.%-d")

    # Extract current_ver_date
    current_ver_date = re.sub(r"-post\d+", "", current_ver)

    if today == current_ver_date:
        if "post" in current_ver:
            # Increment post by 1
            new_ver = re.sub(r"post(\d+)", lambda exp: "post{}".format(int(exp.group(1)) + 1), current_ver)
        else:
            # Append "-post0" to version
            new_ver = "{}-post0".format(current_ver)
    else:
        # Set new version as today
        new_ver = today

    return new_ver
