import os
import hashlib
import json
from collections import OrderedDict
from .parameters_lookup import get_protocol_parameters, get_project_sequence, get_diagnostic_parameters
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

    def get_path(self, string, default=None):
        return get(self, string, default=default)

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
