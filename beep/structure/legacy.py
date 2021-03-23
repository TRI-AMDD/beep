from monty.serialization import loadfn

from beep.structure.base import BEEPDatapath
from beep.structure.arbin import ArbinDatapath


def load_processed_json_legacy(path):
    """
    Enable loadfn capability for legacy BEEP files, since calling
    loadfn on legacy files will return dictionaries instead of objects
    or will outright fail.

    Args:
        path (str, Pathlike): Path to the structured json file.

    Returns:
        BEEPDatapath

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
