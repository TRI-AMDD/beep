import os
import json
from typing import Union

from monty.io import zopen

from beep import logger
from beep.structure.core.run import Run



def load_run_from_ProcessedCyclerRun_file(
        filename: Union[str, os.PathLike]
    ) -> Run:
    """
    Load a Run from a legacy ProcessedCyclerRun file.
    Note that since processed cycler runs did not contain raw data,
    they cannot be restructured.

    Args:
        filename (str): Path to the ProcessedCyclerRun file.
    
    Returns:
        Run: The Run loaded from the file.
    """
    d = load_json_safe(filename)
    
    metadata = {}

    # for k in 
    pass


def load_run_from_BEEPDatapath_file(
        filename: Union[str, os.PathLike]
    ) -> Run:
    """
    Load a Run from a legacy BEEPDatapath file.
    Note that BEEPDatapath files saved with omit_raw=True will not contain
    raw data, and thus cannot be restructured.
    
    Args:
        filename (str): Path to the BEEPDatapath file.

    Returns:
        Run: The Run loaded from the file.
    """
    d = load_json_safe(filename)
    




def load_json_safe(filename: Union[str, os.PathLike]):
    """
    Safely load a json object which may be zipped or not, but
    do not uncompress it with monty.
    """
    with zopen(filename, "r") as f:
        d = json.load(f)
    return d