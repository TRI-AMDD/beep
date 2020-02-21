"""
Utility script for modifying @module attributes in serialized beep
classes, e. g.

Usage:
    beep_compatibility.py [DIRECTORY]

Options:
    -h --help        Show this screen
    --version        Show version

"""
import json
from tqdm import tqdm
from docopt import docopt
from monty.os.path import find_exts


def update_directory(directory):
    """
    Modifies all objects in a directory such that
    any incompatible module names (e. g. beep_ep)
    are renamed to beep

    Args:
        directory (str): directory

    Returns:
        None

    """
    fnames = find_exts(directory, "json")
    for fname in tqdm(fnames):
        with open(fname, "r") as f:
            data = json.loads(f.read())
        # For now just skip lists
        if isinstance(data, dict):
            module = data.get("@module", "")
            if "beep_ep" in module:
                data["@module"] = module.replace("beep_ep", "beep")
                with open(fname, "w") as f:
                    f.write(json.dumps(data))


if __name__ == "__main__":
    args = docopt(__doc__)
    update_directory(args['DIRECTORY'])
