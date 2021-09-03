"""
For assembling datasets, running ML experiments, and reading/writing static
files for battery prediction given a feature set.
"""
import json
import copy
import hashlib
import os
from typing import List, Union
from functools import reduce

import pandas as pd
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
from monty.io import zopen

from beep.features.base import BEEPFeaturizer


class BEEPMLExperiment(MSONable):
    """
    A class for training, predicting, managing, and (de)serializing
    BEEP-based ML battery cycler experiments.

    """

    MODEL_SPACE = {
        "linear": {"regularization": ("lasso", "ridge", "elastic")}

    }

    def __init__(self, dataset, model_name, model_hps):
        self.dataset = dataset


if __name__ == "__main__":
    from monty.serialization import loadfn


    bfs = []
    dirname = "/Users/ardunn/alex/tri/code/beep/beep/CLI_TEST_FILES_FEATURIZATION/output"
    for fname in os.listdir(dirname):
        abs_fname = os.path.join(dirname, fname)
        d = loadfn(abs_fname)
        bfs.append(d)

    bfm = BEEPFeatureMatrix(bfs)

    print(bfm.matrix)