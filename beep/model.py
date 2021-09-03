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
from sklearn.base import BaseEstimator

from beep.features.base import BEEPFeaturizer, BEEPFeatureMatrix


class BEEPMLExperiment(MSONable):
    """
    A class for training, predicting, managing, and (de)serializing
    BEEP-based ML battery cycler experiments.

    """

    MODEL_SPACE = {
        "linear": {
            "regularization": ("lasso", "ridge", "elasticnet")
        },

    }

    def __init__(
            self,
            feature_matrix: BEEPFeatureMatrix,
            target_matrix: BEEPFeatureMatrix,
            model: Union[BaseEstimator, "str"]

    ):
        self.X = feature_matrix
        self.y = target_matrix

    @classmethod
    def from_sklearn_obj(cls):
        pass

    def train(self, target: str):

        pass

    def train_multi

    def as_dict(self):
        """Serialize a BEEPDatapath as a dictionary.

        Must not be loaded from legacy.

        Returns:
            (dict): corresponding to dictionary for serialization.

        """

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,

            # Core parts of BEEPFeaturizer
            "featurizers": [f.as_dict() for f in self.featurizers],
            "matrix": self.matrix.to_dict("list"),
        }

    @classmethod
    def from_dict(cls, d):
        """Create a BEEPDatapath object from a dictionary.

        Args:
            d (dict): dictionary represenation.

        Returns:
            beep.structure.ProcessedCyclerRun: deserialized ProcessedCyclerRun.
        """

        # no need for original datapath
        featurizers = [BEEPFeaturizer.from_dict(f) for f in d["featurizers"]]
        return cls(featurizers)

    @classmethod
    def from_json_file(cls, filename):
        """Load a structured run previously saved to file.

        .json.gz files are supported.

        Loads a BEEPFeatureMatrix from json.

        Can be used in combination with files serialized with BEEPFeatures.to_json_file.

        Args:
            filename (str, Pathlike): a json file from a structured run, serialzed with to_json_file.

        Returns:
            None
        """
        return loadfn(d)

    def to_json_file(self, filename):
        """Save a BEEPFeatureMatrix to disk as a json.

        .json.gz files are supported.

        Not named from_json to avoid conflict with MSONable.from_json(*)

        Args:
            filename (str, Pathlike): The filename to save the file to.
            omit_raw (bool): If True, saves only structured (NOT RAW) data.
                More efficient for saving/writing to disk.

        Returns:
            None
        """
        d = self.as_dict()
        dumpfn(d, filename)


if __name__ == "__main__":
    pass
