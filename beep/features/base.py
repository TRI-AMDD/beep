# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module and scripts for generating feature objects from structured
battery cycling data, to be used as inputs for machine learning
early prediction models.

Usage:
    featurize [INPUT_JSON]

Options:
    -h --help        Show this screen
    --version        Show version


The `featurize` script will generate features according to the methods
contained in beep.featurize.  It places output files corresponding to
features in `/data-share/features/`.

The input json must contain the following fields

* `file_list` - a list of processed cycler runs for which to generate features

The output json file will contain the following:

* `file_list` - a list of filenames corresponding to the locations of the features

Example:
```angular2
$ featurize '{"invalid_file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
    "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv"],
    "file_list": ["/data-share/structure/FastCharge_2_CH29_structure.json"]}'
{"file_list": ["/data-share/features/FastCharge_2_CH29_full_model_features.json"]}
```
"""
import abc
import os
import pandas as pd
import json
from typing import Iterable, Union
from monty.json import MSONable
from monty.serialization import dumpfn, zopen

from beep.structure.base import BEEPDatapath


class BEEPFeaturizationError(BaseException):
    """Raise when a featurization-specific error occurs"""
    pass


class BEEPFeaturizer(MSONable, abc.ABC):
    """
    Base class for all beep feature generation.

    Input a structured datapath, get an object which can generate features based
    on that data.

    """

    DEFAULT_HYPERPARAMETERS = {}

    def __init__(self, structured_datapath: Union[BEEPDatapath, None], hyperparameters: Union[dict, None] = None):
        # If all required hyperparameters are specified, use those
        # If some subset of required hyperparameters are specified, throw error
        # If no hyperparameters are specified, use defaults
        if hyperparameters:
            if all(k in hyperparameters for k in self.DEFAULT_HYPERPARAMETERS):
                self.hyperparameters = hyperparameters
            else:
                raise BEEPFeaturizationError(
                    f"Features cannot be created with incomplete set of "
                    f"hyperparameters {self.hyperparameters} < "
                    f"{self.DEFAULT_HYPERPARAMETERS.keys()}!")
        else:
            self.hyperparameters = self.DEFAULT_HYPERPARAMETERS

        if not structured_datapath is not None and structured_datapath.is_structured:
            raise BEEPFeaturizationError("BEEPDatapath input is not structured!")
        self.datapath = structured_datapath
        self.features = None

        # In case these features are loaded from file
        # Allow attrs which can hold relevant metadata without having
        # to reload the original datapath
        self.paths = self.datapath.paths
        self.metadata = self.datapath.metadata

    @abc.abstractmethod
    @property
    def required_hyperparameters(self) -> Iterable:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def create_features(self) -> None:
        """
        Should assign a dataframe to self.features.

        Returns:
            None
        """
        raise NotImplementedError


    def as_dict(self):
        """Serialize a BEEPDatapath as a dictionary.

        Must not be loaded from legacy.

        Returns:
            (dict): corresponding to dictionary for serialization.

        """

        if not self.features:
            raise BEEPFeaturizationError("Cannot serialize features which have not been generated.")

        features = self.features.to_dict("list")

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,

            # Core parts of BEEPFeaturizer
            "features": features,
            "hyperparameters": self.hyperparameters,
            "paths": self.paths,
            "metadata": self.metadata
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
        bf = cls.__init__(structured_datapath=None, hyperparameters=d["hyperparameters"])
        bf.features = pd.DataFrame(d["features"])
        bf.paths = d["paths"]
        bf.metadata = d["metadata"]
        return bf

    @classmethod
    def from_json_file(cls, filename):
        """Load a structured run previously saved to file.

        .json.gz files are supported.

        Loads a BEEPFeatures from json.

        Can be used in combination with files serialized with BEEPFeatures.to_json_file.

        Args:
            filename (str, Pathlike): a json file from a structured run, serialzed with to_json_file.

        Returns:
            None
        """
        with zopen(filename, "r") as f:
            d = json.load(f)

        # Add this structured file path to the paths dict
        paths = d.get("paths", {})
        paths["features"] = os.path.abspath(filename)
        d["paths"] = paths
        return cls.from_dict(d)

    def to_json_file(self, filename, omit_raw=False):
        """Save a BEEPFeatures to disk as a json.

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
