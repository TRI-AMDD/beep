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

For creating features and organizing them into datasets.

"""
import os
import abc
import json
from typing import Union, Tuple

import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import dumpfn

from beep.structure.base import BEEPDatapath


class BEEPFeaturizationError(BaseException):
    """Raise when a featurization-specific error occurs"""
    pass


class BEEPFeaturizer(MSONable, abc.ABC):
    """
    Base class for all beep feature generation.

    From a structured battery file representing many cycles of one cell,
    (AKA a structured datapath), produce a feature vector.

    Works for generating both
     - Vectors X to use as training vectors
     - Vectors or scalars y to use as ML targets
        (as problems may have multiple metrics to predict)

    """

    DEFAULT_HYPERPARAMETERS = {}

    def __init__(self, structured_datapath: Union[BEEPDatapath, None],
                 hyperparameters: Union[dict, None] = None):
        # If all required hyperparameters are specified, use those
        # If some subset of required hyperparameters are specified, throw error
        # If no hyperparameters are specified, use defaults
        if hyperparameters:
            if all(k in hyperparameters for k in self.DEFAULT_HYPERPARAMETERS):
                self.hyperparameters = hyperparameters
            else:
                raise BEEPFeaturizationError(
                    f"Features cannot be created with incomplete set of "
                    f"hyperparameters {hyperparameters.keys()} < "
                    f"{self.DEFAULT_HYPERPARAMETERS.keys()}!")
        else:
            self.hyperparameters = self.DEFAULT_HYPERPARAMETERS

        if structured_datapath is not None and not structured_datapath.is_structured:
            raise BEEPFeaturizationError(
                "BEEPDatapath input is not structured!")
        self.datapath = structured_datapath

        self.features = None

        # In case these features are loaded from file
        # Allow attrs which can hold relevant metadata without having
        # to reload the original datapath
        self.paths = self.datapath.paths if self.datapath else {}
        self.metadata = self.datapath.metadata.raw if self.datapath else {}
        self.linked_semiunique_id = self.datapath.semiunique_id if self.datapath else None

    @abc.abstractmethod
    def validate(self) -> Tuple[bool, Union[str, None]]:
        """
        Validate a featurizer on it's ingested datapath.

        Returns:
            (bool, str/None): The validation result and it's message.

        """
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

        if self.features is None:
            raise BEEPFeaturizationError(
                "Cannot serialize features which have not been generated.")

        features = self.features.to_dict("list")

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,

            # Core parts of BEEPFeaturizer
            "features": features,
            "hyperparameters": self.hyperparameters,
            "paths": self.paths,
            "metadata": self.metadata,
            "linked_datapath_semiunique_id": self.linked_semiunique_id
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
        bf = cls(structured_datapath=None, hyperparameters=d["hyperparameters"])
        bf.features = pd.DataFrame(d["features"])
        bf.paths = d["paths"]
        bf.metadata = d["metadata"]
        bf.linked_semiunique_id = d["linked_datapath_semiunique_id"]
        return bf

    @classmethod
    def from_json_file(cls, filename):
        """Load a structured run previously saved to file.

        .json.gz files are supported.

        Loads a BEEPFeaturizer from json.

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

    def to_json_file(self, filename):
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


class BEEPAllCyclesFeaturizer(BEEPFeaturizer):
    """Base class for featurizers that return a constant number of features
    for any number of cycles in a structured datapath.

    These features are typically used for early prediction.

    A BEEPAllCyclesFeaturizer always returns the same number of features
    for files for datapaths with any number of samples. Thus,


    [Datapath w/ 2 cycles]   ---> (vector of k features)

    [Datapath w/ 100 cycles] ---> (vector of k features)
    """
    PER_CYCLE = False


class BEEPPerCycleFeaturizer(BEEPFeaturizer):
    """Base class for featurizers that return a vector of features for
    EACH cycle in a structured datapath.

    These features are generally used for analysis

    A BEEPPerCycleFeaturizer always returns an (n x k) matrix of features
    for datapaths with n cycles each producing k features. Thus,

    [Datapath w/ 2 cycles]   ---> (2 x k feature matrix)

    [Datapath w/ 100 cycles] ---> (100 x k feature matrix)

    """
    PER_CYCLE = True
    SPECIAL_COLUMNS = ("cycle_index", "diag_pos")






