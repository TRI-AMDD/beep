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
import copy
import abc
import json
import hashlib
from typing import Union, Tuple, List

import pandas as pd
from monty.io import zopen
from monty.json import MSONable, MontyDecoder
from monty.serialization import loadfn, dumpfn

from beep.structure.base import BEEPDatapath


class BEEPFeaturizationError(BaseException):
    """Raise when a featurization-specific error occurs"""
    pass


class BEEPFeatureMatrixError(BaseException):
    """ Raise when there is a BEEP-specific problem with a dataset"""
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
                    f"hyperparameters {hyperparameters.keys()} < "
                    f"{self.DEFAULT_HYPERPARAMETERS.keys()}!")
        else:
            self.hyperparameters = self.DEFAULT_HYPERPARAMETERS

        if not (structured_datapath is None or structured_datapath.is_structured):
            raise BEEPFeaturizationError("BEEPDatapath input is not structured!")
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
            raise BEEPFeaturizationError("Cannot serialize features which have not been generated.")

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


class BEEPFeatureMatrix(MSONable):
    """
    Create an (n battery cycler files) x (k features) array composed of
    m BEEPFeaturizer objects.

    Args:
        beepfeaturizers ([BEEPFeaturizer]): A list of BEEPFeaturizer objects

    """

    OP_DELIMITER = "::"

    def __init__(self, beepfeaturizers: List[BEEPFeaturizer]):

        if beepfeaturizers:
            dfs_by_file = {bf.paths.get("structured", "no file found"): [] for bf in beepfeaturizers}
            # big_df_rows = {bf.__class__.__name__: [] for bf in beepfeaturizers}
            unique_features = {}
            for i, bf in enumerate(beepfeaturizers):
                if bf.features is None:
                    raise BEEPFeatureMatrixError(f"BEEPFeaturizer {bf} has not created features")
                elif bf.features.shape[0] != 1:
                    raise BEEPFeatureMatrixError(f"BEEPFeaturizer {bf} features are not 1-dimensional.")
                else:
                    bfcn = bf.__class__.__name__

                    fname = bf.paths.get("structured", None)
                    if not fname:
                        raise BEEPFeatureMatrixError(
                            "Cannot join features automatically as no linking can be done "
                            "based on original structured filename."
                        )

                    # Check for any possible feature collisions using identical featurizers
                    # on identical files

                    # sort params for this featurizer obj by key
                    params = sorted(list(bf.hyperparameters.items()), key=lambda x: x[0])

                    # Prevent identical features from identical input files
                    # create a unique operation string for the application of this featurizer
                    # on a specific file, this op string will be the same as long as
                    # the featurizer class name, hyperparameters, and class are the same

                    param_str = "-".join([f"{k}:{v}" for k, v in params])
                    param_hash = hashlib.sha256(param_str.encode("utf-8")).hexdigest()

                    # Get an id for this featurizer operation (including hyperparameters)
                    # regardless of the file it is applied on
                    feature_op_id = f"{bfcn}{self.OP_DELIMITER}{param_hash}"

                    # Get an id for this featurizer operation (including hyperparameters)
                    # on THIS SPECIFIC file.
                    file_feature_op_id = f"{fname}{self.OP_DELIMITER}{bfcn}{self.OP_DELIMITER}{param_hash}"

                    # Get a unique id for every feature generated by a specific
                    # featurizer on a specific file.
                    this_file_feature_columns_ids = \
                        [
                            f"{file_feature_op_id}{self.OP_DELIMITER}{c}" for c in bf.features.columns
                        ]

                    # Check to make sure there are no duplicates of the exact same feature for
                    # the exact same featurizer with the exact same hyperparameters on the exact
                    # same file.
                    collisions = {c: f for c, f in unique_features.items() if c in this_file_feature_columns_ids}
                    if collisions:
                        raise BEEPFeatureMatrixError(
                            f"Multiple features generated with identical classes and identical hyperparameters"
                            f" attempted to be joined into same dataset; \n"
                            f"{bfcn} features collide with existing: \n{collisions}"
                        )
                    for c in this_file_feature_columns_ids:
                        unique_features[c] = bfcn

                    # Create consistent scheme for naming features regardless of file
                    df = copy.deepcopy(bf.features)
                    consistent_column_names = [f"{c}{self.OP_DELIMITER}{feature_op_id}" for c in df.columns]
                    df.columns = consistent_column_names

                    df.index = [fname] * df.shape[0]
                    df.index.rename("filename", inplace=True)
                    dfs_by_file[fname].append(df)

            rows = []
            for filename, dfs in dfs_by_file.items():
                row = pd.concat(dfs, axis=1)
                row = row[sorted(row.columns)]
                rows.append(row)
            self.matrix = pd.concat(rows, axis=0)

        else:
            self.matrix = None

        self.featurizers = beepfeaturizers

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
        # no need for original datapaths, as their ref paths should
        # be in the subobjects
        featurizers = [MontyDecoder().process_decoded(f) for f in d["featurizers"]]
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
        return loadfn(filename)

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
