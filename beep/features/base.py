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

import os
import pandas as pd
from abc import ABCMeta, abstractmethod
from monty.json import MSONable
from monty.serialization import loadfn

from beep.collate import scrub_underscore_suffix, add_suffix_to_filename
from beep import MODULE_DIR

FEATURE_HYPERPARAMS = loadfn(
    os.path.join(MODULE_DIR, "features/feature_hyperparameters.yaml")
)

s = {"service": "DataAnalyzer"}


class BeepFeatures(MSONable, metaclass=ABCMeta):
    """
    Class corresponding to feature baseline feature object.

    Attributes:
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    """

    class_feature_name = "Base"

    def __init__(self, name, X, metadata):
        """
        Invokes BeepFeatures object

        Args:
            name (str): predictor object name.
            X (pandas.DataFrame): features in DataFrame format.
            metadata (dict): information about the conditions, data
                and code used to produce features

        """
        self.name = name
        self.X = X
        self.metadata = metadata

    @classmethod
    def from_run(
        cls, input_filename, feature_dir, processed_cycler_run,
            params_dict=None, parameters_path="data-share/raw/parameters"
    ):
        """
        This method contains the workflow for the creation of the feature class
        Since the workflow should be the same for all of the feature classed this
        method should not be overridden in any of the derived classes. If the class
        can be created (feature generation succeeds, etc.) then the class is returned.
        Otherwise the return value is False
        Args:
            input_filename (str): path to the input data from processed cycler run
            feature_dir (str): path to the base directory for the feature sets.
            processed_cycler_run (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
            gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            (beep.featurize.BeepFeatures): class object for the feature set
        """

        if cls.validate_data(processed_cycler_run, params_dict):
            output_filename = cls.get_feature_object_name_and_path(
                input_filename, feature_dir
            )
            feature_object = cls.features_from_processed_cycler_run(
                processed_cycler_run, params_dict, parameters_path=parameters_path
            )
            metadata = cls.metadata_from_processed_cycler_run(
                processed_cycler_run, params_dict
            )
            return cls(output_filename, feature_object, metadata)
        else:
            return False

    @classmethod
    @abstractmethod
    def validate_data(cls, processed_cycler_run, params_dict=None):
        """
        Method for validation of input data, e.g. processed_cycler_runs

        Args:
            processed_cycler_run (ProcessedCyclerRun): processed_cycler_run
                to validate
            params_dict (dict): parameter dictionary for validation

        Returns:
            (bool): boolean for whether data is validated

        """
        raise NotImplementedError

    @classmethod
    def get_feature_object_name_and_path(cls, input_path, feature_dir):
        """
        This function determines how to name the object for a specific feature class
        and creates the full path to save the object. This full path is also used as
        the feature name attribute
        Args:
            input_path (str): path to the input data from processed cycler run
            feature_dir (str): path to the base directory for the feature sets.
        Returns:
            str: the full path (including filename) to use for saving the feature
                object
        """
        new_filename = os.path.basename(input_path)
        new_filename = scrub_underscore_suffix(new_filename)

        # Append model_name along with "features" to demarcate
        # different models when saving the feature vectors.
        new_filename = add_suffix_to_filename(
            new_filename, "_features" + "_" + cls.class_feature_name
        )
        if not os.path.isdir(os.path.join(feature_dir, cls.class_feature_name)):
            os.makedirs(os.path.join(feature_dir, cls.class_feature_name))
        feature_path = os.path.join(feature_dir, cls.class_feature_name, new_filename)
        feature_path = os.path.abspath(feature_path)
        return feature_path

    @classmethod
    @abstractmethod
    def features_from_processed_cycler_run(cls, processed_cycler_run, params_dict=None,
                                           parameters_path="data-share/raw/parameters"):
        raise NotImplementedError

    @classmethod
    def metadata_from_processed_cycler_run(cls, processed_cycler_run, params_dict=None):
        if params_dict is None:
            params_dict = FEATURE_HYPERPARAMS[cls.class_feature_name]
        metadata = {
            "barcode": processed_cycler_run.metadata.barcode,
            "protocol": processed_cycler_run.metadata.protocol,
            "channel_id": processed_cycler_run.metadata.channel_id,
            "parameters": params_dict,
        }
        return metadata

    def as_dict(self):
        """
        Method for dictionary serialization
        Returns:
            dict: corresponding to dictionary for serialization
        """
        obj = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "name": self.name,
            "X": self.X.to_dict("list"),
            "metadata": self.metadata,
        }
        return obj

    @classmethod
    def from_dict(cls, d):
        """MSONable deserialization method"""
        d["X"] = pd.DataFrame(d["X"])
        return cls(**d)
