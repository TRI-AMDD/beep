"""
For assembling datasets, running ML experiments, and reading/writing static
files for battery prediction given a feature set.
"""

from typing import List, Union

from monty.json import MSONable

from beep.features.base import BEEPFeaturizer


class BEEPDatasetError(BaseException):
    """ Raise when there is a BEEP-specific problem with a dataset"""
    pass


class BEEPDataset(MSONable):
    """Assemble a dataset from:

    - BEEPFeaturizers objects
    - IDs of feature classes to be used for features
        - will automatically select all features from
          all of these classes
    - IDs of target class and target feature name
    """

    def __init__(
            self,
            beepfeaturizers: List[BEEPFeaturizer],
            features_classes: List[BEEPFeaturizer],
            target_class: Union[BEEPFeaturizer, None],
            target_name: Union[str, None]
    ) -> None:

        required_classes = features_classes + [target_class] if target_class else features_classes
        # Check to make sure at least some features are present
        # for each featurizer
        for bf in beepfeaturizers:
            if bf.features is None:
                raise BEEPDatasetError(f"BEEPFeaturizer {bf} has not created features")
            if bf.__class__ not in required_classes:
                raise BEEPDatasetError(f"Extraneous class {bf.__class__} not listed in features nor target")

        passed_classed = [bf.__class__ for bf in beepfeaturizers]
        required_classes_present = {bfc.__name__: bfc in passed_classed for bfc in beepfeaturizers}

        if not all(required_classes_present):
            missing_classes = [bfc for bfc, present in required_classes_present.items() if not present]
            raise BEEPDatasetError(f"The following classes are missing from the input featurizers: {missing_classes}")

        self.featurizers = beepfeaturizers
        self.feature_classes = features_classes
        self.target_class = target_class
        self.target_name = target_name




        X_dfs = []
        y = []

        for bf in self.featurizers:
            bfcn = bf.__class__.__name__
            if bfcn in self.feature_classes:





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
