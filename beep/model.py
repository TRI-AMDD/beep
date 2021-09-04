"""
For assembling datasets, running ML experiments, and reading/writing static
files for battery prediction given a feature set.
"""
import json
import copy
import hashlib
import os
from typing import List, Union, Iterable, Tuple
from functools import reduce
from math import sqrt

import numpy as np
import pandas as pd
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
from monty.io import zopen
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    RidgeCV,
    Ridge,
    ElasticNetCV,
    ElasticNet,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    max_error,
    r2_score
)

from beep import logger
from beep.features.base import BEEPFeaturizer, BEEPFeatureMatrix


class BEEPMLExperimentError(BaseException):
    """Raise when error arises specific to BEEP ML Experiments"""
    pass


class BEEPLinearModelExperiment(MSONable):
    """
    A class for training, predicting, managing, and (de)serializing
    BEEP-based linear ML battery cycler experiments.


    1 BEEPMLExperiment = 1 model on 1 set of training data
        - Can predict on any number of prediction sets

    Args:


        homogenize_features (bool): Allow features generated with mismatching
            hyperparameters to be coalesced into the same "feature". I.e.,
            features generated with SomeFeaturizer(param=1) to be used as
            features generated with SomeFeaturizer(param=2).

    """

    ALLOWED_MODELS = ("elasticnet", "ridge", "lasso")
    ERROR_METRICS = {
        "rmse": lambda x: sqrt(mean_squared_error(*x)),
        "mae": lambda x: mean_absolute_error(*x),
        "r2": lambda x: r2_score(*x),
        "max_error": lambda x: max_error(*x)
    }

    def __init__(
            self,
            feature_matrix: BEEPFeatureMatrix,
            target_matrix: BEEPFeatureMatrix,
            targets: List[str],
            model_name: str,
            alphas: Union[None, Iterable[float]],
            train_feature_drop_nan_thresh: float = 0.95,
            train_sample_drop_nan_thresh: float = 0.50,
            predict_sample_nan_thresh: float = 0.75,
            impute_strategy: str = "median",
            drop_nan_targets: bool = False,
            kfold: int = 5,
            max_iter: int = 1e6,
            tol: float = 1e-4,
            # only relevant for elasticnet
            l1_ratio: Union[Tuple[float], List[float]] = (
            0.1, 0.5, 0.7, 0.9, 0.95, 1),
            homogenize_features: bool = True
    ):

        if model_name not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported by {self.__class__.__name__}")

        if len(targets) < 1:
            raise ValueError(f"At least one target must be specified")

        self.feature_matrix = feature_matrix
        self.target_matrix = target_matrix

        missing_targets = \
            [t for t in targets if t not in self.target_matrix.matrix.columns]

        if missing_targets:
            raise BEEPMLExperimentError(
                f"Required target columns missing from "
                f"target matrix: {missing_targets}"
            )

        X = self.feature_matrix.matrix.replace([np.inf, -np.inf], np.nan)
        y = self.target_matrix.matrix.replace([np.inf, -np.inf], np.nan)

        if homogenize_features:
            X = self._remove_param_hash_from_features(X)
            y = self._remove_param_hash_from_features(y)

        if X.shape[0] != y.shape[0]:
            raise BEEPMLExperimentError(
                "Can't run experiment on unequal numbers of input samples."
            )

        if X.shape[0] < X.shape[1]:
            logger.warning(
                f"Number of samples ({X.shape[0]}) less than number of "
                f"features ({X.shape[1]}; may cause overfitting."
            )

        # Form the clean feature matrix
        X = X.dropna(axis=1, thresh=train_feature_drop_nan_thresh)
        X = X.dropna(axis=0, thresh=train_sample_drop_nan_thresh)

        X = self._impute_df(X, method=impute_strategy)
        self.impute_strategy = impute_strategy

        if X.shape[0] < 2 or X.shape[1] < 1:
            raise BEEPMLExperimentError(
                f"Cleaned feature matrix has dimensions of less "
                f"than 1 feature or less than 2 samples. Try adjusting "
                f"the thresholds for cleaning or examine your feature "
                f"matrix."
            )

        # Form the clean target matrix
        y = y[targets].loc[X.index]
        if y.isna().any():
            if drop_nan_targets:
                y = y.dropna(axis=0)
            else:
                raise BEEPMLExperimentError(
                    "Target matrix contains nans and drop_nan_targets is "
                    "set to False."
                )

        if y.shape[0] < 2:
            raise BEEPMLExperimentError(
                "Target matrix after dropping nans is less than 2 samples."
            )

        # Ensure there will be an equal number of X samples
        # and y samples
        self.X = X.loc[y.index]
        self.y = y

        # These features must be present in passed dfs for predictions to work
        self.feature_labels = self.X.columns.tolist()

        self.targets = targets

        self.multi = len(self.targets) > 1

        if self.multi and model_name != "elasticnet":
            raise BEEPMLExperimentError(
                f"Model {model_name} not supported for multiple target "
                f"regression."
            )

        self.model_name = model_name if model_name else "elasticnet"
        self.model = None

        self.train_feature_drop_thresh = train_feature_drop_nan_thresh
        self.train_sample_drop_thresh = train_sample_drop_nan_thresh
        self.predict_sample_nan_thresh = predict_sample_nan_thresh

        self.scaler = StandardScaler()
        self.kfold = kfold
        self.alphas = alphas
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio

        self.optimal_hyperparameters = None

        self.homogenize_features = homogenize_features

    def train(self, X: pd.DataFrame = None, y: pd.DataFrame = None):
        """
        Train on 100% of available data.

            Args:
                X (pd.Dataframe): Clean and homogenized learning features.
                    If not specified, df defined in __init__ (all training
                    data) is used.
                y (pd.DataFrame): Clean and homogenized targets. If not
                    specified, df defined in __init__ (all training data)
                    is used.

        Returns:
            None
        """
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if not self.multi:
            y = y[self.targets[0]]

        X = self.scaler.fit_transform(X)

        logger.info(f"Training on {X.shape[0]} samples with {X.shape[1]} features")

        kwargs = {
            "fit_intercept": True,
            "alphas": self.alphas,
            "cv": self.kfold,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "l1_ratio": self.l1_ratio
        }

        if self.model_name == "elasticnet":
            if self.multi:
                cv_class = MultiTaskElasticNetCV
                model_class = MultiTaskElasticNet
            else:
                cv_class = ElasticNetCV
                model_class = ElasticNet
        elif self.model_name == "lasso":
            cv_class = LassoCV
            model_class = Lasso
            kwargs.pop("l1_ratio")
        elif self.model_name == "ridge":
            cv_class = RidgeCV
            model_class = Ridge
            kwargs.pop("l1_ratio")
        else:
            raise NotImplementedError(f"Unsupported model '{self.model_name}'")

        # Search for optimal hyperparameters
        cv = cv_class(**kwargs)
        cv.fit(X, y)

        # Set optimal hyperparameters and refit
        optimal_hyperparameters = {"alpha": cv.alpha_}
        if self.model_name == "elasticnet":
            optimal_hyperparameters["l1_ratio"] = cv.l1_ratio_

        model_kwargs = {
            "fit_intercept": True,
            "normalize": False,
            "max_iter": self.max_iter,
        }
        model_kwargs.update(optimal_hyperparameters)
        self.optimal_hyperparameters = optimal_hyperparameters

        model = model_class(**model_kwargs)
        model.fit(X, y)
        self.model = model

        logger.info(f"Trained {model.__class__.__name__} and found hyperparameters {optimal_hyperparameters}")

        y_training = model.predict(X)
        training_errors = self._score_arrays(y, y_training)
        return model, training_errors

    def predict(self, feature_matrix: BEEPFeatureMatrix):
        # condense features down to those required, throwing error if not present

        X = feature_matrix.matrix

        # make sure features will have the same names if homogenize features
        # even if featurizer' hyperparameters are different
        if self.homogenize_features:
            X = self._remove_param_hash_from_features(X)

        missing_features = [f for f in self.feature_labels if
                            f not in X.columns]
        extra_features = [f for f in X.columns if f not in self.feature_labels]
        if missing_features:
            raise BEEPMLExperimentError(
                f"Features present in training set not present in prediction set:"
                f"\n{missing_features}"
            )

        if extra_features:
            logger.warning(
                "Extra features not in training set present in prediction set - "
                f"these will be dropped: \n{extra_features}"
            )

        # Assemble the correct data while retaining all features
        X_old = copy.deepcopy(X)
        X = X[self.feature_labels].dropna(axis=0, thresh=self.predict_sample_nan_thresh)
        X = self._impute_df(X, self.impute_strategy)
        X = self.scaler.transform(X)

        dropped = []
        if X_old.shape[0] != X.shape[0]:
            dropped = [s for s in X_old.index if s not in X]
            logger.warning(
                f"{len(dropped)} samples dropped due to nan sample threshold "
                f"of {self.predict_sample_nan_thresh}. List of those dropped "
                f"indices is returned by .predict()."
            )

        if not self.model:
            raise BEEPMLExperimentError("No model has been trained.")

        y_pred = self.model.predict(X)

        #y_pred is an array, so we reattach the same indices
        # e.g., if idx contains filenames
        # which is important in case samples were dropped
        y_pred = pd.DataFrame(data=y_pred, columns=self.targets, index=X.index)
        return y_pred, dropped

    def train_and_score(self, train_and_val_frac=0.8):
        """Train and adjust hyperparameters on a subset of data, then predict
        on a test set and obtain scores automatically.

        Args:
            train_and_val_frac (float): None

        Returns:

        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_and_val_frac)
        model, training_errors = self.train(X=X_train, y=y_train)
        y_pred, dropped = self.predict(X_test)
        test_errors = self._score_arrays(y_test, y_pred)
        return model, training_errors, test_errors

    def _score_arrays(self, y: Union[pd.DataFrame, pd.Series], y_pred: Union[pd.DataFrame, pd.Series]) -> dict:
        errors = {}
        for metric, f in self.ERROR_METRICS:
            if self.multi:
                errors = {}
                for target in self.targets:
                    errors[target] = f(y[target], y_pred[target])
                errors[metric] = errors
            else:
                errors[metric] = f(y, y_pred)
        return errors

    @staticmethod
    def _remove_param_hash_from_features(X):
        d = BEEPFeatureMatrix.OP_DELIMITER
        cols_stripped = [d.join(c.split(d)[:-1]) for c in X.columns]
        X.columns = cols_stripped
        return cols_stripped

    @staticmethod
    def _impute_df(df, method="median"):
        if method == "median":
            df = df.apply(lambda x: x.fillna(x.median()), axis=0)
        elif method == "mean":
            df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
        else:
            raise ValueError(f"impute_strategy {method} unsupported!")
        return df


    # todo: save mu and std

    # def as_dict(self):
    #     """Serialize a BEEPDatapath as a dictionary.
    #
    #     Must not be loaded from legacy.
    #
    #     Returns:
    #         (dict): corresponding to dictionary for serialization.
    #
    #     """
    #
    #     return {
    #         "@module": self.__class__.__module__,
    #         "@class": self.__class__.__name__,
    #
    #         # Core parts of BEEPFeaturizer
    #         "featurizers": [f.as_dict() for f in self.featurizers],
    #         "matrix": self.matrix.to_dict("list"),
    #     }
    #
    # @classmethod
    # def from_dict(cls, d):
    #     """Create a BEEPDatapath object from a dictionary.
    #
    #     Args:
    #         d (dict): dictionary represenation.
    #
    #     Returns:
    #         beep.structure.ProcessedCyclerRun: deserialized ProcessedCyclerRun.
    #     """
    #
    #     # no need for original datapath
    #     featurizers = [BEEPFeaturizer.from_dict(f) for f in d["featurizers"]]
    #     return cls(featurizers)
    #
    # @classmethod
    # def from_json_file(cls, filename):
    #     """Load a structured run previously saved to file.
    #
    #     .json.gz files are supported.
    #
    #     Loads a BEEPFeatureMatrix from json.
    #
    #     Can be used in combination with files serialized with BEEPFeatures.to_json_file.
    #
    #     Args:
    #         filename (str, Pathlike): a json file from a structured run, serialzed with to_json_file.
    #
    #     Returns:
    #         None
    #     """
    #     return loadfn(d)
    #
    # def to_json_file(self, filename):
    #     """Save a BEEPFeatureMatrix to disk as a json.
    #
    #     .json.gz files are supported.
    #
    #     Not named from_json to avoid conflict with MSONable.from_json(*)
    #
    #     Args:
    #         filename (str, Pathlike): The filename to save the file to.
    #         omit_raw (bool): If True, saves only structured (NOT RAW) data.
    #             More efficient for saving/writing to disk.
    #
    #     Returns:
    #         None
    #     """
    #     d = self.as_dict()
    #     dumpfn(d, filename)


if __name__ == "__main__":
    # bfm = BEEPFeatureMatrix.from_json_file(
    #     "/Users/ardunn/alex/tri/code/beep/beep/CLI_TEST_FILES_FEATURIZATION/FeatureMatrix-2021-02-09_21.07.50.514178.json.gz")

    features_file = "/Users/ardunn/alex/tri/code/beep/beep/CLI_TEST_FILES_FEATURIZATION/features.json.gz"
    targets_file = "/Users/ardunn/alex/tri/code/beep/beep/CLI_TEST_FILES_FEATURIZATION/targets.json.gz"

    feats = BEEPFeatureMatrix.from_json_file(features_file)
    targets = BEEPFeatureMatrix.from_json_file(targets_file)


    print(targets.matrix)


    # beep_model = BEEPLinearModelExperiment(feats, targets, )
