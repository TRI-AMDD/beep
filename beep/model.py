"""
For running ML experiments on pregenerated feature files and
reading/writing static files for battery prediction given a feature set.
"""
import copy
import pprint
from typing import List, Union, Iterable, Tuple
from math import sqrt

import numpy as np
import pandas as pd
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
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
from beep.features.base import BEEPFeatureMatrix


class BEEPMLExperimentError(BaseException):
    """Raise when error arises specific to BEEP ML Experiments"""
    pass


class BEEPLinearModelExperiment(MSONable):
    """A class for training, predicting, managing, and (de)serializing
    BEEP-based linear ML battery cycler experiments.


    1 BEEPMLExperiment = 1 model on 1 set of training data
        - Can predict on any number of prediction sets

    Args:
        feature_matrix (BEEPFeatureMatrix): The feature matrix for learning.
        target_matrix (BEEPFeatureMatrix): A matrix of targets for learning.
        targets ([str]): List of string target names. Should be found in the
            target_matrix dataframe.
        model_name (str): String specifying the linear model to use.
        alphas (str): Alpha values to try durig hyperparameter optimization.
        train_feature_drop_nan_thresh (float): Threshold 0-1 fraction of
            samples that must be present for a feature in order to avoid
            being dropped. Applies to training set only.
        train_sample_drop_nan_thresh (float): Threshold 0-1 fraction of
            features that must be present in order for a sample to avoid
            being dropped. Applies to training data only.
        predict_sample_nan_thresh (float): Threshold 0-1 fraction of
            features that must be present in order for a sample to avoid
            being dropped. Applies to prediction data only.
        drop_nan_training_targets (float): Drop nan training targets samples.
        impute_strategy (str): Define the strategy for imputing unknown
            or nan values in the feature matrix. Applies to both training
            and prediction.
        kfold (int): Number of folds k to use in running hyperparameter
            optimization with cross validation.
        max_iter (int): Number of iterations to use in determining
            optimal hyperparameters during cross validation.
        tol (float): Tolerance for CV-based hyperparameter optimization.
        l1_ratio ([float]): Ratios of L1/L2 losses to explore during hyperparameter
            optimization.
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
            alphas: Union[None, Iterable[float]] = None,
            train_feature_drop_nan_thresh: float = 0.95,
            train_sample_drop_nan_thresh: float = 0.50,
            predict_sample_nan_thresh: float = 0.75,
            drop_nan_training_targets: bool = False,
            impute_strategy: str = "median",
            kfold: int = 5,
            max_iter: int = 1e6,
            tol: float = 1e-4,
            # only relevant for elasticnet
            l1_ratio: Union[Tuple[float], List[float]] = (
            0.001, 0.1, 0.5, 0.7, 0.9, 0.95, 1),
            homogenize_features: bool = True
    ):
        if model_name not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported by {self.__class__.__name__}")

        if len(targets) < 1:
            raise ValueError("At least one target must be specified")

        self.feature_matrix = feature_matrix
        self.target_matrix = target_matrix

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
                f"features ({X.shape[1]}); may cause overfitting."
            )

        # Form the clean feature matrix
        X = X.dropna(axis=1, thresh=train_sample_drop_nan_thresh * X.shape[0])
        X = X.dropna(axis=0, thresh=train_sample_drop_nan_thresh * X.shape[1])
        X = self._impute_df(X, method=impute_strategy)
        self.impute_strategy = impute_strategy
        if X.shape[0] < 2 or X.shape[1] < 1:
            raise BEEPMLExperimentError(
                "Cleaned feature matrix has dimensions of less "
                "than 1 feature or less than 2 samples. Try adjusting "
                "the thresholds for cleaning or examine your feature "
                "matrix."
            )

        # Form the clean target matrix
        missing_targets = [t for t in targets if t not in y.columns]
        if missing_targets:
            raise BEEPMLExperimentError(
                f"Required target columns missing from "
                f"target matrix: {missing_targets}"
            )
        y = y[targets].loc[X.index]
        if y.isna().any().any():
            if drop_nan_training_targets:
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
        self.drop_nan_training_targets = drop_nan_training_targets

        # todo: this is only to help with deserialization, this could cause
        # todo: contamination in judging test scores when used with
        # todo: train_and_score()
        self.scaler = StandardScaler().fit(X)
        self.kfold = kfold
        self.alphas = alphas
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio

        self.optimal_hyperparameters = None
        self.homogenize_features = homogenize_features

    def train(self, X: pd.DataFrame = None, y: pd.DataFrame = None):
        """Train on 100% of available data.

        Args:
            X (pd.Dataframe): Clean and homogenized learning features.
                If not specified, df defined in __init__ (all training
                data) is used.
            y (pd.DataFrame): Clean and homogenized targets. If not
                specified, df defined in __init__ (all training data)
                is used.

        Returns:
            model (BaseEstimator): The sklearn model, fit on training data.
            training_errors (dict): Training errors based on multiple metrics.

        """
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if not self.multi:
            y = y[self.targets[0]]

        X = self.scaler.fit_transform(X)

        logger.info(
            f"Training on {X.shape[0]} samples with {X.shape[1]} features "
            f"predicting {y.shape[0]}"
        )

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
            kwargs.pop("max_iter")
            kwargs.pop("tol")

            # Ridge has to have alphas set by hand as it has no
            # default alphas
            if not kwargs["alphas"]:
                kwargs["alphas"] = (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)
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

        y_training = model.predict(X)
        y_training = pd.DataFrame(data=y_training, columns=self.targets)
        training_errors = self._score_arrays(y, y_training)
        return model, training_errors

    def predict(
            self,
            feature_matrix: Union[BEEPFeatureMatrix, pd.DataFrame],
            homogenize_features: Union[None, bool] = None,
    ):
        """Use the trained model to predict new degradation characteristics
        based on an incoming feature matrix.


        Args:
            feature_matrix (BEEPFeatureMatrix): The feature matrix to use
                for predicting degradation character.
            homogenize_features (bool, None): Whether to homogenize the
                incoming matrix's features. Overrides homogenize_features
                as set in __init__.

        Returns:
            y_pred (pd.DataFrame): The predictions, in dataframe format.
            dropped (list): List of dropped samples, by incoming df
                index (e.g., filename).

        """
        if not self.model:
            raise BEEPMLExperimentError("No model has been trained.")

        # condense features down to those required, throwing error if not present

        if isinstance(feature_matrix, BEEPFeatureMatrix):
            X = feature_matrix.matrix
        else:
            X = feature_matrix

        # make sure features will have the same names if homogenize features
        # even if featurizer' hyperparameters are different
        homogenize_features = self.homogenize_features if homogenize_features is None else homogenize_features
        if homogenize_features:
            X = self._remove_param_hash_from_features(X)

        missing_features = [f for f in self.feature_labels if
                            f not in X.columns]
        extra_features = [f for f in X.columns if f not in self.feature_labels]
        if missing_features:
            raise BEEPMLExperimentError(
                f"{len(missing_features)} features present in training set not present "
                f"in prediction: "
                f"\n{pprint.pformat(missing_features)}"
            )
        if extra_features:
            logger.warning(
                f"{len(extra_features)} extra features not in training set present in "
                f"prediction set due to fitting with nan threshold ({self.train_feature_drop_thresh}) - "
                f"these will be dropped: \n{pprint.pformat(extra_features)}"
            )

        # Assemble the correct data while retaining all features
        X_old = copy.deepcopy(X)
        X = X[self.feature_labels].dropna(
            axis=0,
            thresh=self.predict_sample_nan_thresh * X.shape[1]
        )
        X = self._impute_df(X, self.impute_strategy)

        dropped = []
        if X_old.shape[0] != X.shape[0]:
            dropped = [s for s in X_old.index if s not in X]
            logger.warning(
                f"{len(dropped)} samples dropped due to nan sample threshold "
                f"of {self.predict_sample_nan_thresh}. List of those dropped "
                f"indices is returned by .predict()."
            )

        X_indices = X.index
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)

        # y_pred is an array, so we reattach the same indices
        # e.g., if idx contains filenames
        # which is important in case samples were dropped
        y_pred = pd.DataFrame(data=y_pred, columns=self.targets, index=X_indices)
        return y_pred, dropped

    def train_and_score(self, train_and_val_frac=0.8):
        """Train and adjust hyperparameters on a subset of data, then predict
        on a test set and obtain scores automatically.

        Args:
            train_and_val_frac (float): The fraction to train and validate
                on during hyperparameter optimization. 1 minus this fraction
                is the size of the test data that will be scored and returned.

        Returns:
            model (BaseEstimator): The sklearn model, trained on train_and_val_frac
                of the available data.
            training_errors (dict): Metrics of training error found during
                hyperparameter optimization and fitting.
            test_errors (dict): Metrics of test errors found after training by
                predicting on the test set of the data.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=train_and_val_frac)
        model, training_errors = self.train(X=X_train, y=y_train)

        # Features here are already homogenized
        y_pred, dropped = self.predict(X_test, homogenize_features=False)
        test_errors = self._score_arrays(y_test, y_pred)
        return model, training_errors, test_errors

    def _score_arrays(
            self,
            y: Union[pd.DataFrame, pd.Series],
            y_pred: Union[pd.DataFrame, pd.Series]
    ) -> dict:
        """Take two numerical arrays of equal size, return various error metrics
        about them.

        Works with multiple targets across all metrics.

        Args:
            y (pd.DataFrame, pd.Series): True values
            y_pred (pd.DataFrame, pd.Series): Predicted values

        Returns:

            errors (dict): Error metrics comparing y and y_pred.
        """
        errors = {}
        for metric, f in self.ERROR_METRICS.items():
            if self.multi:
                errors_per_metric = {}
                for target in self.targets:
                    errors_per_metric[target] = f((y[target], y_pred[target]))
                errors[metric] = errors_per_metric
            else:
                errors[metric] = f((y, y_pred))
        return errors

    @staticmethod
    def _remove_param_hash_from_features(X):
        """
        Remove a parameter hash (identifying cases where the same featurizer
        is applied with different parameters) from all features in a dataframe.


        Args:
            X (pd.Dataframe): The dataframe to remove parameter hashes from.

        Returns:
            X (pd.Dataframe): The dataframe with columns stripped of parameter hashes.

        """
        d = BEEPFeatureMatrix.OP_DELIMITER
        cols_stripped = [d.join(c.split(d)[:-1]) for c in X.columns]
        X.columns = cols_stripped
        return X

    @staticmethod
    def _impute_df(df, method="median"):
        """
        Impute a dataframe using a specified method.

        Args:
            df (pd.Dataframe): A dataframe, with zero or more nan values.
            method (str): One of "median", "mean", or "none".

        Returns:
            df (pd.Dataframe): The imputed dataframe.

        """
        if method == "median":
            return df.apply(lambda x: x.fillna(x.median()), axis=0)
        elif method == "mean":
            return df.apply(lambda x: x.fillna(x.mean()), axis=0)
        elif method == 'none':
            return df
        else:
            raise ValueError(f"impute_strategy {method} unsupported!")

    def as_dict(self):
        """Serialize a BEEPDatapath as a dictionary.

        Must not be loaded from legacy.

        Returns:
            (dict): corresponding to dictionary for serialization.

        """
        if not self.model:
            raise ValueError("Model must be fit before serializing.")

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,

            # To be passed as args in from_dict
            "feature_matrix": self.feature_matrix.as_dict(),
            "target_matrix": self.target_matrix.as_dict(),
            "targets": self.targets,
            "model_name": self.model_name,
            "alphas": self.alphas,
            "train_feature_drop_nan_thresh": self.train_feature_drop_thresh,
            "train_sample_drop_nan_thresh": self.train_sample_drop_thresh,
            "predict_sample_nan_thresh": self.predict_sample_nan_thresh,
            "impute_strategy": self.impute_strategy,
            "drop_nan_training_targets": self.drop_nan_training_targets,
            "kfold": self.kfold,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "l1_ratio": self.l1_ratio,
            "homogenize_features": self.homogenize_features,

            # Serialize the model and scaler themselves
            # requires some sklearn hackery
            "model_sklearn": {
                "coef_": self.model.coef_.tolist(),
                "intercept_": self.model.intercept_.tolist(),
                "optimal_hyperparameters": self.optimal_hyperparameters,
            },

        }

    @classmethod
    def from_dict(cls, d):
        """Create a BEEPDatapath object from a dictionary.

        Args:
            d (dict): dictionary represenation.

        Returns:
            beep.structure.ProcessedCyclerRun: deserialized ProcessedCyclerRun.
        """

        feature_matrix = BEEPFeatureMatrix.from_dict(d["feature_matrix"])
        target_matrix = BEEPFeatureMatrix.from_dict(d["target_matrix"])

        o = cls(
            feature_matrix=feature_matrix,
            target_matrix=target_matrix,
            targets=d["targets"],
            model_name=d["model_name"],
            alphas=d["alphas"],
            train_feature_drop_nan_thresh=d["train_feature_drop_nan_thresh"],
            train_sample_drop_nan_thresh=d["train_sample_drop_nan_thresh"],
            predict_sample_nan_thresh=d["predict_sample_nan_thresh"],
            impute_strategy=d["impute_strategy"],
            drop_nan_training_targets=d["drop_nan_training_targets"],
            kfold=d["kfold"],
            max_iter=d["max_iter"],
            tol=d["tol"],
            l1_ratio=d["l1_ratio"],
            homogenize_features=d["homogenize_features"]
        )

        # Hack sklearn a little bit to serialize these models to json
        modelcls = {
            "elasticnet": ElasticNet,
            "lasso": Lasso,
            "ridge": Ridge
        }[d["model_name"]]

        model_params = d["model_sklearn"]
        model = modelcls(**model_params["optimal_hyperparameters"])
        model.coef_ = np.asarray(model_params["coef_"])
        model.intercept_ = np.asarray(model_params["intercept_"])
        o.model = model
        o.optimal_hyperparameters = model_params["optimal_hyperparameters"]
        return o

    @classmethod
    def from_json_file(cls, filename):
        """Load a BEEPLinearModelExperiment from file.

        Args:
            filename (str): The filename to load. Should be json.

        Returns:
            (BEEPLinearModelExperiment)
        """
        return loadfn(filename)

    def to_json_file(self, filename):
        """Serialize a BEEPLinearModelExperiment to file.

        Args:
            filename (str): The filename to write. Should be json.

        Returns:
            None
        """
        d = self.as_dict()
        dumpfn(d, filename)
