"""
For splitting a BEEPFeatureMatrix into train/test sets in a format usable by a BEEPPredictionModel object
"""
from beep import logger
from beep.features.base import BEEPFeatureMatrix
from sklearn.model_selection import KFold
from typing import List
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
import pandas as pd
from functools import reduce


class BEEPDataSplitterError(BaseException):
    pass


class BEEPDataset(MSONable):
    """
    A wrapper class for a single train/test set + metadata

    Args:
    train_X (pd.DataFrame): The feature dataframe for training
    train_y (pd.DataFrame): The target dataframe for training
    test_X (pd.DataFrame): The feature dataframe for testing
    test_y (pd.DataFrame): The target dataframe for testing

    """

    def __init__(
        self,
        train_X,
        train_y,
        test_X,
        test_y
    ):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "train_X": self.train_X.to_dict(),
            "train_y": self.train_y.to_dict(),
            "test_X": self.test_X.to_dict(),
            "test_y": self.test_y.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        dataset = cls(pd.DataFrame.from_dict(d["train_X"]), pd.DataFrame.from_dict(
            d["train_y"]), pd.DataFrame.from_dict(d["test_X"]), pd.DataFrame.from_dict(d["test_y"]))

        return dataset


class BEEPDataSplitter(MSONable):
    """
    A class for splitting a BEEPFeatureMatrix into train/test sets for use by ML models

        Args:
        feature_matrix (BEEPFeatureMatrix): The feature matrix for learning.
        features ([str]): List of string feature names. Should be found in the
            feature_matrix dataframe
        targets ([str]): List of string target names. Should be found in the
            feature_matrix dataframe.
        train_feature_drop_nan_thresh (float): Threshold 0-1 fraction of
            samples that must be present for a feature in order to avoid
            being dropped. Applies to training set only.
        train_sample_drop_nan_thresh (float): Threshold 0-1 fraction of
            features that must be present in order for a sample to avoid
            being dropped. Applies to training data only.
        drop_nan_training_targets (float): Drop nan training targets samples.
        impute_strategy (str): Define the strategy for imputing unknown
            or nan values in the feature matrix. Applies to both training
            and prediction.
        n_splits (int): Number of folds to split data into
        homogenize_features (bool): Allow features generated with mismatching
            hyperparameters to be coalesced into the same "feature". I.e.,
            features generated with SomeFeaturizer(param=1) to be used as
            features generated with SomeFeaturizer(param=2).
        split_columns ([str]): List of string names of features by which to
            split into train/test data (e.g. charging protocol)
        exclusion_columns ([str]): List of string names of bool features
            (from e.g. exclusionCriteria featurizer) by which to exclude columns from analysis
        drop_split_threshold (float): Only used if exclusion_columns and split_columns are both
            set. Threshold of samples from a given split that must be included to avoid dropping
            the entire split
    """

    def __init__(
            self,
            feature_matrix: BEEPFeatureMatrix,
            features: List[str],
            targets: List[str],
            train_feature_drop_nan_thresh: float = 0.75,
            train_sample_drop_nan_thresh: float = 0.50,
            drop_nan_training_targets: bool = True,
            impute_strategy: str = "median",
            n_splits: int = 5,
            homogenize_features: bool = True,
            random_state: int = 10,
            split_columns: List[str] = None,
            exclusion_columns: List[str] = None,
            drop_split_threshold: float = 0.5,
    ):

        self.feature_matrix = feature_matrix

        if homogenize_features:
            self.feature_matrix.matrix = self._remove_param_hash_from_features(self.feature_matrix.matrix)

        # Form the clean feature and target matrices
        missing_columns = [t for t in targets+features if t not in self.feature_matrix.matrix.columns]

        if split_columns is not None:
            missing_columns += [t for t in split_columns if t not in self.feature_matrix.matrix.columns] 
        if exclusion_columns is not None:
            missing_columns += [t for t in exclusion_columns if t not in self.feature_matrix.matrix.columns] 

        if missing_columns:
            raise BEEPDataSplitterError(
                f"Required columns missing from "
                f"feature matrix: {missing_columns}"
            )

        retain_columns = features + (split_columns if split_columns is not None else []) + \
            (exclusion_columns if exclusion_columns is not None else []) 
        X = self.feature_matrix.matrix[retain_columns]
        y = self.feature_matrix.matrix[targets]

        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        # Form the clean feature matrix
        X = X.dropna(axis=1, thresh=train_feature_drop_nan_thresh * X.shape[0])
        X = X.dropna(axis=0, thresh=train_sample_drop_nan_thresh * X.shape[1])

        if exclusion_columns is not None:
            X[exclusion_columns] = X[exclusion_columns].fillna(value=False, axis='columns')

        X = self._impute_df(X, method=impute_strategy)

        self.impute_strategy = impute_strategy

        # Create an aggregate column to group splits on by concatenating split column values
        if split_columns is not None:
            X["grouping_column"] = X.apply(lambda x: "::".join([str(x[s]) for s in split_columns]), axis=1)
            unique_grouping_values = X["grouping_column"].unique()

        if exclusion_columns is not None:

            if len(exclusion_columns) > 1:
                is_included_condition = reduce(lambda c1, c2: c1 & c2, [
                                               X[e] for e in exclusion_columns[1:]], X[exclusion_columns[0]])
            else:
                is_included_condition = X[exclusion_columns[0]]

            X_incl = X[is_included_condition]
            # Check if any entire split should be excluded
            if split_columns is not None:
                exclude_groups = []
                for group in unique_grouping_values:
                    X_group = X[X["grouping_column"] == group]
                    X_incl_group = X_incl[X_incl["grouping_column"] == group]

                    if len(X_incl_group)/len(X_group) < drop_split_threshold:
                        exclude_groups.append(group)

                self.exclude_groups = exclude_groups
                X_incl = X_incl[~X_incl["grouping_column"].isin(exclude_groups)]

            X = X_incl

        if X.shape[0] < X.shape[1]:
            logger.warning(
                f"Number of samples ({X.shape[0]}) less than number of "
                f"features ({X.shape[1]}); may cause overfitting."
            )

        if X.shape[0] < 2 or X.shape[1] < 1:
            raise BEEPDataSplitterError(
                "Cleaned feature matrix has dimensions of less "
                "than 1 feature or less than 2 samples. Try adjusting "
                "the thresholds for cleaning or examine your feature "
                "matrix."
            )

        y = y.loc[X.index]
        if y.isna().any().any():
            if drop_nan_training_targets:
                y = y.dropna(axis=0)
            else:
                raise BEEPDataSplitterError(
                    "Target matrix contains nans and drop_nan_targets is "
                    "set to False."
                )
        if y.shape[0] < 2:
            raise BEEPDataSplitterError(
                "Target matrix after dropping nans is less than 2 samples."
            )

        # Ensure there will be an equal number of X samples
        # and y samples
        self.X = X.loc[y.index]
        self.y = y

        self.feature_labels = [c for c in self.X.columns if c in features]

        self.targets = targets

        self.multi = len(self.targets) > 1

        self.train_feature_drop_nan_thresh = train_feature_drop_nan_thresh
        self.train_sample_drop_nan_thresh = train_sample_drop_nan_thresh
        self.drop_nan_training_targets = drop_nan_training_targets
        self.homogenize_features = homogenize_features
        self.n_splits = n_splits
        self.random_state = random_state
        self.split_columns = split_columns
        self.datasets = None

    def split(self):
        """
        Split the data into a list of BEEPDataset objects
        """
        self.datasets = []

        if self.split_columns is None:
            self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            for train_indices, test_indices in self.kfold.split(self.X): 

                train_X = self.X[self.feature_labels].iloc[train_indices]
                train_y = self.y.iloc[train_indices]

                test_X = self.X[self.feature_labels].iloc[test_indices]
                test_y = self.y.iloc[test_indices]

                dataset = BEEPDataset(
                    train_X,
                    train_y,
                    test_X,
                    test_y
                    )

                self.datasets.append(dataset)
        else:

            self.kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            unique_grouping_values = self.X["grouping_column"].unique()

            for train_indices, test_indices in self.kfold.split(unique_grouping_values):

                train_grouping_labels = unique_grouping_values[train_indices]
                test_grouping_labels = unique_grouping_values[test_indices]

                train_X = self.X[self.X["grouping_column"].isin(train_grouping_labels)][self.feature_labels]
                train_y = self.y.loc[train_X.index]

                test_X = self.X[self.X["grouping_column"].isin(test_grouping_labels)][self.feature_labels]
                test_y = self.y.loc[test_X.index]

                dataset = BEEPDataset(
                    train_X,
                    train_y,
                    test_X,
                    test_y
                    )

                self.datasets.append(dataset)
        return self.datasets

    @staticmethod
    def _remove_param_hash_from_features(X):
        """
        Remove a parameter hash (identifying cases where the same featurizer
        is applied with different parameters) from all features in a dataframe.


        Args:
            X (pd.DataFrame): The dataframe to remove parameter hashes from.

        Returns:
            X (pd.DataFrame): The dataframe with columns stripped of parameter hashes.

        """
        d = BEEPFeatureMatrix.OP_DELIMITER
        cols_stripped = [d.join(c.split(d)[:2]) for c in X.columns]
        X.columns = cols_stripped
        return X

    @staticmethod
    def _impute_df(df, method="median"):
        """
        Impute a dataframe using a specified method.

        Args:
            df (pd.DataFrame): A dataframe, with zero or more nan values.
            method (str): One of "median", "mean", or "none".

        Returns:
            df (pd.DataFrame): The imputed dataframe.

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
        if not self.datasets:
            self.split()

        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "feature_matrix": self.feature_matrix.as_dict(),
            "features": self.feature_labels,
            "targets": self.targets,
            "train_feature_drop_nan_thresh": self.train_feature_drop_nan_thresh,
            "train_sample_drop_nan_thresh": self.train_sample_drop_nan_thresh,
            "drop_nan_training_targets": self.drop_nan_training_targets,
            "homogenize_features": self.homogenize_features,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "split_columns": self.split_columns,
            "datasets": [d.as_dict() for d in self.datasets]
        }

    @classmethod
    def from_dict(cls, d):
        feature_matrix = BEEPFeatureMatrix.from_dict(d['feature_matrix'])

        bds = cls(
            feature_matrix=feature_matrix,
            features=d["features"],
            targets=d["targets"],
            train_feature_drop_nan_thresh=d["train_feature_drop_nan_thresh"],
            train_sample_drop_nan_thresh=d["train_sample_drop_nan_thresh"],
            drop_nan_training_targets=d["drop_nan_training_targets"],
            homogenize_features=d["homogenize_features"],
            n_splits=d["n_splits"],
            random_state=d["random_state"],
            split_columns=d["split_columns"]
        )

        bds.datasets = [BEEPDataset.from_dict(dataset) for dataset in d["datasets"]]
        return bds

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
