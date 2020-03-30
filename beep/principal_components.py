import numpy as np
import pandas as pd
import json
from monty.json import MSONable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from monty.serialization import loadfn


class PrincipalComponents(MSONable):
    """
    PCA object.

    Attributes:
        data (pandas.DataFrame): dataframe to be decomposed using PCA.
        name (str): name for PCA instance.
        n_components (int): number of principal components to use.
        explained_variance_threshold (float): desired variance to be explained.
        pca (sklearn.pca.PCA): pca object.
    """
    def __init__(self, data, name='FastCharge', n_components=15, explained_variance_threshold=0.90):
        """
        Args:
            data (pandas.DataFrame): dataframe to be decomposed using PCA.
            name (str): name for PCA instance.
            n_components (int): number of principal components to use.
            explained_variance_threshold (float): desired variance to be explained.
        """
        self.data = data
        self.name = name
        self.explained_variance_threshold = explained_variance_threshold
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.fit()
        self.get_reconstruction_errors()

    @classmethod
    def from_interpolated_data(cls, file_list_json, name='FastCharge', qty_to_pca='discharge_capacity',
                               pivot_column='voltage', cycles_to_pca=np.linspace(20, 500, 20, dtype='int')):
        """
        Method to take a list of structure jsons containing interpolated capacity vs voltage,
        create a PCA object and perform fitting.

        Args:
            file_list_json (str): json string or json filename corresponding.
            name (str):
            qty_to_pca (str): string denoting quantity to pca.
            pivot_column (str): string denoting column to pivot on. For PCA of
                Q(V), pivot_column would be voltage.
            cycles_to_pca (int): how many cycles per file to use for pca decomposition.

        Returns:
            beep.principal_components.PrincipalComponents:
        """
        return cls(pivot_data(file_list_json, qty_to_pca, pivot_column, cycles_to_pca), name)

    def as_dict(self):
        """
        Method for dictionary/json serialization.

        Returns:
            dict: object representation as dictionary.
        """
        obj = {"@module": self.__class__.__module__,
               "@class": self.__class__.__name__,
               "pca": self.pca.__dict__,
               "scaler": self.scaler.__dict__,
               "embeddings": self.embeddings,
               "reconstruction_errors": self.reconstruction_errors
               }
        return obj

    def fit(self):
        """
        Method to scale the dataframe, run PCA and evaluate embeddings.
        """
        # Center and scale training data
        scaled_data = self.scaler.fit_transform(self.data)
        self.pca.fit(scaled_data)
        # Find minimum number of components to explain threshold amount of variance in the data.
        self.min_components = np.min(
            np.where(np.cumsum(self.pca.explained_variance_ratio_) > self.explained_variance_threshold)) + 1
        # Eval embeddings of training data
        self.embeddings = self.pca.transform(scaled_data)
        self.white_embeddings = (self.embeddings - np.mean(self.embeddings, axis=0)) / np.std(self.embeddings, axis=0)
        self.reconstructions = self.scaler.inverse_transform(self.pca.inverse_transform(self.embeddings))
        return

    def get_pca_embeddings(self, data):
        """
        Method to compute PCA embeddings on new data using the trained PCA fit.

        Args:
            data (pandas.DataFrame): data frame

        Returns:
            numpy.array: transformed to embedded space, shape (n_samples, n_components)

        """
        return self.pca.transform(self.scaler.transform(data))

    def get_pca_reconstruction(self, embeddings):
        """
        Method to inverse transform PCA embeddings to reconstruct data

        Returns:
            numpy.array, shape [n_samples, n_features]. Transformed array.
        """
        return self.scaler.inverse_transform(self.pca.inverse_transform(embeddings))

    def get_pca_decomposition_outliers(self, data, upper_quantile=95, lower_quantile=5):
        """
        Outlier detection using PCA decomposition.

        Args:
            data (pandas.DataFrame): dataframe for which outlier detection needs
                to be performed
            upper_quantile (int): upper quantile for outlier detection
            lower_quantile (int): lower quantile for outlier detection

        Returns:
            numpy.array: distances to center of PCA set
            numpy.array: boolean vector of same length as data
        """
        # Compute center of the PCA training set
        center = np.median(self.white_embeddings, axis=0)
        # Define upper and lower quantiles for detecting outliers
        q_upper, q_lower = np.percentile(self.white_embeddings, [upper_quantile, lower_quantile], axis=0)
        # Transform new data
        embeddings = self.pca.transform(self.scaler.transform(data))
        # Compute centered embeddings and distances
        white_embeddings = (embeddings - np.mean(self.embeddings, axis=0)) / np.std(self.embeddings, axis=0)
        distances = np.linalg.norm(white_embeddings - center, axis=1)
        # Flag outliers even if one of the principal components falls out of inter-quantile range
        outlier_list = (white_embeddings > q_upper).any(axis=1) | (white_embeddings < q_lower).any(axis=1)
        return distances, outlier_list

    def get_reconstruction_errors(self):
        """
        Method to compute reconstruction errors of training dataset
        """
        self.reconstruction_errors = np.mean(np.abs(self.reconstructions - self.data), axis=1)
        return

    def get_reconstruction_error_outliers(self, data, threshold=1.5):
        """
        Get outliers based on PCA reconstruction errors.

        Args:
            data (pandas.DataFrame): dataframe for which outlier detection needs to be performed.
            threshold (float): threshold for outlier detection

        Returns:
            numpy.array: vector of same length as data.
        """
        embeddings = self.pca.transform(self.scaler.transform(data))
        reconstructions = self.scaler.inverse_transform(self.pca.inverse_transform(embeddings))
        reconstruction_errors = np.mean(np.abs(reconstructions - data), axis=1)
        return reconstruction_errors, reconstruction_errors > max(self.reconstruction_errors)*threshold


def pivot_data(file_list_json, qty_to_pca='discharge_capacity', pivot_column='voltage',
                               cycles_to_pca=np.linspace(10, 100, 10, dtype='int')):
    """
    Method to take a list of structure jsons, construct a dataframe to PCA using
    a pivoting column.

    Args:
        file_list_json (str): json string or json filename corresponding to a
            dictionary with a file_list and validity attribute, if this string
            ends with ".json", a json file is assumed.
        qty_to_pca (str): string denoting quantity to pca.
        pivot_column (str): string denoting column to pivot on. For PCA of Q(V),
            pivot_column would be voltage.
        cycles_to_pca (np.array): how many cycles per file to use for pca
            decomposition.

    Returns:
        pandas.DataFrame: pandas dataframe to PCA.
    """
    if file_list_json.endswith(".json"):
        file_list_data = loadfn(file_list_json)
    else:
        file_list_data = json.loads(file_list_json)
    file_list = file_list_data['file_list']
    df_to_pca = pd.DataFrame()
    for file in file_list:
        processed_run = loadfn(file)

        df = processed_run.cycles_interpolated
        df = df[df.cycle_index.isin(cycles_to_pca)]
        df_to_pca = df_to_pca.append(df.pivot(index='cycle_index', columns=pivot_column,
                                              values=qty_to_pca), ignore_index=True)
    return df_to_pca
