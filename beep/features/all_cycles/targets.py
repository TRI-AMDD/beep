import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers

from beep.features.featurizer import BEEPFeaturizer, BEEPFeaturizationError


class TrajectoryFastCharge(BEEPFeaturizer):
    """
    Object corresponding to cycle numbers at which the capacity drops below
     specific percentages of the initial capacity. Computed on the discharge
     portion of the regular fast charge cycles.

    """

    DEFAULT_HYPERPARAMETERS = {
        "thresh_max_cap": 0.98,
        "thresh_min_cap": 0.78,
        "interval_cap": 0.03
    }

    def validate(self):
        """
        This function determines if the input data has the necessary attributes for
        creation of this feature class. It should test for all of the possible reasons
        that feature generation would fail for this particular input data.

        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
            gets featurized. These could be filters for column or row operations
        Returns:
            bool: True/False indication of ability to proceed with feature generation
        """
        cap = self.datapath.structured_summary.discharge_capacity
        cap_ratio = cap.min() / cap.max()
        max_cap = self.hyperparameters["thresh_max_cap"]
        if not cap_ratio < max_cap:
            return False, f"thresh_max_cap hyperparameter exceeded: {cap_ratio} !< {max_cap}"
        else:
            return True, None

    def create_features(self):
        """
        Calculate the outcomes from the input data. In particular, the number of cycles
        where we expect to reach certain thresholds of capacity loss
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
            gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            pd.DataFrame: cycles at which capacity/energy degradation exceeds thresholds
        """
        y = self.datapath.capacities_to_cycles(
            self.hyperparameters["thresh_max_cap"],
            self.hyperparameters["thresh_min_cap"],
            self.hyperparameters["interval_cap"],
        )
        self.features = y


class DiagnosticProperties(BEEPFeaturizer):
    """
    This class stores fractional levels of degradation in discharge capacity and discharge energy
    relative to the first cycle at each diagnostic cycle, grouped by diagnostic cycle type.

        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features

    Hyperparameters:
        parameters_dir (str): Full path to directory of parameters to analyse the
            diagnostic cycles
        quantities ([str]): Quantities to extract/get fractional metrics for
            diagnostic cycles
        cycle_type (str): Type of diagnostic cycle being used to measure the
            fractional metric
        metric (str): The metric being used for fractional capacity
        interpolation_axes (list): List of column names to use for
            x_axis interpolation (distance to threshold)
        threshold (float): Value for the fractional metric to be considered above
            or below threshold
        filter_kinks (float): If set, cutoff value for the second derivative of
            the fractional metric (cells with an abrupt change in degradation
            rate might have something else going on). Typical value might be 0.04
        extrapolate_threshold (bool): Should threshold crossing point be
            extrapolated for cells that have not yet reached the threshold
            (warning: this uses a linear extrapolation from the last two
            diagnostic cycles)
    """
    DEFAULT_HYPERPARAMETERS = {
        "parameters_dir": PROTOCOL_PARAMETERS_DIR,
        "quantities": ['discharge_energy', 'discharge_capacity'],
        "threshold": 0.8,
        "metric": "discharge_energy",
        "filter_kinks": None,
        "interpolation_axes": ["normalized_regular_throughput", "cycle_index"],
        "cycle_type": "rpt_1C",
        "extrapolate_threshold": True
    }

    def validate(self):
        """
        This function determines if the input data has the necessary attributes for
        creation of this feature class. It should test for all of the possible reasons
        that feature generation would fail for this particular input data.

        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
            gets featurized. These could be filters for column or row operations
        Returns:
            bool: True/False indication of ability to proceed with feature generation
        """
        return featurizer_helpers.check_diagnostic_validation(self.datapath)

    def create_features(self):
        """
        Generates diagnostic-property features from processed cycler run, including values for n*x method
        Args:
            self.datapath (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            pd.DataFrame: with "cycle_index", "fractional_metric", "x", "n", "cycle_type" and "metric" columns, rows
            for each diagnostic cycle of the cell
        """

        parameters_path = self.hyperparameters["parameters_dir"]

        cycle_types = self.datapath.diagnostic_summary.cycle_type.unique()
        X = pd.DataFrame()
        for quantity in self.hyperparameters["quantities"]:
            for cycle_type in cycle_types:
                summary_diag_cycle_type = featurizer_helpers.get_fractional_quantity_remaining_nx(
                    self.datapath, quantity, cycle_type,
                    parameters_path=parameters_path
                )

                summary_diag_cycle_type.loc[:, "cycle_type"] = cycle_type
                summary_diag_cycle_type.loc[:, "metric"] = quantity
                X = X.append(summary_diag_cycle_type)

        X_condensed = self.get_threshold_targets(X)
        self.features = X_condensed

    def get_threshold_targets(self, df):
        """
        Apply a threshold via interpolation for determining various
        metrics (e.g., discharge energy) from diagnostic cycles.

        Args:
            df (pd.DataFrame): A dataframe of diagnostic cycle data
                for a single battery cycler run.

        Returns:
            (pd.DataFrame): Contains a vector for interpolated/intercept
                data for determining threshold.

        """
        cycle_type = self.hyperparameters["cycle_type"]
        metric = self.hyperparameters["metric"]
        interpolation_axes = self.hyperparameters["interpolation_axes"]
        threshold = self.hyperparameters["threshold"]
        filter_kinks = self.hyperparameters["filter_kinks"]
        extrapolate_threshold = self.hyperparameters["extrapolate_threshold"]

        if filter_kinks:
            if np.any(df['fractional_metric'].diff().diff() < filter_kinks):
                last_good_cycle = df[
                    df['fractional_metric'].diff().diff() < filter_kinks][
                    'cycle_index'].min()
                df = df[df['cycle_index'] < last_good_cycle]

        x_axes = []
        for type in interpolation_axes:
            x_axes.append(df[type])
        y_interpolation_axis = df['fractional_metric']

        # Logic around how to deal with cells that have not crossed threshold
        if df['fractional_metric'].min() > threshold and \
                not extrapolate_threshold:
            BEEPFeaturizationError(
                "DiagnosticProperties data has not crossed threshold "
                "and extrapolation inaccurate"
            )
        elif df['fractional_metric'].min() > threshold and \
                extrapolate_threshold:
            fill_value = "extrapolate"
            bounds_error = False
            x_linspaces = []
            for x_axis in x_axes:
                y1 = y_interpolation_axis.iloc[-2]
                y2 = y_interpolation_axis.iloc[-1]
                x1 = x_axis.iloc[-2]
                x2 = x_axis.iloc[-1]
                x_thresh_extrap = (threshold - 0.1 - y1) * (x2 - x1) / (
                        y2 - y1) + x1
                x_linspaces.append(
                    np.linspace(x_axis.min(), x_thresh_extrap, num=1000)
                )
        else:
            fill_value = np.nan
            bounds_error = True
            x_linspaces = []
            for x_axis in x_axes:
                x_linspaces.append(
                    np.linspace(x_axis.min(), x_axis.max(), num=1000))

        f_axis = []
        for x_axis in x_axes:
            f_axis.append(
                interp1d(
                    x_axis,
                    y_interpolation_axis,
                    kind='linear',
                    bounds_error=bounds_error,
                    fill_value=fill_value
                )
            )

        x_to_threshold = []
        for indx, x_linspace in enumerate(x_linspaces):
            crossing_array = abs(f_axis[indx](x_linspace) - threshold)
            x_to_threshold.append(x_linspace[np.argmin(crossing_array)])

        if ~(x_to_threshold[0] > 0) or ~(x_to_threshold[1] > 0):
            raise BEEPFeaturizationError(
                "DiagnosticProperties data does not have a positive value "
                "to threshold"
            )

        if "normalized_regular_throughput" in interpolation_axes:
            real_throughput_to_threshold = x_to_threshold[
                                               interpolation_axes.index(
                                                   "normalized_regular_throughput")] * \
                                           df[
                                               'initial_regular_throughput'].values[
                                               0]
            x_to_threshold.append(real_throughput_to_threshold)
            interpolation_axes = interpolation_axes + [
                "real_regular_throughput"]

        threshold_dict = {
            'initial_regular_throughput':
                df['initial_regular_throughput'].values[0],
        }

        for indx, x_axis in enumerate(interpolation_axes):
            threshold_dict[
                cycle_type + metric + str(threshold) + '_' + x_axis] = [
                x_to_threshold[indx]]

        return pd.DataFrame(threshold_dict)