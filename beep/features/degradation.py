import json
import os
import numpy as np
import pandas as pd
from monty.json import MSONable
from monty.serialization import loadfn, dumpfn
from scipy.stats import skew, kurtosis

from beep import FEATURES_DIR
from beep.structure.cli import auto_load_processed
from beep.utils import WorkflowOutputs
from beep.features import featurizer_helpers, intracell_losses
from beep.features.base import BeepFeatures, FEATURE_HYPERPARAMS, BEEPFeaturizer, BEEPFeaturizationError

class DegradationPredictor(MSONable):
    """
    Object corresponding to feature matrix. Includes constructors
    to initialize the feature vectors.
    Attributes:
        name (str): predictor object name.
        X (pandas.DataFrame): data as records x features.
        y (pandas.DataFrame): targets.
        feature_labels (list): feature labels.
        predict_only (bool): True/False to specify predict/train mode.
        prediction_type (str): Type of regression - 'single' vs 'multi'.
        predicted_quantity (str): 'cycle' or 'capacity'.
        nominal_capacity (float):
    """

    def __init__(
            self,
            name,
            X,
            feature_labels=None,
            y=None,
            nominal_capacity=1.1,
            predict_only=False,
            predicted_quantity="cycle",
            prediction_type="multi",
    ):
        """
        Args:
            name (str): predictor object name
            X (pandas.DataFrame): features in DataFrame format.
            name (str): name of method for featurization.
            y (pandas.Dataframe or float): one or more outcomes.
            predict_only (bool): True/False to specify predict/train mode.
            predicted_quantity (str): 'cycle' or 'capacity'.
            prediction_type (str): Type of regression - 'single' vs 'multi'.
        """
        self.name = name
        self.X = X
        self.feature_labels = feature_labels
        self.predict_only = predict_only
        self.prediction_type = prediction_type
        self.predicted_quantity = predicted_quantity
        self.y = y
        self.nominal_capacity = nominal_capacity

    @classmethod
    def from_processed_cycler_run_file(
            cls,
            path,
            features_label="full_model",
            predict_only=False,
            predicted_quantity="cycle",
            prediction_type="multi",
            diagnostic_features=False,
    ):
        """
        Args:
            path (str): string corresponding to file path with ProcessedCyclerRun object.
            features_label (str): name of method for featurization.
            predict_only (bool): True/False to specify predict/train mode.
            predicted_quantity (str): 'cycle' or 'capacity'.
            prediction_type (str): Type of regression - 'single' vs 'multi'.
            diagnostic_features (bool): whether to compute diagnostic features.
        """
        processed_cycler_run = auto_load_processed(path)

        if features_label == "full_model":
            return cls.init_full_model(
                processed_cycler_run,
                predict_only=predict_only,
                predicted_quantity=predicted_quantity,
                diagnostic_features=diagnostic_features,
                prediction_type=prediction_type,
            )
        else:
            raise NotImplementedError

    @classmethod
    def init_full_model(
            cls,
            processed_cycler_run,
            init_pred_cycle=10,
            mid_pred_cycle=91,
            final_pred_cycle=100,
            predict_only=False,
            prediction_type="multi",
            predicted_quantity="cycle",
            diagnostic_features=False,
    ):
        """
        Generate features listed in early prediction manuscript
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun): information about cycler run
            init_pred_cycle (int): index of initial cycle index used for predictions
            mid_pred_cycle (int): index of intermediate cycle index used for predictions
            final_pred_cycle (int): index of highest cycle index used for predictions
            predict_only (bool): whether or not to include cycler life in the object
            prediction_type (str): 'single': cycle life to reach 80% capacity.
                                   'multi': remaining capacity at fixed cycles
            predicted_quantity (str): quantity being predicted - cycles/capacity
            diagnostic_features (bool): whether or not to compute diagnostic features
        Returns:
            beep.featurize.DegradationPredictor: DegradationPredictor corresponding to the ProcessedCyclerRun file.
        """
        assert mid_pred_cycle > 10, "Insufficient cycles for analysis"
        assert (
                final_pred_cycle > mid_pred_cycle
        ), "Must have final_pred_cycle > mid_pred_cycle"
        i_final = final_pred_cycle - 1  # python indexing
        i_mid = mid_pred_cycle - 1
        summary = processed_cycler_run.structured_summary
        assert (
                len(processed_cycler_run.structured_summary) > final_pred_cycle
        ), "cycle count must exceed final_pred_cycle"
        cycles_to_average_over = (
            40
        # For nominal capacity, use median discharge capacity of first n cycles
        )

        # Features in "nature energy" set only use discharge portion of the cycle
        if "step_type" in processed_cycler_run.structured_data.columns:
            interpolated_df = processed_cycler_run.structured_data[
                processed_cycler_run.structured_data.step_type == "discharge"
                ]
        else:
            interpolated_df = processed_cycler_run.structured_data

        X = pd.DataFrame(np.zeros((1, 20)))
        labels = []
        # Discharge capacity, cycle 2 = Q(n=2)
        X[0] = summary.discharge_capacity[1]
        labels.append("discharge_capacity_cycle_2")

        # Max discharge capacity - discharge capacity, cycle 2 = max_n(Q(n)) - Q(n=2)
        X[1] = max(
            summary.discharge_capacity[np.arange(final_pred_cycle)]
            - summary.discharge_capacity[1]
        )
        labels.append("max_discharge_capacity_difference")

        # Discharge capacity, cycle 100 = Q(n=100)
        X[2] = summary.discharge_capacity[i_final]
        labels.append("discharge_capacity_cycle_100")

        # Feature representing time-temperature integral over cycles 2 to 100
        X[3] = np.nansum(
            summary.time_temperature_integrated[np.arange(final_pred_cycle)]
        )
        labels.append("integrated_time_temperature_cycles_1:100")

        # Mean of charge times of first 5 cycles
        X[4] = np.nanmean(summary.charge_duration[1:6])
        labels.append("charge_time_cycles_1:5")

        # Descriptors based on capacity loss between cycles 10 and 100.
        Qd_final = interpolated_df.discharge_capacity[
            interpolated_df.cycle_index == i_final
            ]
        Qd_10 = interpolated_df.discharge_capacity[
            interpolated_df.cycle_index == 9]

        Qd_diff = Qd_final.values - Qd_10.values

        X[5] = np.log10(np.abs(np.min(Qd_diff)))  # Minimum
        labels.append("abs_min_discharge_capacity_difference_cycles_2:100")

        X[6] = np.log10(np.abs(np.mean(Qd_diff)))  # Mean
        labels.append("abs_mean_discharge_capacity_difference_cycles_2:100")

        X[7] = np.log10(np.abs(np.var(Qd_diff)))  # Variance
        labels.append("abs_variance_discharge_capacity_difference_cycles_2:100")

        X[8] = np.log10(np.abs(skew(Qd_diff)))  # Skewness
        labels.append("abs_skew_discharge_capacity_difference_cycles_2:100")

        X[9] = np.log10(np.abs(kurtosis(Qd_diff)))  # Kurtosis
        labels.append("abs_kurtosis_discharge_capacity_difference_cycles_2:100")

        X[10] = np.log10(np.abs(Qd_diff[0]))  # First difference
        labels.append("abs_first_discharge_capacity_difference_cycles_2:100")

        X[11] = max(
            summary.temperature_maximum[list(range(1, final_pred_cycle))]
        )  # Max T
        labels.append("max_temperature_cycles_1:100")

        X[12] = min(
            summary.temperature_minimum[list(range(1, final_pred_cycle))]
        )  # Min T
        labels.append("min_temperature_cycles_1:100")

        # Slope and intercept of linear fit to discharge capacity as a fn of cycle #, cycles 2 to 100

        X[13], X[14] = np.polyfit(
            list(range(1, final_pred_cycle)),
            summary.discharge_capacity[list(range(1, final_pred_cycle))],
            1,
        )

        labels.append("slope_discharge_capacity_cycle_number_2:100")
        labels.append("intercept_discharge_capacity_cycle_number_2:100")

        # Slope and intercept of linear fit to discharge capacity as a fn of cycle #, cycles 91 to 100
        X[15], X[16] = np.polyfit(
            list(range(i_mid, final_pred_cycle)),
            summary.discharge_capacity[list(range(i_mid, final_pred_cycle))],
            1,
        )
        labels.append("slope_discharge_capacity_cycle_number_91:100")
        labels.append("intercept_discharge_capacity_cycle_number_91:100")

        IR_trend = summary.dc_internal_resistance[
            list(range(1, final_pred_cycle))]
        if any(v == 0 for v in IR_trend):
            IR_trend[IR_trend == 0] = np.nan

        # Internal resistance minimum
        X[17] = np.nanmin(IR_trend)
        labels.append("min_internal_resistance_cycles_2:100")

        # Internal resistance at cycle 2
        X[18] = summary.dc_internal_resistance[1]
        labels.append("internal_resistance_cycle_2")

        # Internal resistance at cycle 100 - cycle 2
        X[19] = (
                summary.dc_internal_resistance[i_final] -
                summary.dc_internal_resistance[1]
        )
        labels.append("internal_resistance_difference_cycles_2:100")

        X.columns = labels
        if predict_only:
            y = None
        else:
            if prediction_type == "single":
                y = processed_cycler_run.get_cycle_life()
            elif prediction_type == "multi":
                if predicted_quantity == "cycle":
                    y = processed_cycler_run.capacities_to_cycles(
                        thresh_max_cap=0.98, thresh_min_cap=0.78,
                        interval_cap=0.03
                    )
                elif predicted_quantity == "capacity":
                    y = processed_cycler_run.cycles_to_capacities()
                else:
                    raise NotImplementedError(
                        "{} predicted_quantity type not implemented".format(
                            predicted_quantity
                        )
                    )
        nominal_capacity = np.median(
            summary.discharge_capacity.iloc[0:cycles_to_average_over]
        )

        return cls(
            "full_model",
            X,
            feature_labels=labels,
            y=y,
            nominal_capacity=nominal_capacity,
            predict_only=predict_only,
            prediction_type=prediction_type,
            predicted_quantity=predicted_quantity,
        )

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
            "feature_labels": self.feature_labels,
            "predict_only": self.predict_only,
            "prediction_type": self.prediction_type,
            "nominal_capacity": self.nominal_capacity,
        }
        if isinstance(self.y, pd.DataFrame):
            obj["y"] = self.y.to_dict("list")
        else:
            obj["y"] = self.y
        return obj

    @classmethod
    def from_dict(cls, d):
        """MSONable deserialization method"""
        d["X"] = pd.DataFrame(d["X"])
        return cls(**d)