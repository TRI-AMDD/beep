import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from beep.features.featurizer import BEEPAllCyclesFeaturizer


class DeltaQFastCharge(BEEPAllCyclesFeaturizer):
    """
    Object corresponding to feature object. Includes constructors
    to create the features, object names and metadata attributes in the
    object
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    """
    DEFAULT_HYPERPARAMETERS = {
        "init_pred_cycle": 10,
        "mid_pred_cycle": 91,
        "final_pred_cycle": 100,
        "n_nominal_cycles": 40
    }

    def validate(self):
        """
        This function determines if the input data has the necessary attributes for
        creation of this feature class. It should test for all of the possible reasons
        that feature generation would fail for this particular input data.

        Args:
            self.datapath (beep.structure.ProcessedCyclerRun): data from cycler run
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
            gets featurized. These could be filters for column or row operations
        Returns:
            bool: True/False indication of ability to proceed with feature generation
        """

        if not self.datapath.structured_summary.index.max() > \
               self.hyperparameters["final_pred_cycle"]:
            return False, "Structured summary index max is less than final pred cycle"
        elif not self.datapath.structured_summary.index.min() <= \
                 self.hyperparameters["init_pred_cycle"]:
            return False, "Structured summary index min is more than initial pred cycle"
        elif "cycle_index" not in self.datapath.structured_summary.columns:
            return False, "Structured summary missing critical data: 'cycle_index'"
        elif "cycle_index" not in self.datapath.structured_data.columns:
            return False, "Structured data missing critical data: 'cycle_index'"
        elif not self.hyperparameters["mid_pred_cycle"] > 10:
            return False, "Middle pred. cycle less than threshold value of 10"
        elif not self.hyperparameters["final_pred_cycle"] > \
                 self.hyperparameters["mid_pred_cycle"]:
            return False, "Final pred cycle less than middle pred cycle"
        else:
            return True, None

    def create_features(self):
        """
        """
        i_final = self.hyperparameters[
                      "final_pred_cycle"] - 1  # python indexing
        i_mid = self.hyperparameters["mid_pred_cycle"] - 1

        summary = self.datapath.structured_summary
        self.hyperparameters[
            "n_nominal_cycles"
        ] = 40  # For nominal capacity, use median discharge capacity of first n cycles

        if "step_type" in self.datapath.structured_data.columns:
            interpolated_df = self.datapath.structured_data[
                self.datapath.structured_data.step_type == "discharge"
                ]
        else:
            interpolated_df = self.datapath.structured_data
        X = pd.DataFrame(np.zeros((1, 20)))
        labels = []
        # Discharge capacity, cycle 2 = Q(n=2)
        X[0] = summary.discharge_capacity.iloc[1]
        labels.append("discharge_capacity_cycle_2")

        # Max discharge capacity - discharge capacity, cycle 2 = max_n(Q(n)) - Q(n=2)
        X[1] = max(
            summary.discharge_capacity.iloc[np.arange(i_final + 1)]
            - summary.discharge_capacity.iloc[1]
        )
        labels.append("max_discharge_capacity_difference")

        # Discharge capacity, cycle 100 = Q(n=100)
        X[2] = summary.discharge_capacity.iloc[i_final]
        labels.append("discharge_capacity_cycle_100")

        # Feature representing time-temperature integral over cycles 2 to 100
        X[3] = np.nansum(
            summary.time_temperature_integrated.iloc[np.arange(i_final + 1)])
        labels.append("integrated_time_temperature_cycles_1:100")

        # Mean of charge times of first 5 cycles
        X[4] = np.nanmean(summary.charge_duration.iloc[1:6])
        labels.append("charge_time_cycles_1:5")

        # Descriptors based on capacity loss between cycles 10 and 100.
        Qd_final = interpolated_df.discharge_capacity[
            interpolated_df.cycle_index == i_final
            ]
        Qd_10 = interpolated_df.discharge_capacity[
            interpolated_df.cycle_index == 9]

        Qd_diff = Qd_final.values - Qd_10.values

        # If DeltaQ(V) is not an empty array, compute summary stats, else initialize with np.nan
        # Cells discharged rapidly over a narrow voltage window run into have no interpolated discharge steps
        if len(Qd_diff):
            X[5] = np.log10(np.abs(np.nanmin(Qd_diff)))  # Minimum
            X[6] = np.log10(np.abs(np.nanmean(Qd_diff)))  # Mean
            X[7] = np.log10(np.abs(np.nanvar(Qd_diff)))  # Variance
            X[8] = np.log10(np.abs(skew(Qd_diff)))  # Skewness
            X[9] = np.log10(np.abs(kurtosis(Qd_diff)))  # Kurtosis
            X[10] = np.log10(np.abs(Qd_diff[0]))  # First difference
        else:
            X[5:11] = np.nan

        labels.append("abs_min_discharge_capacity_difference_cycles_2:100")
        labels.append("abs_mean_discharge_capacity_difference_cycles_2:100")
        labels.append("abs_variance_discharge_capacity_difference_cycles_2:100")
        labels.append("abs_skew_discharge_capacity_difference_cycles_2:100")
        labels.append("abs_kurtosis_discharge_capacity_difference_cycles_2:100")
        labels.append("abs_first_discharge_capacity_difference_cycles_2:100")

        X[11] = np.max(summary.temperature_maximum.iloc[
                           list(range(1, i_final + 1))])  # Max T
        labels.append("max_temperature_cycles_1:100")

        X[12] = np.min(summary.temperature_minimum.iloc[
                           list(range(1, i_final + 1))])  # Min T
        labels.append("min_temperature_cycles_1:100")

        # Slope and intercept of linear fit to discharge capacity as a fn of cycle #, cycles 2 to 100

        X[13], X[14] = np.polyfit(
            list(range(1, i_final + 1)),
            summary.discharge_capacity.iloc[list(range(1, i_final + 1))],
            1,
        )

        labels.append("slope_discharge_capacity_cycle_number_2:100")
        labels.append("intercept_discharge_capacity_cycle_number_2:100")

        # Slope and intercept of linear fit to discharge capacity as a fn of cycle #, cycles 91 to 100
        X[15], X[16] = np.polyfit(
            list(range(i_mid, i_final + 1)),
            summary.discharge_capacity.iloc[list(range(i_mid, i_final + 1))],
            1,
        )
        labels.append("slope_discharge_capacity_cycle_number_91:100")
        labels.append("intercept_discharge_capacity_cycle_number_91:100")

        IR_trend = summary.dc_internal_resistance.iloc[
            list(range(1, i_final + 1))]
        if any(v == 0 for v in IR_trend):
            IR_trend[IR_trend == 0] = np.nan

        # Internal resistance minimum
        X[17] = np.nanmin(IR_trend)
        labels.append("min_internal_resistance_cycles_2:100")

        # Internal resistance at cycle 2
        X[18] = summary.dc_internal_resistance.iloc[1]
        labels.append("internal_resistance_cycle_2")

        # Internal resistance at cycle 100 - cycle 2
        X[19] = (
                summary.dc_internal_resistance.iloc[i_final] -
                summary.dc_internal_resistance.iloc[1]
        )
        labels.append("internal_resistance_difference_cycles_2:100")

        # Nominal capacity
        end = self.hyperparameters["n_nominal_cycles"]
        X[20] = np.median(summary.discharge_capacity.iloc[0: end])
        labels.append("nominal_capacity_by_median")

        X.columns = labels
        self.features = X