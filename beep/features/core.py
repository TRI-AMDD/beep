import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from beep.features.base import BEEPFeaturizer, BEEPFeaturizationError


class HPPCResistanceVoltageFeatures(BEEPFeaturizer):
    DEFAULT_HYPERPARAMETERS = {
        "test_time_filter_sec": 1000000,
        "cycle_index_filter": 6,
        "diag_pos": 1,
        "soc_window": 8,
        "parameters_path": PROTOCOL_PARAMETERS_DIR
    }

    def validate(self):
        val, msg = featurizer_helpers.check_diagnostic_validation(self.datapath)
        if val:
            conditions = []
            conditions.append(
                any(
                    [
                        "hppc" in x
                        for x in
                        self.datapath.diagnostic_summary.cycle_type.unique()
                    ]
                )
            )
            if all(conditions):
                return True, None
            else:
                return False, "HPPC conditions not met for this cycler run"
        else:
            return val, msg

    def create_features(self):
        # Filter out low cycle numbers at the end of the test, corresponding to the "final" diagnostic
        self.datapath.diagnostic_data = self.datapath.diagnostic_data[
            ~((self.datapath.diagnostic_data.test_time > self.hyperparameters[
                'test_time_filter_sec']) &
              (self.datapath.diagnostic_data.cycle_index < self.hyperparameters[
                  'cycle_index_filter']))
        ]
        self.datapath.diagnostic_data = self.datapath.diagnostic_data.groupby(
            ["cycle_index", "step_index", "step_index_counter"]
        ).filter(lambda x: ~x["test_time"].isnull().all())

        # diffusion features
        diffusion_features = featurizer_helpers.get_diffusion_features(
            self.datapath, self.hyperparameters["diag_pos"]
        )

        hppc_r = pd.DataFrame()
        # the 9 by 6 dataframe
        df_dr = featurizer_helpers.get_dr_df(
            self.datapath, self.hyperparameters["diag_pos"]
        )
        # transform this dataframe to be 1 by 54
        columns = df_dr.columns
        for column in columns:
            for r in range(len(df_dr[column])):
                name = column + str(r)
                hppc_r[name] = [df_dr[column][r]]

        # the variance of ocv features
        hppc_ocv = featurizer_helpers.get_hppc_ocv(
            self.datapath,
            self.hyperparameters["diag_pos"],
            parameters_path=self.hyperparameters["parameters_path"]
        )

        # the v_diff features
        v_diff = featurizer_helpers.get_v_diff(
            self.datapath,
            self.hyperparameters["diag_pos"],
            self.hyperparameters["soc_window"],
            self.hyperparameters["parameters_path"]
        )

        # merge everything together as a final result dataframe
        self.features = pd.concat(
            [hppc_r, hppc_ocv, v_diff, diffusion_features], axis=1)


class CycleSummaryStats(BEEPFeaturizer):
    DEFAULT_HYPERPARAMETERS = {
        "cycle_comp_num": [10, 100],
        "statistics": ["var", "min", "mean", "skew", "kurtosis", "abs",
                       "square"]
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

        # TODO: not sure this is necessary
        # Check for data in each of the selected cycles
        index_1, index_2 = self.hyperparameters['cycle_comp_num']
        cycle_1 = self.datapath.structured_data[
            self.datapath.structured_data.cycle_index == index_1]
        cycle_2 = self.datapath.structured_data[
            self.datapath.structured_data.cycle_index == index_2]
        if len(cycle_1) == 0 or len(cycle_2) == 0:
            return False, "Length of one or more comparison cycles is zero"

        # TODO: check whether this is good
        # Check for relevant data
        required_columns = [
            'charge_capacity',
            'discharge_capacity',
            'charge_energy',
            'discharge_energy',
        ]
        pcycler_run_columns = self.datapath.structured_data.columns
        if not all(
                [column in pcycler_run_columns for column in required_columns]):
            return False, f"Required column not present in all structured data " \
                          f"(must have all of: {required_columns})"

        return True, None

    def create_features(self):
        """
        Generate features listed in early prediction manuscript using both diagnostic and regular cycles

        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            X (pd.DataFrame): Dataframe containing the feature
        """

        # TODO: extend this dataframe and uncomment energy features when
        #   structuring is refactored
        X = pd.DataFrame(np.zeros((1, 28)))

        reg_cycle_comp_num = self.hyperparameters.get("cycle_comp_num")
        cycle_comp_1 = self.datapath.structured_data[
            self.datapath.structured_data.cycle_index == reg_cycle_comp_num[1]
            ]
        cycle_comp_0 = self.datapath.structured_data[
            self.datapath.structured_data.cycle_index == reg_cycle_comp_num[0]
            ]
        Qc100_1 = cycle_comp_1[
            cycle_comp_1.step_type == "charge"].charge_capacity
        Qc10_1 = cycle_comp_0[
            cycle_comp_0.step_type == "charge"].charge_capacity
        QcDiff = Qc100_1.values - Qc10_1.values
        QcDiff = QcDiff[~np.isnan(QcDiff)]

        X.loc[0, 0:6] = self.get_summary_statistics(QcDiff)

        Qd100_1 = cycle_comp_1[
            cycle_comp_1.step_type == "discharge"].discharge_capacity
        Qd10_1 = cycle_comp_0[
            cycle_comp_0.step_type == "discharge"].discharge_capacity
        QdDiff = Qd100_1.values - Qd10_1.values
        QdDiff = QdDiff[~np.isnan(QdDiff)]

        X.loc[0, 7:13] = self.get_summary_statistics(QdDiff)

        # # Charging Energy features
        Ec100_1 = cycle_comp_1[cycle_comp_1.step_type == "charge"].charge_energy
        Ec10_1 = cycle_comp_0[cycle_comp_0.step_type == "charge"].charge_energy
        EcDiff = Ec100_1.values - Ec10_1.values
        EcDiff = EcDiff[~np.isnan(EcDiff)]

        X.loc[0, 14:20] = self.get_summary_statistics(EcDiff)

        # # Discharging Energy features
        Ed100_1 = cycle_comp_1[
            cycle_comp_1.step_type == "charge"].discharge_energy
        Ed10_1 = cycle_comp_0[
            cycle_comp_0.step_type == "charge"].discharge_energy
        EdDiff = Ed100_1.values - Ed10_1.values
        EdDiff = EdDiff[~np.isnan(EdDiff)]

        X.loc[0, 21:27] = self.get_summary_statistics(EdDiff)

        quantities = [
            "charging_capacity",
            "discharging_capacity",
            "charging_energy",
            "discharging_energy",
        ]

        X.columns = [y + "_" + x for x in quantities for y in
                     self.hyperparameters["statistics"]]

        self.features = X

    def get_summary_statistics(self, array):
        """
        Static method for getting values corresponding
        to standard 7 operations that many beep features
        use, i.e. log of absolute value of each of
        variance, min, mean, skew, kurtosis, the sum of
        the absolute values and the sum of squares

        Args:
            array (list, np.ndarray): array of values to get
                standard operation values for, e.g. cycle
                discharging capacity, QcDiff, etc.

        Returns:
            [float]: list of features

        """

        stats_names = self.hyperparameters["statistics"]
        supported_stats = self.DEFAULT_HYPERPARAMETERS["statistics"]

        if any(s not in supported_stats for s in stats_names):
            raise ValueError(
                f"Unsupported statistics in {stats_names}: supported statistics are {supported_stats}")

        stats = []

        if "var" in stats_names:
            stats.append(np.log10(np.absolute(np.var(array))))
        if "min" in stats_names:
            stats.append(np.log10(np.absolute(min(array))))
        if "mean" in stats_names:
            stats.append(np.log10(np.absolute(np.mean(array))))
        if "skew" in stats_names:
            stats.append(np.log10(np.absolute(skew(array))))
        if "kurtosis" in stats_names:
            stats.append(np.log10(
                np.absolute(kurtosis(array, fisher=False, bias=False))))
        if "abs" in stats_names:
            stats.append(np.log10(np.sum(np.absolute(array))))
        if "square" in stats_names:
            stats.append(np.log10(np.sum(np.square(array))))

        return np.asarray(stats)


class DiagnosticSummaryStats(CycleSummaryStats):
    """
    Object corresponding to summary statistics from a diagnostic cycle of
    specific type. Includes constructors to create the features, object names
    and metadata attributes in the object.  Inherits from RegularCycleSummaryStats
    to reuse standard feature generation

    name (str): predictor object name.
    X (pandas.DataFrame): features in DataFrame format.
    metadata (dict): information about the conditions, data
        and code used to produce features
    """
    DEFAULT_HYPERPARAMETERS = {
        "test_time_filter_sec": 1000000,
        "cycle_index_filter": 6,
        "diagnostic_cycle_type": 'rpt_0.2C',
        "diag_pos_list": [0, 1],
        "statistics": ["var", "min", "mean", "skew", "kurtosis", "abs",
                       "square"],
        "parameters_path": PROTOCOL_PARAMETERS_DIR
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
        val, msg = featurizer_helpers.check_diagnostic_validation(self.datapath)
        if val:
            df = self.datapath.diagnostic_summary
            df = df[
                df.cycle_type == self.hyperparameters["diagnostic_cycle_type"]]
            if df.cycle_index.nunique() >= max(
                    self.hyperparameters["diag_pos_list"]) + 1:
                return True, None
            else:
                return False, "Diagnostic cycles insufficient for featurization"
        else:
            return val, msg

    def get_summary_diff(
            self,
            pos=None,
            cycle_types=("rpt_0.2C", "rpt_1C", "rpt_2C"),
            metrics=(
                    "discharge_capacity", "discharge_energy", "charge_capacity",
                    "charge_energy")
    ):
        """
        Helper function to calculate difference between summary values in the diagnostic cycles

                Args:
                    processed_cycler_run (beep.structure.ProcessedCyclerRun)
                    pos (list): position of the diagnostics to use in the calculation
                    cycle_types (list): calculate difference for these diagnostic types
                    metrics (str): Calculate difference for these metrics

                Returns:
                    values (list): List of difference values to insert into the dataframe
                    names (list): List of column headers to use in the creation of the dataframe
                """
        pos = self.hyperparameters["diag_pos_list"] if not pos else pos

        values = []
        names = []
        for cycle_type in cycle_types:
            diag_type_summary = self.datapath.diagnostic_summary[
                self.datapath.diagnostic_summary.cycle_type == cycle_type]
            for metric in metrics:
                diff = (diag_type_summary.iloc[pos[1]][metric] -
                        diag_type_summary.iloc[pos[0]][metric]) \
                       / diag_type_summary.iloc[pos[0]][metric]
                values.append(diff)
                names.append("diag_sum_diff_" + str(pos[0]) + "_" + str(
                    pos[1]) + "_" + cycle_type + metric)
        return values, names

    def create_features(self):
        """
        Generate features listed in early prediction manuscript using both diagnostic and regular cycles

        Args:
            self.datapathn (beep.structure.ProcessedCyclerRun)
            self.hyperparameters (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            X (pd.DataFrame): Dataframe containing the feature
        """
        # Filter out "final" diagnostic cycles that have been appended to the end of the file with the wrong
        # cycle number(test time is monotonic)
        self.datapath.diagnostic_data = self.datapath.diagnostic_data[
            ~((self.datapath.diagnostic_data.test_time > self.hyperparameters[
                'test_time_filter_sec']) &
              (self.datapath.diagnostic_data.cycle_index < self.hyperparameters[
                  'cycle_index_filter']))
        ]
        self.datapath.diagnostic_data = self.datapath.diagnostic_data.groupby(
            ["cycle_index", "step_index", "step_index_counter"]
        ).filter(lambda x: ~x["test_time"].isnull().all())

        diag_intrp = self.datapath.diagnostic_data

        X = pd.DataFrame(np.zeros((1, 54)))

        # Calculate the cycles and the steps for the selected diagnostics
        cycles = diag_intrp.cycle_index[diag_intrp.cycle_type ==
                                        self.hyperparameters[
                                            "diagnostic_cycle_type"]].unique()
        step_dict_0 = featurizer_helpers.get_step_index(
            self.datapath,
            cycle_type=self.hyperparameters["diagnostic_cycle_type"],
            diag_pos=self.hyperparameters["diag_pos_list"][0],
            parameters_path=self.hyperparameters["parameters_path"]
        )
        step_dict_1 = featurizer_helpers.get_step_index(
            self.datapath,
            cycle_type=self.hyperparameters["diagnostic_cycle_type"],
            diag_pos=self.hyperparameters["diag_pos_list"][1],
            parameters_path=self.hyperparameters["parameters_path"]
        )

        # Create masks for each position in the data
        mask_pos_0_charge = ((diag_intrp.cycle_index == cycles[
            self.hyperparameters["diag_pos_list"][0]]) &
                             (diag_intrp.step_index == step_dict_0[
                                 self.hyperparameters[
                                     "diagnostic_cycle_type"] + '_charge']))
        mask_pos_1_charge = ((diag_intrp.cycle_index == cycles[
            self.hyperparameters["diag_pos_list"][1]]) &
                             (diag_intrp.step_index == step_dict_1[
                                 self.hyperparameters[
                                     "diagnostic_cycle_type"] + '_charge']))
        mask_pos_0_discharge = ((diag_intrp.cycle_index == cycles[
            self.hyperparameters["diag_pos_list"][0]]) &
                                (diag_intrp.step_index ==
                                 step_dict_0[self.hyperparameters[
                                                 "diagnostic_cycle_type"] + '_discharge']))
        mask_pos_1_discharge = ((diag_intrp.cycle_index == cycles[
            self.hyperparameters["diag_pos_list"][1]]) &
                                (diag_intrp.step_index ==
                                 step_dict_1[self.hyperparameters[
                                                 "diagnostic_cycle_type"] + '_discharge']))

        # Charging Capacity features
        Qc_1 = diag_intrp.charge_capacity[mask_pos_1_charge]
        Qc_0 = diag_intrp.charge_capacity[mask_pos_0_charge]
        QcDiff = Qc_1.values - Qc_0.values
        QcDiff = QcDiff[~np.isnan(QcDiff)]

        X.loc[0, 0:6] = self.get_summary_statistics(QcDiff)

        # Discharging Capacity features
        Qd_1 = diag_intrp.discharge_capacity[mask_pos_1_discharge]
        Qd_0 = diag_intrp.discharge_capacity[mask_pos_0_discharge]
        QdDiff = Qd_1.values - Qd_0.values
        QdDiff = QdDiff[~np.isnan(QdDiff)]

        X.loc[0, 7:13] = self.get_summary_statistics(QdDiff)

        # Charging Energy features
        Ec_1 = diag_intrp.charge_energy[mask_pos_1_charge]
        Ec_0 = diag_intrp.charge_energy[mask_pos_0_charge]
        EcDiff = Ec_1.values - Ec_0.values
        EcDiff = EcDiff[~np.isnan(EcDiff)]

        X.loc[0, 14:20] = self.get_summary_statistics(EcDiff)

        # Discharging Energy features
        Ed_1 = diag_intrp.discharge_energy[mask_pos_1_discharge]
        Ed_0 = diag_intrp.discharge_energy[mask_pos_0_discharge]
        EdDiff = Ed_1.values - Ed_0.values
        EdDiff = EdDiff[~np.isnan(EdDiff)]

        X.loc[0, 21:27] = self.get_summary_statistics(EdDiff)

        # Charging dQdV features
        dQdVc_1 = diag_intrp.charge_dQdV[mask_pos_1_charge]
        dQdVc_0 = diag_intrp.charge_dQdV[mask_pos_0_charge]
        dQdVcDiff = dQdVc_1.values - dQdVc_0.values
        dQdVcDiff = dQdVcDiff[~np.isnan(dQdVcDiff)]

        X.loc[0, 28:34] = self.get_summary_statistics(dQdVcDiff)

        # Discharging Capacity features
        dQdVd_1 = diag_intrp.discharge_dQdV[mask_pos_1_discharge]
        dQdVd_0 = diag_intrp.discharge_dQdV[mask_pos_0_discharge]
        dQdVdDiff = dQdVd_1.values - dQdVd_0.values
        dQdVdDiff = dQdVdDiff[~np.isnan(dQdVdDiff)]

        X.loc[0, 35:41] = self.get_summary_statistics(dQdVdDiff)

        X.loc[0, 42:53], names = self.get_summary_diff(
            self.hyperparameters["diag_pos_list"]
        )

        quantities = [
            "charging_capacity",
            "discharging_capacity",
            "charging_energy",
            "discharging_energy",
            "charging_dQdV",
            "discharging_dQdV",
        ]

        X.columns = [y + "_" + x for x in quantities for y in
                     self.hyperparameters["statistics"]] + names
        self.features = X


class DeltaQFastCharge(BEEPFeaturizer):
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

        if not self.datapath.structured_summary.index.max() > self.hyperparameters["final_pred_cycle"]:
            return False, "Structured summary index max is less than final pred cycle"
        elif not self.datapath.structured_summary.index.min() <= self.hyperparameters["init_pred_cycle"]:
            return False, "Structured summary index min is more than initial pred cycle"
        elif "cycle_index" not in self.datapath.structured_summary.columns:
            return False, "Structured summary missing critical data: 'cycle_index'"
        elif "cycle_index" not in self.datapath.structured_data.columns:
            return False, "Structured data missing critical data: 'cycle_index'"
        elif not self.hyperparameters["mid_pred_cycle"] > 10:
            return False, "Middle pred. cycle less than threshold value of 10"
        elif not self.hyperparameters["final_pred_cycle"] > self.hyperparameters["mid_pred_cycle"]:
            return False, "Final pred cycle less than middle pred cycle"
        else:
            return True, None

    def create_features(self):
        """
        Generate features listed in early prediction manuscript, primarily related to the
        so called delta Q feature
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun): data from cycler run
            self.hyperparameters (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.

        Returns:
            pd.DataFrame: features indicative of degradation, derived from the input data
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


class TrajectoryFastCharge(DeltaQFastCharge):
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
                    df['fractional_metric'].diff().diff() < filter_kinks]['cycle_index'].min()
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
            interpolation_axes = interpolation_axes + ["real_regular_throughput"]

        threshold_dict = {
            'initial_regular_throughput':
                df['initial_regular_throughput'].values[0],
        }

        for indx, x_axis in enumerate(interpolation_axes):
            threshold_dict[
                cycle_type + metric + str(threshold) + '_' + x_axis] = [
                x_to_threshold[indx]]

        return pd.DataFrame(threshold_dict)
