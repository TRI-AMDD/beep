import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features.featurizer import BEEPEarlyCyclesFeaturizer


class CycleSummaryStats(BEEPEarlyCyclesFeaturizer):
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
