import os

import pandas as pd

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from beep.features.base import BEEPFeaturizer
from beep.features.intracell_analysis_v2 import IntracellAnalysisV2


DEFAULT_CELL_INFO_DIR = os.path.join(PROTOCOL_PARAMETERS_DIR, "intracell_info")


class IntracellCyclesV2(BEEPFeaturizer):
    """
    Object corresponding to the fitted material parameters of the cell. Material parameters
    are determined by using high resolution half cell data to fit full cell dQdV curves. Rows
    of the output dataframe correspond to each of the diagnostics throughout the life of the
    cell.
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    """

    DEFAULT_HYPERPARAMETERS = {
        "diagnostic_cycle_type": 'rpt_0.2C',
        "step_type": 0,
        # Paths for anode files should be absolute
        # Defaults are for the specified names in the current dir
        "anode_file": os.path.join(
            DEFAULT_CELL_INFO_DIR,
            'anode_test.csv'
        ),
        "cathode_file": os.path.join(
            DEFAULT_CELL_INFO_DIR,
            'cathode_test.csv'
        ),
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
            conditions = []

            # Ensure overlap of cycle indices above threshold and matching cycle type
            eol_cycle_index_list = self.datapath.diagnostic_summary[
                (self.datapath.diagnostic_summary.cycle_type == self.hyperparameters["diagnostic_cycle_type"]) &
                (self.datapath.diagnostic_summary.discharge_capacity > IntracellAnalysisV2.THRESHOLD)
                ].cycle_index.to_list()
            if not eol_cycle_index_list:
                return False, "Overlap of cycle indices not above threshold for matching cycle type"

            conditions.append(
                any(
                    [
                        "rpt" in x
                        for x in self.datapath.diagnostic_summary.cycle_type.unique()
                    ]
                )
            )

            if all(conditions):
                return True, None
            else:
                return False, "Insufficient RPT cycles in diagnostic"
        else:
            return val, msg

    def create_features(self):
        """
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            self.hyperparameters (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.
            cell_info_path (str): Root directory for cell half cell data

        Returns:
             (pd.DataFrame) containing the cell material parameters as a function of cycle index
        """
        ia = IntracellAnalysisV2(
            self.hyperparameters["cathode_file"],
            self.hyperparameters["anode_file"],
            cycle_type=self.hyperparameters["diagnostic_cycle_type"],
            step_type=self.hyperparameters["step_type"]
        )

        # (cell_init_aligned, cell_init_profile, PE_matched, NE_matched) = ia.intracell_wrapper_init(
        #     self.datapath,
        # )

        eol_cycle_index_list = self.datapath.diagnostic_summary[
            (self.datapath.diagnostic_summary.cycle_type == ia.cycle_type) &
            (self.datapath.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()

        # initialize dicts before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list):
            loss_dict, profiles_dict = ia.intracell_values_wrapper_ah(cycle_index,
                                                                      self.datapath
                                                                      )
            dataset_dict_of_cell_degradation_path.update(loss_dict)
            real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['rmse_error', 'LLI_opt', 'Q_pe_opt', 'Q_ne_opt', 'x_NE_2',
                                             'pe_voltage_FC4p2V', 'pe_voltage_FC4p1V', 'pe_voltage_FC4p0V',
                                             'pe_voltage_FC3p9V', 'pe_voltage_FC3p8V', 'pe_voltage_FC3p7V',
                                             'pe_voltage_FC3p6V', 'pe_voltage_FC3p5V', 'pe_voltage_FC3p4V',
                                             'pe_voltage_FC3p3V', 'pe_voltage_FC3p2V', 'pe_voltage_FC3p1V',
                                             'pe_voltage_FC3p0V', 'pe_voltage_FC2p9V', 'pe_voltage_FC2p8V',
                                             'pe_voltage_FC2p7V',
                                             'pe_soc_FC4p2V', 'pe_soc_FC4p1V', 'pe_soc_FC4p0V', 'pe_soc_FC3p9V',
                                             'pe_soc_FC3p8V', 'pe_soc_FC3p7V',
                                             'pe_soc_FC3p6V', 'pe_soc_FC3p5V', 'pe_soc_FC3p4V', 'pe_soc_FC3p3V',
                                             'pe_soc_FC3p2V', 'pe_soc_FC3p1V',
                                             'pe_soc_FC3p0V', 'pe_soc_FC2p9V', 'pe_soc_FC2p8V', 'pe_soc_FC2p7V',
                                             'ne_voltage_FC4p2V', 'ne_voltage_FC4p1V', 'ne_voltage_FC4p0V',
                                             'ne_voltage_FC3p9V', 'ne_voltage_FC3p8V', 'ne_voltage_FC3p7V',
                                             'ne_voltage_FC3p6V', 'ne_voltage_FC3p5V', 'ne_voltage_FC3p4V',
                                             'ne_voltage_FC3p3V', 'ne_voltage_FC3p2V', 'ne_voltage_FC3p1V',
                                             'ne_voltage_FC3p0V', 'ne_voltage_FC2p9V', 'ne_voltage_FC2p8V',
                                             'ne_voltage_FC2p7V',
                                             'ne_soc_FC4p2V', 'ne_soc_FC4p1V', 'ne_soc_FC4p0V', 'ne_soc_FC3p9V',
                                             'ne_soc_FC3p8V', 'ne_soc_FC3p7V',
                                             'ne_soc_FC3p6V', 'ne_soc_FC3p5V', 'ne_soc_FC3p4V', 'ne_soc_FC3p3V',
                                             'ne_soc_FC3p2V', 'ne_soc_FC3p1V',
                                             'ne_soc_FC3p0V', 'ne_soc_FC2p9V', 'ne_soc_FC2p8V', 'ne_soc_FC2p7V',
                                             'Q_fc', 'Q_pe', 'Q_ne', 'Q_li']).T
        self.features = degradation_df


class IntracellFeaturesV2(IntracellCyclesV2):
    """
    Object corresponding to the fitted material parameters of the cell. Material parameters
    are determined by using high resolution half cell data to fit full cell dQdV curves. The
    parameters from the first and second diagnostics are used as the feature values.
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    """

    def create_features(self):
        """
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.
            cell_info_path (str): Root directory for cell half cell data


        Returns:
             (pd.DataFrame) containing the cell material parameters for the first and second diagnotics
                as a single row dataframe
        """

        ia = IntracellAnalysisV2(
            self.hyperparameters["cathode_file"],
            self.hyperparameters["anode_file"],
            cycle_type=self.hyperparameters["diagnostic_cycle_type"],
            step_type=self.hyperparameters["step_type"]
        )

        # (cell_init_aligned, cell_init_profile, PE_matched, NE_matched) = ia.intracell_wrapper_init(
        #     self.datapath,
        # )

        eol_cycle_index_list = self.datapath.diagnostic_summary[
            (self.datapath.diagnostic_summary.cycle_type == ia.cycle_type) &
            (self.datapath.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()

        # # initializations before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list[0:2]):
            loss_dict, profiles_dict = ia.intracell_values_wrapper_ah(cycle_index,
                                                                      self.datapath
                                                                      )
            dataset_dict_of_cell_degradation_path.update(loss_dict)
            real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['rmse_error', 'LLI_opt', 'Q_pe_opt', 'Q_ne_opt', 'x_NE_2',
                                             'pe_voltage_FC4p2V', 'pe_voltage_FC4p1V', 'pe_voltage_FC4p0V',
                                             'pe_voltage_FC3p9V', 'pe_voltage_FC3p8V', 'pe_voltage_FC3p7V',
                                             'pe_voltage_FC3p6V', 'pe_voltage_FC3p5V', 'pe_voltage_FC3p4V',
                                             'pe_voltage_FC3p3V', 'pe_voltage_FC3p2V', 'pe_voltage_FC3p1V',
                                             'pe_voltage_FC3p0V', 'pe_voltage_FC2p9V', 'pe_voltage_FC2p8V',
                                             'pe_voltage_FC2p7V',
                                             'pe_soc_FC4p2V', 'pe_soc_FC4p1V', 'pe_soc_FC4p0V', 'pe_soc_FC3p9V',
                                             'pe_soc_FC3p8V', 'pe_soc_FC3p7V',
                                             'pe_soc_FC3p6V', 'pe_soc_FC3p5V', 'pe_soc_FC3p4V', 'pe_soc_FC3p3V',
                                             'pe_soc_FC3p2V', 'pe_soc_FC3p1V',
                                             'pe_soc_FC3p0V', 'pe_soc_FC2p9V', 'pe_soc_FC2p8V', 'pe_soc_FC2p7V',
                                             'ne_voltage_FC4p2V', 'ne_voltage_FC4p1V', 'ne_voltage_FC4p0V',
                                             'ne_voltage_FC3p9V', 'ne_voltage_FC3p8V', 'ne_voltage_FC3p7V',
                                             'ne_voltage_FC3p6V', 'ne_voltage_FC3p5V', 'ne_voltage_FC3p4V',
                                             'ne_voltage_FC3p3V', 'ne_voltage_FC3p2V', 'ne_voltage_FC3p1V',
                                             'ne_voltage_FC3p0V', 'ne_voltage_FC2p9V', 'ne_voltage_FC2p8V',
                                             'ne_voltage_FC2p7V',
                                             'ne_soc_FC4p2V', 'ne_soc_FC4p1V', 'ne_soc_FC4p0V', 'ne_soc_FC3p9V',
                                             'ne_soc_FC3p8V', 'ne_soc_FC3p7V',
                                             'ne_soc_FC3p6V', 'ne_soc_FC3p5V', 'ne_soc_FC3p4V', 'ne_soc_FC3p3V',
                                             'ne_soc_FC3p2V', 'ne_soc_FC3p1V',
                                             'ne_soc_FC3p0V', 'ne_soc_FC2p9V', 'ne_soc_FC2p8V', 'ne_soc_FC2p7V',
                                             'Q_fc', 'Q_pe', 'Q_ne', 'Q_li']).T

        diag_0_names = ["diag_0_" + name for name in degradation_df.columns]
        diag_1_names = ["diag_1_" + name for name in degradation_df.columns]
        values = {0: degradation_df.iloc[0].tolist() + degradation_df.iloc[1].tolist()}
        features_df = pd.DataFrame(values, index=diag_0_names+diag_1_names).T
        self.features = features_df
