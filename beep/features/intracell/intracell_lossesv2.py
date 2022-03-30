import os
import numpy as np
import pandas as pd

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from beep.features.featurizer import BEEPPerCycleFeaturizer
from beep.features.intracell.intracell_analysisv2 import IntracellAnalysisV2

DEFAULT_CELL_INFO_DIR = os.path.join(PROTOCOL_PARAMETERS_DIR, "intracell_info")


class IntracellCyclesV2(BEEPPerCycleFeaturizer):
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
        'pe_pristine_files_dict':{},
        'pe_pristine_usecols':['Ecell/V','Capacity/mA.h',
                               'SOC_aligned','c_rate',
                               'BVV_c_rate','Voltage_aligned'],
        'ne_1_pristine_files_dict':{},
        'ne_1_pristine_usecols':['Ecell/V','Capacity/mA.h',
                                 'SOC_aligned','c_rate',
                                 'BVV_c_rate','Voltage_aligned'],
        'Q_fc_nom':4.84,
        'C_nom':-0.2,
        'cycle_type':'rpt_0.2C',
        'step_type': 1, 
        'error_type':'dVdQ',
        'error_weighting':'dQdV',
        'dvdq_bound':None,
        'ne_2pos_file':None,
        'ne_2neg_file':None
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
                (self.datapath.diagnostic_summary.cycle_type == self.hyperparameters["cycle_type"]) &
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
            pe_pristine_files_dict=self.hyperparameters["pe_pristine_files_dict"],
            pe_pristine_usecols=self.hyperparameters["pe_pristine_usecols"],
            ne_1_pristine_files_dict=self.hyperparameters["ne_1_pristine_files_dict"],
            ne_1_pristine_usecols=self.hyperparameters["ne_1_pristine_usecols"],
            Q_fc_nom=self.hyperparameters["Q_fc_nom"],
            C_nom=self.hyperparameters["C_nom"],
            cycle_type=self.hyperparameters["cycle_type"],
            step_type=self.hyperparameters["step_type"],
            error_type=self.hyperparameters["error_type"],
            error_weighting=self.hyperparameters["error_weighting"],
            dvdq_bound=self.hyperparameters["dvdq_bound"],
            ne_2pos_file=self.hyperparameters["ne_2pos_file"],
            ne_2neg_file=self.hyperparameters["ne_2neg_file"]
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
        for diag_pos, cycle_index in enumerate(eol_cycle_index_list):
            loss_dict, profiles_dict = ia.intracell_values_wrapper_ah(cycle_index,
                                                                      self.datapath
                                                                      )
            loss_dict[cycle_index] = np.append(diag_pos,loss_dict[cycle_index])
            dataset_dict_of_cell_degradation_path.update(loss_dict)
#             real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['diag_pos','rmse_error', 'LLI_opt', 'Q_pe_opt', 'Q_ne_opt', 'x_NE_2',
                                             'IR_coef_pe_opt','IR_coef_ne_opt',
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
        degradation_df = degradation_df.reset_index().rename(columns={'index':'cycle_index'})
        degradation_df['diag_pos'] = degradation_df['diag_pos'].astype(int)
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
            pe_pristine_dict=self.hyperparameters["pe_pristine_dict"],
            ne_1_pristine_dict=self.hyperparameters["ne_1_pristine_dict"],
            Q_fc_nom=self.hyperparameters["Q_fc_nom"],
            C_nom=self.hyperparameters["C_nom"],
            cycle_type=self.hyperparameters["cycle_type"],
            step_type=self.hyperparameters["step_type"],
            error_type=self.hyperparameters["error_type"],
            error_weighting=self.hyperparameters["error_weighting"],
            dvdq_bound=self.hyperparameters["dvdq_bound"],
            ne_2pos_file=self.hyperparameters["ne_2pos_file"],
            ne_2neg_file=self.hyperparameters["ne_2neg_file"]
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