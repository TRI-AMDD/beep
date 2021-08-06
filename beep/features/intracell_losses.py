import os
import pandas as pd
from monty.serialization import loadfn

from beep.features.base import BeepFeatures
from beep.features.intracell_analysis import IntracellAnalysis
from beep import MODULE_DIR


FEATURE_HYPERPARAMS = loadfn(
    os.path.join(MODULE_DIR, "features/feature_hyperparameters.yaml")
)

s = {"service": "DataAnalyzer"}


class IntracellCycles(BeepFeatures):
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

    # Class name for the feature object
    class_feature_name = "IntracellCycles"

    def __init__(self, name, X, metadata):
        """
        Args:
            name (str): predictor object name
            X (pandas.DataFrame): features in DataFrame format.
            metadata (dict): information about the data and code used to produce features
        """
        super().__init__(name, X, metadata)
        self.name = name
        self.X = X
        self.metadata = metadata

    @classmethod
    def validate_data(cls, processed_cycler_run, params_dict=None):
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
        conditions = []

        if not hasattr(processed_cycler_run, "diagnostic_summary") & hasattr(
            processed_cycler_run, "diagnostic_data"
        ):
            return False
        if processed_cycler_run.diagnostic_summary is None:
            return False
        elif processed_cycler_run.diagnostic_summary.empty:
            return False
        else:
            # Ensure overlap of cycle indices above threshold and matching cycle type
            eol_cycle_index_list = processed_cycler_run.diagnostic_summary[
                (processed_cycler_run.diagnostic_summary.cycle_type == params_dict["diagnostic_cycle_type"]) &
                (processed_cycler_run.diagnostic_summary.discharge_capacity > IntracellAnalysis.THRESHOLD)
                ].cycle_index.to_list()
            if not eol_cycle_index_list:
                return False

            conditions.append(
                any(
                    [
                        "rpt" in x
                        for x in processed_cycler_run.diagnostic_summary.cycle_type.unique()
                    ]
                )
            )

        return all(conditions)

    @classmethod
    def features_from_processed_cycler_run(cls, processed_cycler_run, params_dict=None,
                                           parameters_path="data-share/raw/parameters",
                                           cell_info_path="data-share/raw/cell_info"):
        """
        Args:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            params_dict (dict): dictionary of parameters governing how the ProcessedCyclerRun object
                gets featurized. These could be filters for column or row operations
            parameters_path (str): Root directory storing project parameter files.
            cell_info_path (str): Root directory for cell half cell data

        Returns:
             (pd.DataFrame) containing the cell material parameters as a function of cycle index
        """
        if params_dict is None:
            params_dict = FEATURE_HYPERPARAMS[cls.class_feature_name]

        cell_dir = os.path.join(
            os.environ.get("BEEP_PROCESSING_DIR", "/"), cell_info_path
        )
        ia = IntracellAnalysis(os.path.join(cell_dir, params_dict["cathode_file"]),
                               os.path.join(cell_dir, params_dict["anode_file"]),
                               cycle_type=params_dict["diagnostic_cycle_type"],
                               step_type=params_dict["step_type"]
                               )

        (cell_init_aligned, cell_init_profile, PE_matched, NE_matched) = ia.intracell_wrapper_init(
            processed_cycler_run,
        )

        eol_cycle_index_list = processed_cycler_run.diagnostic_summary[
            (processed_cycler_run.diagnostic_summary.cycle_type == ia.cycle_type) &
            (processed_cycler_run.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()

        # initialize dicts before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list):
            loss_dict, profiles_dict = ia.intracell_values_wrapper(cycle_index,
                                                                   processed_cycler_run,
                                                                   cell_init_aligned,
                                                                   cell_init_profile,
                                                                   PE_matched,
                                                                   NE_matched,
                                                                   )
            dataset_dict_of_cell_degradation_path.update(loss_dict)
            real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['LLI', 'LAM_PE', 'LAM_NE', 'x_NE_2', 'alpha_real', 'alpha_emulated',
                                             'PE_upper_voltage', 'PE_lower_voltage', 'PE_upper_SOC', 'PE_lower_SOC',
                                             'PE_mass', 'NE_upper_voltage', 'NE_lower_voltage', 'NE_upper_SOC',
                                             'NE_lower_SOC', 'NE_mass', 'Li_mass'
                                             ]).T
        return degradation_df


class IntracellFeatures(IntracellCycles):
    """
    Object corresponding to the fitted material parameters of the cell. Material parameters
    are determined by using high resolution half cell data to fit full cell dQdV curves. The
    parameters from the first and second diagnostics are used as the feature values.
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    """

    # Class name for the feature object
    class_feature_name = "IntracellFeatures"

    @classmethod
    def features_from_processed_cycler_run(cls, processed_cycler_run, params_dict=None,
                                           parameters_path="data-share/raw/parameters",
                                           cell_info_path="data-share/raw/cell_info"):
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
        if params_dict is None:
            params_dict = FEATURE_HYPERPARAMS[cls.class_feature_name]

        cell_dir = os.path.join(
            os.environ.get("BEEP_PROCESSING_DIR", "/"), cell_info_path
        )
        ia = IntracellAnalysis(os.path.join(cell_dir, params_dict["cathode_file"]),
                               os.path.join(cell_dir, params_dict["anode_file"]),
                               cycle_type=params_dict["diagnostic_cycle_type"],
                               step_type=params_dict["step_type"]
                               )

        (cell_init_aligned, cell_init_profile, PE_matched, NE_matched) = ia.intracell_wrapper_init(
            processed_cycler_run,
        )

        eol_cycle_index_list = processed_cycler_run.diagnostic_summary[
            (processed_cycler_run.diagnostic_summary.cycle_type == ia.cycle_type) &
            (processed_cycler_run.diagnostic_summary.discharge_capacity > ia.THRESHOLD)
            ].cycle_index.to_list()

        # # initializations before for loop
        dataset_dict_of_cell_degradation_path = dict()
        real_cell_dict_of_profiles = dict()
        for i, cycle_index in enumerate(eol_cycle_index_list[0:2]):
            loss_dict, profiles_dict = ia.intracell_values_wrapper(cycle_index,
                                                                   processed_cycler_run,
                                                                   cell_init_aligned,
                                                                   cell_init_profile,
                                                                   PE_matched,
                                                                   NE_matched,
                                                                   )
            dataset_dict_of_cell_degradation_path.update(loss_dict)
            real_cell_dict_of_profiles.update(profiles_dict)

        degradation_df = pd.DataFrame(dataset_dict_of_cell_degradation_path,
                                      index=['LLI', 'LAM_PE', 'LAM_NE', 'x_NE_2', 'alpha_real', 'alpha_emulated',
                                             'PE_upper_voltage', 'PE_lower_voltage', 'PE_upper_SOC', 'PE_lower_SOC',
                                             'PE_mass', 'NE_upper_voltage', 'NE_lower_voltage', 'NE_upper_SOC',
                                             'NE_lower_SOC', 'NE_mass', 'Li_mass'
                                             ]).T
        diag_0_names = ["diag_0_" + name for name in degradation_df.columns]
        diag_1_names = ["diag_1_" + name for name in degradation_df.columns]
        values = {0: degradation_df.iloc[0].tolist() + degradation_df.iloc[1].tolist()}
        features_df = pd.DataFrame(values, index=diag_0_names+diag_1_names).T
        return features_df
