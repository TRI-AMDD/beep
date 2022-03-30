
from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from beep.utils.parameters_lookup import get_protocol_parameters

from beep.features.featurizer import BEEPPerCycleFeaturizer


class CyclingProtocolPerCycle(BEEPPerCycleFeaturizer):
    """
    This class stores information about the charging protocol used
        name (str): predictor object name.
        X (pandas.DataFrame): features in DataFrame format.
        metadata (dict): information about the conditions, data
            and code used to produce features
    Hyperparameters:
        parameters_dir (str): Full path to directory of charging protocol parameters
        quantities ([str]): list of parameters to return
    """
    DEFAULT_HYPERPARAMETERS = {
        "parameters_dir": PROTOCOL_PARAMETERS_DIR,
        "quantities": ["charge_constant_current_1", "charge_constant_current_2",
                       "charge_cutoff_voltage", "charge_constant_voltage_time",
                       "discharge_constant_current",
                       "discharge_cutoff_voltage"],
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
        if not (
                'raw' in self.datapath.paths.keys() or 'structured' in self.datapath.paths.keys()):
            message = "datapath paths not set, unable to fetch charging protocol"
            return False, message
        else:
            return featurizer_helpers.check_diagnostic_validation(self.datapath)

    def create_features(self):
        """
        Fetches charging protocol features
        """

        parameters_path = self.hyperparameters["parameters_dir"]
        file_path = self.datapath.paths[
            'raw'] if 'raw' in self.datapath.paths.keys() else \
        self.datapath.paths['structured']

        parameters, _ = get_protocol_parameters(file_path, parameters_path)

        parameters = parameters[self.hyperparameters["quantities"]]
        parameters['cycle_index'] = int(
            0)  # create a cycle index column for merging with other featurizers
        parameters['diag_pos'] = int(
            0)  # create a diag_pos column for merging with other featurizers
        self.features = parameters