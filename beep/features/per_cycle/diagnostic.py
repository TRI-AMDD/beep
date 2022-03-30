
import pandas as pd

from beep import PROTOCOL_PARAMETERS_DIR
from beep.features import featurizer_helpers
from functools import reduce

from beep.features.featurizer import BEEPPerCycleFeaturizer


class DiagnosticFeaturesPerCycle(BEEPPerCycleFeaturizer):
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
    """
    DEFAULT_HYPERPARAMETERS = {
        "parameters_dir": PROTOCOL_PARAMETERS_DIR,
        "nominal_capacity": 4.84,

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
            pd.DataFrame: cycle_index, RPT discharge capacities and energies, aging cycle discharge capacity and energy,
                            equivalent full cycles of aging cycle discharge, cumulative discharge throughput.
            for each diagnostic cycle of the cell
        """

        parameters_path = self.hyperparameters["parameters_dir"]

        # RPT discharge capacities
        data_rpt_02C = self.datapath.diagnostic_data.loc[
            self.datapath.diagnostic_data.cycle_type == 'rpt_0.2C']
        Q_rpt_02C = data_rpt_02C.groupby('cycle_index')[
            ['discharge_capacity', 'discharge_energy']].max().reset_index(
            drop=False)
        Q_rpt_02C.rename(
            columns={'discharge_capacity': 'rpt_0.2C_discharge_capacity',
                     'discharge_energy': 'rpt_0.2C_discharge_energy'},
            inplace=True)
        Q_rpt_02C = Q_rpt_02C.reset_index(drop=False).rename(
            columns={'index': 'diag_pos'})

        rpt_02C_cycles = data_rpt_02C.cycle_index.unique()  # for referencing last regular cycle before diagnostic

        data_rpt_1C = self.datapath.diagnostic_data.loc[
            self.datapath.diagnostic_data.cycle_type == 'rpt_1C']
        Q_rpt_1C = data_rpt_1C.groupby('cycle_index')[
            ['discharge_capacity', 'discharge_energy']].max().reset_index(
            drop=False)
        Q_rpt_1C.rename(
            columns={'discharge_capacity': 'rpt_1C_discharge_capacity',
                     'discharge_energy': 'rpt_1C_discharge_energy'},
            inplace=True)
        Q_rpt_1C = Q_rpt_1C.reset_index(drop=False).rename(
            columns={'index': 'diag_pos'})

        data_rpt_2C = self.datapath.diagnostic_data.loc[
            self.datapath.diagnostic_data.cycle_type == 'rpt_2C']
        Q_rpt_2C = data_rpt_2C.groupby('cycle_index')[
            ['discharge_capacity', 'discharge_energy']].max().reset_index(
            drop=False)
        Q_rpt_2C.rename(
            columns={'discharge_capacity': 'rpt_2C_discharge_capacity',
                     'discharge_energy': 'rpt_2C_discharge_energy'},
            inplace=True)
        Q_rpt_2C = Q_rpt_2C.reset_index(drop=False).rename(
            columns={'index': 'diag_pos'})

        # cumuative discharge throughput
        aging_df = self.datapath.structured_summary[
            ['cycle_index', 'charge_throughput', 'energy_throughput',
             'energy_efficiency', 'charge_duration', 'CV_time', 'CV_current',
             'energy_efficiency']]
        aging_df = aging_df.loc[aging_df.cycle_index.isin(rpt_02C_cycles - 3)]

        cumulative_discharge_throughput = aging_df[
            ['cycle_index', 'charge_throughput']].rename(
            columns={'charge_throughput': 'discharge_throughput'}).reset_index(
            drop=True)
        cumulative_discharge_throughput = cumulative_discharge_throughput.reset_index(
            drop=False).rename(columns={'index': 'diag_pos'})

        cumulative_energy_throughput = aging_df[
            ['cycle_index', 'energy_throughput']].reset_index(drop=True)
        cumulative_energy_throughput = cumulative_energy_throughput.reset_index(
            drop=False).rename(columns={'index': 'diag_pos'})

        equivalent_full_cycles = cumulative_discharge_throughput.copy()
        equivalent_full_cycles.rename(
            columns={'discharge_throughput': 'equivalent_full_cycles'},
            inplace=True)
        equivalent_full_cycles['equivalent_full_cycles'] = \
        equivalent_full_cycles['equivalent_full_cycles'] / self.hyperparameters[
            'nominal_capacity']

        # Q_aging_pre_diag - discharge capacity of aging cycle before diagnostic
        Q_aging_pre_diag = self.datapath.structured_data.groupby('cycle_index')[
            'discharge_capacity'].max().loc[rpt_02C_cycles[1:] - 3].reset_index(
            drop=False)  # ignore first diagnostic, adjust cycle index to Q_aging_pre_diag
        Q_aging_pre_diag.rename(
            columns={'discharge_capacity': 'Q_aging_pre_diag'}, inplace=True)
        Q_aging_pre_diag = Q_aging_pre_diag.reset_index(
            drop=False).rename(columns={'index': 'diag_pos'})
        Q_aging_pre_diag['diag_pos'] = Q_aging_pre_diag[
                                           'diag_pos'] + 1  # since, first diag is ignored, add one to diag_pos

        # Q_aging_post_diag - discharge capacity of aging cycle after diagnostic
        Q_aging_post_diag = \
        self.datapath.structured_data.groupby('cycle_index')[
            'discharge_capacity'].max().loc[rpt_02C_cycles + 3].reset_index(
            drop=False)  # does not ignore first diag since Q_aging exists after first diag
        Q_aging_post_diag.rename(
            columns={'discharge_capacity': 'Q_aging_post_diag'}, inplace=True)
        Q_aging_post_diag = Q_aging_post_diag.reset_index(
            drop=False).rename(columns={'index': 'diag_pos'})

        # Diagnostic time
        diagnostic_time = data_rpt_02C.groupby('cycle_index')[
            'test_time'].min().reset_index(drop=False).rename(
            columns={'test_time': 'diagnostic_time'})
        diagnostic_time = diagnostic_time.reset_index(
            drop=False).rename(columns={'index': 'diag_pos'})

        # Combine dataframes
        df_list = [Q_rpt_02C, Q_rpt_1C, Q_rpt_2C,
                   cumulative_discharge_throughput,
                   cumulative_energy_throughput,
                   equivalent_full_cycles,
                   Q_aging_pre_diag,
                   Q_aging_post_diag,
                   diagnostic_time]

        for df in df_list:
            df['cycle_index'] = df['cycle_index'].copy().astype(int)
            df['diag_pos'] = df['diag_pos'].copy().astype(int)

        cycle_features = reduce(
            lambda x, y: pd.merge(x, y, on=['cycle_index', 'diag_pos'],
                                  how='outer'), df_list)
        self.features = cycle_features.sort_values('cycle_index').reset_index(
            drop=True)



