import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.optimize import differential_evolution


class IntracellAnalysis:
    # IA constants
    UPPER_VOLTAGE = 4.2
    LOWER_VOLTAGE = 2.7
    THRESHOLD = 4.84 * 0.7

    def __init__(self,
                 pe_pristine_file,
                 ne_pristine_file,
                 cycle_type='rpt_0.2C',
                 step_type=0,
                 ne_2pos_file=None,
                 ne_2neg_file=None
                 ):
        """
        Invokes the cell electrode analysis class. This is a class designed to fit the cell and electrode
        parameters in order to determine changes of electrodes within the full cell from only full cell cycling data.
        Args:
            pe_pristine_file (str): file name for the half cell data of the pristine (uncycled) positive
                electrode
            ne_pristine_file (str): file name for the half cell data of the pristine (uncycled) negative
                electrode
            cycle_type (str): type of diagnostic cycle for the fitting
            step_type (int): charge or discharge (0 for charge, 1 for discharge)
            ne_2neg_file (str): file name of the data for the negative component of the anode
            ne_2pos_file (str): file name of the data for the positive component of the anode
        """
        self.pe_pristine = pd.read_csv(pe_pristine_file, usecols=['SOC_aligned', 'Voltage_aligned'])
        self.ne_1_pristine = pd.read_csv(ne_pristine_file, usecols=['SOC_aligned', 'Voltage_aligned'])

        if ne_2neg_file and ne_2pos_file:
            self.ne_2_pristine_pos = pd.read_csv(ne_2pos_file)
            self.ne_2_pristine_neg = pd.read_csv(ne_2neg_file)
        else:
            self.ne_2_pristine_pos = pd.DataFrame()
            self.ne_2_pristine_neg = pd.DataFrame()

        self.cycle_type = cycle_type
        self.step_type = step_type

    def process_beep_cycle_data_for_candidate_halfcell_analysis(self,
                                                                cell_struct,
                                                                real_cell_initial_charge_profile_aligned,
                                                                real_cell_initial_charge_profile,
                                                                cycle_index):
        """
        Inputs:
        diag_type_cycles: beep cell_struct.diagnostic_interpolated filtered to one diagnostic type
        real_cell_initial_charge_profile_aligned: dataframe containing SOC (equally spaced) and voltage columns
        real_cell_initial_charge_profile: dataframe containing SOC and voltage columns
        cycle_index: cycle number to evaluate at
        Outputs
        real_cell_candidate_charge_profile_aligned: a dataframe containing columns SOC_aligned (evenly spaced) and
        Voltage_aligned
        """

        diag_type_cycles = cell_struct.diagnostic_data.loc[cell_struct.diagnostic_data['cycle_type'] == self.cycle_type]
        real_cell_candidate_charge_profile = diag_type_cycles.loc[
            (diag_type_cycles.cycle_index == cycle_index)
            & (diag_type_cycles.step_type == 0)  # step_type = 0 is charge, 1 is discharge
            & (diag_type_cycles.voltage < self.UPPER_VOLTAGE)
            & (diag_type_cycles[self.capacity_col] > 0)][['voltage', 'charge_capacity']]

        real_cell_candidate_charge_profile['SOC'] = (
                (real_cell_candidate_charge_profile['charge_capacity'] -
                 np.min(real_cell_initial_charge_profile['charge_capacity']))
                / (np.max(real_cell_initial_charge_profile['charge_capacity']) -
                   np.min(real_cell_initial_charge_profile['charge_capacity'])) * 100
                                                     )
        real_cell_candidate_charge_profile['Voltage'] = real_cell_candidate_charge_profile['voltage']
        real_cell_candidate_charge_profile.drop('voltage', axis=1, inplace=True)

        SOC_vec = np.linspace(0, np.max(real_cell_candidate_charge_profile['SOC']),
                              1001)  # 100 ; np.max(real_cell_candidate_charge_profile['SOC']

        real_cell_candidate_charge_profile_aligned = pd.DataFrame()
        real_cell_candidate_charge_profile_interper = interp1d(real_cell_candidate_charge_profile['SOC'],
                                                               real_cell_candidate_charge_profile['Voltage'],
                                                               bounds_error=False,
                                                               fill_value=(self.LOWER_VOLTAGE, self.UPPER_VOLTAGE))
        real_cell_candidate_charge_profile_aligned['Voltage_aligned'] = real_cell_candidate_charge_profile_interper(
            SOC_vec)

        # real_cell_candidate_charge_profile_aligned['Voltage_aligned'].fillna(self.LOWER_VOLTAGE, inplace=True)
        real_cell_candidate_charge_profile_aligned['SOC_aligned'] = SOC_vec / np.max(
            real_cell_initial_charge_profile_aligned['SOC_aligned'].loc[
                ~real_cell_initial_charge_profile_aligned['Voltage_aligned'].isna()]) * 100

        return real_cell_candidate_charge_profile_aligned

    def process_beep_cycle_data_for_initial_halfcell_analysis(self,
                                                              cell_struct,
                                                              step_type=0):
        """
        This function extracts the initial (non-degraded) voltage and soc profiles for the cell with columns
        interpolated on voltage and soc.
        Inputs
        cell_struct: beep cell_struct.diagnostic_interpolated filtered to one diagnostic type
        step_type: specifies whether the cell is charging or discharging. 0 is charge, 1 is discharge.
        Outputs
        real_cell_initial_charge_profile_aligned: a dataframe containing columns SOC_aligned (evenly spaced)
            and Voltage_aligned
        real_cell_initial_charge_profile: a dataframe containing columns Voltage (evenly spaced), capacity, and SOC
        """
        if step_type == 0:
            self.capacity_col = 'charge_capacity'
        else:
            self.capacity_col = 'discharge_capacity'

        diag_type_cycles = cell_struct.diagnostic_data.loc[cell_struct.diagnostic_data['cycle_type'] == self.cycle_type]
        soc_vec = np.linspace(0, 100.0, 1001)
        cycle_index_of_cycle_type = cell_struct.diagnostic_summary[
            cell_struct.diagnostic_summary.cycle_type == self.cycle_type].cycle_index.iloc[0]
        real_cell_initial_charge_profile = diag_type_cycles.loc[
            (diag_type_cycles.cycle_index == cycle_index_of_cycle_type)
            & (diag_type_cycles.step_type == step_type)  # step_type = 0 is charge, 1 is discharge
            & (diag_type_cycles.voltage < self.UPPER_VOLTAGE)
            & (diag_type_cycles.voltage > self.LOWER_VOLTAGE)][['voltage', self.capacity_col]]

        real_cell_initial_charge_profile['SOC'] = (
                (
                        real_cell_initial_charge_profile[self.capacity_col] -
                        np.min(real_cell_initial_charge_profile[self.capacity_col])
                ) /
                (
                       np.max(real_cell_initial_charge_profile[self.capacity_col]) -
                       np.min(real_cell_initial_charge_profile[self.capacity_col])
                ) * 100
                                                   )
        real_cell_initial_charge_profile['Voltage'] = real_cell_initial_charge_profile['voltage']
        real_cell_initial_charge_profile.drop('voltage', axis=1, inplace=True)

        real_cell_initial_charge_profile_aligned = pd.DataFrame()
        real_cell_initial_charge_profile_aligned['SOC_aligned'] = soc_vec
        real_cell_initial_charge_profile_interper = interp1d(real_cell_initial_charge_profile['SOC'],
                                                             real_cell_initial_charge_profile['Voltage'],
                                                             bounds_error=False)

        real_cell_initial_charge_profile_aligned['Voltage_aligned'] = real_cell_initial_charge_profile_interper(
            real_cell_initial_charge_profile_aligned['SOC_aligned'])

        return real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile

    def get_dQdV_over_Q_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        # Calculate dQdV from full cell profiles
        dq_dv_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                  columns=['dQdV'])
        dq_dv_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV'])

        # Include original data
        dq_dv_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dq_dv_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dq_dv_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dq_dv_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return df_1, df_2, dq_dv_real, dq_dv_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dQdV_over_Q_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, dQdV_real, dQdV_emulated, df_real_interped, emulated_full_cell_interped = \
            self.get_dQdV_over_Q_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dQdV_real['dQdV'], dQdV_emulated['dQdV']) + 0.01 * len(
            dQdV_emulated['dQdV'].loc[dQdV_emulated['dQdV'].isna()])

        return error

    def get_dQdV_over_V_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        # Calculate dQdV from full cell profiles
        dq_dv_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                  columns=['dQdV']).ewm(alpha=x[-2]).mean()
        dq_dv_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV']).ewm(alpha=x[-1]).mean()

        # Include original data
        dq_dv_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dq_dv_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dq_dv_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dq_dv_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)

        v_dq_dv_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         dq_dv_real['dQdV'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         bounds_error=False, fill_value=0)
        v_soc_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       dq_dv_real['SOC_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        v_dq_dv_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                                ~dq_dv_emulated['Voltage_aligned'].isna()],
                                             dq_dv_emulated['dQdV'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                             bounds_error=False, fill_value=0)
        v_soc_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                               ~dq_dv_emulated['Voltage_aligned'].isna()],
                                           dq_dv_emulated['SOC_aligned'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dq_dv_over_v_real = pd.DataFrame(v_dq_dv_interper_real(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_real['SOC'] = v_soc_interper_real(voltage_vec)
        dq_dv_over_v_real['Voltage'] = voltage_vec

        dq_dv_over_v_emulated = pd.DataFrame(v_dq_dv_interper_emulated(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_emulated['SOC'] = v_soc_interper_emulated(voltage_vec)
        dq_dv_over_v_emulated['Voltage'] = voltage_vec

        return df_1, df_2, dq_dv_over_v_real, dq_dv_over_v_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dQdV_over_V_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, dq_dv_real, dq_dv_emulated, df_real_interped, emulated_full_cell_interped = \
            self.get_dQdV_over_V_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dq_dv_real['dQdV'], dq_dv_emulated['dQdV']) + 0.01 * len(
            dq_dv_emulated['dQdV'].loc[dq_dv_emulated['dQdV'].isna()])
        return error

    def get_dVdQ_over_Q_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        # Calculate dVdQ from full cell profiles
        dv_dq_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'], df_real_interped['SOC_aligned']),
                                  columns=['dVdQ'])
        dv_dq_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['SOC_aligned']),
            columns=['dVdQ'])

        # Include original data
        dv_dq_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dv_dq_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dv_dq_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dv_dq_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return df_1, df_2, dv_dq_real, dv_dq_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dVdQ_over_Q_from_halfcell_initial_matching(self, x, *params):
        (df_1,
         df_2,
         dVdQ_real,
         dVdQ_emulated,
         df_real_interped,
         emulated_full_cell_interped) = self.get_dVdQ_over_Q_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dVdQ_real['dVdQ'], dVdQ_emulated['dVdQ']) + 0.01 * len(
            dVdQ_emulated['dVdQ'].loc[dVdQ_emulated['dVdQ'].isna()])

        return error

    def get_dVdQ_over_V_from_halfcell_initial_matching(self, x, *params):
        (df_1,
         df_2,
         df_real_interped,
         emulated_full_cell_interped) = self.halfcell_initial_matching_v2(x, *params)

        # Calculate dVdQ from full cell profiles
        dv_dq_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                  columns=['dVdQ'])
        dv_dq_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dVdQ'])

        # Include original data
        dv_dq_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dv_dq_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dv_dq_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dv_dq_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)
        v_dv_dq_interper_real = interp1d(dv_dq_real['Voltage_aligned'].loc[~dv_dq_real['Voltage_aligned'].isna()],
                                         dv_dq_real['dVdQ'].loc[~dv_dq_real['Voltage_aligned'].isna()],
                                         bounds_error=False, fill_value=0)
        v_soc_interper_real = interp1d(dv_dq_real['Voltage_aligned'].loc[~dv_dq_real['Voltage_aligned'].isna()],
                                       dv_dq_real['SOC_aligned'].loc[~dv_dq_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        v_dv_dq_interp_emulated = interp1d(dv_dq_emulated['Voltage_aligned'].loc[
                                              ~dv_dq_emulated['Voltage_aligned'].isna()],
                                           dv_dq_emulated['dVdQ'].loc[~dv_dq_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=0)
        v_soc_interper_emulated = interp1d(dv_dq_emulated['Voltage_aligned'].loc[
                                               ~dv_dq_emulated['Voltage_aligned'].isna()],
                                           dv_dq_emulated['SOC_aligned'].loc[~dv_dq_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dv_dq_over_v_real = pd.DataFrame(v_dv_dq_interper_real(voltage_vec), columns=['dVdQ'])
        dv_dq_over_v_real['SOC'] = v_soc_interper_real(voltage_vec)
        dv_dq_over_v_real['Voltage'] = voltage_vec

        dv_dq_over_v_emulated = pd.DataFrame(v_dv_dq_interp_emulated(voltage_vec), columns=['dVdQ'])
        dv_dq_over_v_emulated['SOC'] = v_soc_interper_emulated(voltage_vec)
        dv_dq_over_v_emulated['Voltage'] = voltage_vec

        return df_1, df_2, dv_dq_over_v_real, dv_dq_over_v_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dVdQ_over_V_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, dv_dq_real, dv_dq_emulated, df_real_interped, emulated_full_cell_interped = \
            self.get_dVdQ_over_V_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dv_dq_real['dVdQ'], dv_dq_emulated['dVdQ']) + 0.01 * len(
            dv_dq_emulated['dVdQ'].loc[dv_dq_emulated['dVdQ'].isna()])
        return error

    def blend_electrodes_robust(self, ne_pristine_matched, ne_2_pos, ne_2_neg, x_ne_2):
        """
        """
        if ne_2_pos.empty:
            df_blended = ne_pristine_matched
            return df_blended

        if ne_2_neg.empty:
            df_blended = ne_pristine_matched
            return df_blended

        if x_ne_2 > 0:
            ne_2_pristine = ne_2_pos
        else:
            ne_2_pristine = ne_2_neg
            x_ne_2 = np.abs(x_ne_2)

        # match the two NE materials by SOC
        ne_pristine_matched_interper = interp1d(
            ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()]['SOC_aligned'],
            ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()]['Voltage_aligned'],
            bounds_error=False)
        ne_2_pristine_interper = interp1d(ne_2_pristine.loc[~ne_2_pristine.Voltage_aligned.isna()]['SOC_aligned'],
                                          ne_2_pristine.loc[~ne_2_pristine.Voltage_aligned.isna()]['Voltage_aligned'],
                                          bounds_error=False)

        soc_vec = np.linspace(0, 100, 1001)

        ne_pristine_matched_len1001 = pd.DataFrame(soc_vec, columns=['SOC_aligned'])
        ne_pristine_matched_len1001['Voltage_aligned'] = ne_pristine_matched_interper(soc_vec)

        ne_2_pristine_interper_len1001 = pd.DataFrame(soc_vec, columns=['SOC_aligned'])
        ne_2_pristine_interper_len1001['Voltage_aligned'] = ne_2_pristine_interper(soc_vec)

        df_ne = blend_electrodes(ne_pristine_matched_len1001,
                                 ne_2_pristine_interper_len1001,
                                 pd.DataFrame(),
                                 x_ne_2)

        return df_ne

    def blend_electrodes_robust_v2(self, ne_pristine_matched, ne_2_pos, ne_2_neg, x_ne_2):
        """
        """
        if ne_2_pos.empty:
            df_blended = ne_pristine_matched
            return df_blended

        if ne_2_neg.empty:
            df_blended = ne_pristine_matched
            return df_blended

        if x_ne_2 > 0:
            ne_2_pristine = ne_2_pos
        else:
            ne_2_pristine = ne_2_neg
            x_ne_2 = np.abs(x_ne_2)

        # match the two NE materials by SOC
        soc_vec = np.linspace(0, 100, 1001)

        ne_pristine_matched_0to100 = ((ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()][
                                           'SOC_aligned'] - np.min(
            ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()]['SOC_aligned'])) /
                                      (np.max(ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()][
                                                  'SOC_aligned']) - np.min(
                                          ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()][
                                              'SOC_aligned']))
                                      ) * 100
        ne_pristine_matched_interper = interp1d(ne_pristine_matched_0to100,
                                                ne_pristine_matched.loc[~ne_pristine_matched.Voltage_aligned.isna()][
                                                    'Voltage_aligned'],
                                                bounds_error=False)

        ne_2_pristine_interper = interp1d(ne_2_pristine.loc[~ne_2_pristine.Voltage_aligned.isna()]['SOC_aligned'],
                                          ne_2_pristine.loc[~ne_2_pristine.Voltage_aligned.isna()]['Voltage_aligned'],
                                          bounds_error=False)

        ne_pristine_matched_len1001 = pd.DataFrame(soc_vec, columns=['SOC_aligned'])
        ne_pristine_matched_len1001['Voltage_aligned'] = ne_pristine_matched_interper(soc_vec)

        ne_2_pristine_interper_len1001 = pd.DataFrame(soc_vec, columns=['SOC_aligned'])
        ne_2_pristine_interper_len1001['Voltage_aligned'] = ne_2_pristine_interper(soc_vec)

        df_ne_blended = blend_electrodes(ne_pristine_matched_len1001,
                                         ne_2_pristine_interper_len1001,
                                         pd.DataFrame(),
                                         x_ne_2)

        # restore blend back to original SOC span of NE_pristine_matched

        df_ne_blended_interper = interp1d(df_ne_blended['SOC_aligned'], df_ne_blended['Voltage_aligned'],
                                          bounds_error=False)  # intiializing interpolation across the blend

        len_non_na_ne_pristine_matched = len(ne_pristine_matched.loc[
                                                ~ne_pristine_matched.Voltage_aligned.isna()])  # number of points
        # applicable to NE in NE_pristine_matched

        soc_vec_prematching = np.linspace(np.min(df_ne_blended['SOC_aligned']),
                                          np.max(df_ne_blended['SOC_aligned']),
                                          len_non_na_ne_pristine_matched)  # vector across blended NE with same number
        # of applicable points as original (NE_pristine_matched)

        df_ne_blended_matched = ne_pristine_matched.copy()

        df_ne_slice_for_matched = df_ne_blended_interper(soc_vec_prematching)
        df_ne_blended_matched.at[(df_ne_blended_matched.loc[
            (~df_ne_blended_matched['Voltage_aligned'].isna())]).index, 'Voltage_aligned'] = df_ne_slice_for_matched

        return df_ne_blended_matched

    def halfcell_initial_matching_v2(self, x, *params):
        """
        Augments halfcell voltage profiles by scaling and translating them. Typically used in an optimization routine
        to fit the emulated full cell profile to a real cell profile. Alternatively, this function can be used for
        emulation of full cell voltage profiles from its electrode constituents with specified capacity ratio and
        offset of the two electrodes.
        Inputs:
        x: an array of 2 or 3 parameters containing scale_ratio, offset, and optionally NE_2_x. scale_ratio is equal
            to the capacity of the
        cathode divided by the capacity of the anode. offset is defined as the SOC between the cathode at zero capacity
            and the anode at zero
        capacity. NE_2_x is the fraction of the secondary electrode material in the anode.
        df_real: dataframe for the first diagnostic (pristine) of the real full cell. Columns for SOC (evenly spaced)
            and Voltage.
        df_pe: dataframe for the positive electrode. Columns for SOC (evenly spaced) and Voltage.
        self.ne_1_pristine: dataframe for the primary material in the negative electrode. Columns for SOC
            (evenly spaced) and Voltage.
        df_ne_2: dataframe for the secondary material in the negative electrode. Columns for SOC (evenly spaced)
            and Voltage. Supply empty DataFrame if not emulating a blend from two known elelctrodes.
        """

        df_real, df_pe, df_ne_1, df_ne_2_pos, df_ne_2_neg = params
        df_pe = self.pe_pristine
        df_ne_1 = self.ne_1_pristine
        df_ne_2_pos = self.ne_2_pristine_pos
        df_ne_2_neg = self.ne_2_pristine_neg
        scale_ratio = x[0]
        offset = x[1]

        if df_ne_2_pos.empty | df_ne_2_neg.empty:
            # one-material anode
            df_ne = pd.DataFrame()
            df_ne['Voltage_aligned'] = df_ne_1['Voltage_aligned']
            df_ne['SOC_aligned'] = df_ne_1['SOC_aligned']
        else:
            # blended anode
            x_ne_2 = x[2]
            df_ne = blend_electrodes(df_ne_1, df_ne_2_pos, df_ne_2_neg, x_ne_2)  # _robust_v2

        # shifted cathode
        shifted_pe = df_pe.copy()
        shifted_pe['SOC_aligned'] = shifted_pe['SOC_aligned'] * scale_ratio + offset

        # shifted anode
        shifted_ne = df_ne.copy()

        # Interpolate across the max and min SOC of the half-cell dfs
        df_1 = shifted_pe.copy()
        df_2 = shifted_ne.copy()
        min_soc = np.min((np.min(df_1['SOC_aligned']), np.min(df_2['SOC_aligned'])))
        max_soc = np.max((np.max(df_1['SOC_aligned']), np.max(df_2['SOC_aligned'])))
        soc_vec = np.linspace(min_soc, max_soc, 1001)

        df_1_interper = interp1d(df_1['SOC_aligned'],
                                 df_1['Voltage_aligned'],
                                 bounds_error=False, fill_value=np.nan)
        df_1['SOC_aligned'] = soc_vec.copy()
        df_1['Voltage_aligned'] = df_1_interper(soc_vec)

        df_2_interper = interp1d(df_2['SOC_aligned'],
                                 df_2['Voltage_aligned'],
                                 bounds_error=False, fill_value=np.nan)
        df_2['SOC_aligned'] = soc_vec.copy()
        df_2['Voltage_aligned'] = df_2_interper(soc_vec)

        # Calculate the full-cell profile
        df_3 = pd.DataFrame()
        df_3['Voltage_aligned'] = df_1['Voltage_aligned'].subtract(df_2['Voltage_aligned'])
        df_3['SOC_aligned'] = df_2['SOC_aligned']

        # centering
        centering_value = - np.min(df_3['SOC_aligned'].iloc[np.argmin(np.abs(df_3['Voltage_aligned'] - 2.7))])

        emulated_full_cell_centered = df_3.copy()
        emulated_full_cell_centered['SOC_aligned'] = df_3['SOC_aligned'] + centering_value

        pe_out_centered = df_1.copy()
        pe_out_centered['SOC_aligned'] = df_1['SOC_aligned'] + centering_value

        ne_out_centered = df_2.copy()
        ne_out_centered['SOC_aligned'] = df_2['SOC_aligned'] + centering_value

        # Scaling

        emulated_full_cell_centered.loc[(emulated_full_cell_centered['Voltage_aligned'] > self.UPPER_VOLTAGE) | (
                    emulated_full_cell_centered['Voltage_aligned'] < self.LOWER_VOLTAGE)] = np.nan

        scaling_value = np.max(emulated_full_cell_centered['SOC_aligned'].loc[~emulated_full_cell_centered[
            'Voltage_aligned'].isna()])  # value to scale emulated back to 100% SOC

        emulated_full_cell_centered_scaled = emulated_full_cell_centered.copy()
        emulated_full_cell_centered_scaled['SOC_aligned'] = \
            emulated_full_cell_centered['SOC_aligned'] / scaling_value * 100

        pe_out_centered_scaled = pe_out_centered.copy()
        pe_out_centered_scaled['SOC_aligned'] = pe_out_centered['SOC_aligned'] / scaling_value * 100

        ne_out_centered_scaled = ne_out_centered.copy()
        ne_out_centered_scaled['SOC_aligned'] = ne_out_centered['SOC_aligned'] / scaling_value * 100

        # Make new interpolation across SOC for full-cell error calculation
        emulated_full_cell_interper = interp1d(
            emulated_full_cell_centered_scaled.loc[
                ~emulated_full_cell_centered_scaled.Voltage_aligned.isna()].SOC_aligned,
            emulated_full_cell_centered_scaled.loc[
                ~emulated_full_cell_centered_scaled.Voltage_aligned.isna()].Voltage_aligned,
            bounds_error=False, fill_value='extrapolate', kind='quadratic')
        real_full_cell_interper = interp1d(df_real.SOC_aligned,
                                           df_real.Voltage_aligned,
                                           bounds_error=False, fill_value=(self.LOWER_VOLTAGE, self.UPPER_VOLTAGE))

        # Interpolate the emulated full-cell profile
        SOC_vec_full_cell = np.linspace(0, 100.0, 1001)
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['SOC_aligned'] = SOC_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(SOC_vec_full_cell)

        # Interpolate the true full-cell profile
        df_real_interped = emulated_full_cell_interped.copy()
        df_real_interped['SOC_aligned'] = SOC_vec_full_cell
        df_real_interped['Voltage_aligned'] = real_full_cell_interper(SOC_vec_full_cell)

        return pe_out_centered_scaled, ne_out_centered_scaled, df_real_interped, emulated_full_cell_interped

    def halfcell_initial_matching(self, x, *params):
        """
        Augments halfcell voltage profiles by scaling and translating them. Typically used in an optimization
        routine to fit the emulated full cell profile to a real cell profile.
        Inputs:
        x: an array of 4 or 5 parameters containing scale_pe, offset_pe,scale_pe, scale_ne, offset_ne,
            and optionally ne_2_x
        df_real: dataframe for the first diagnostic (pristine) of the real full cell. Columns for SOC (ev)
        self.pe_pristine: dataframe for the positive electrode. Columns for SOC (evenly spaced) and Voltage.
        df_ne_1: dataframe for the primary material in the negative electrode. Columns for SOC (evenly spaced)
            and Voltage.
        df_ne_2: dataframe for the secondary material in the negative electrode. Columns for SOC (evenly spaced)
            and Voltage. Supply empty DataFrame if not emulating a blend from two known elelctrodes.
        """

        df_real, df_pe, df_ne_1, df_ne_2 = params

        scale_pe = x[0]
        offset_pe = x[1]
        scale_ne = x[2]
        offset_ne = x[3]

        if df_ne_2.empty:
            # one-material anode
            df_ne = pd.DataFrame()
            df_ne['Voltage_aligned'] = df_ne_1['Voltage_aligned']
            df_ne['SOC_aligned'] = df_ne_1['SOC_aligned']
        else:
            # blended anode
            ne_2_x = x[4]  # fraction of NE_2
            df_ne = pd.DataFrame()
            df_ne['Voltage_aligned'] = (ne_2_x * df_ne_2['Voltage_aligned'] + (1 - ne_2_x) * df_ne_1['Voltage_aligned'])
            df_ne['SOC_aligned'] = df_ne_1['SOC_aligned']

        # shifted cathode
        shifted_pe = df_pe.copy()
        shifted_pe['SOC_aligned'] = shifted_pe['SOC_aligned'] * scale_pe + offset_pe

        # shifted anode
        shifted_ne = df_ne.copy()
        shifted_ne['SOC_aligned'] = shifted_ne['SOC_aligned'] * scale_ne + offset_ne

        # Interpolate across the max and min SOC of the half-cell dfs
        df_1 = shifted_pe.copy()
        df_2 = shifted_ne.copy()
        min_soc = np.min((np.min(df_1['SOC_aligned']), np.min(df_2['SOC_aligned'])))
        max_soc = np.max((np.max(df_1['SOC_aligned']), np.max(df_2['SOC_aligned'])))
        soc_vec = np.linspace(min_soc, max_soc, 1001)

        df_1_interper = interp1d(df_1['SOC_aligned'], df_1['Voltage_aligned'], bounds_error=False)
        df_1['SOC_aligned'] = soc_vec.copy()
        df_1['Voltage_aligned'] = df_1_interper(soc_vec)

        df_2_interper = interp1d(df_2['SOC_aligned'], df_2['Voltage_aligned'], bounds_error=False)
        df_2['SOC_aligned'] = soc_vec.copy()
        df_2['Voltage_aligned'] = df_2_interper(soc_vec)

        # Calculate the full-cell profile
        df_3 = pd.DataFrame()
        df_3['Voltage_aligned'] = df_1['Voltage_aligned'].subtract(df_2['Voltage_aligned'])
        df_3.loc[(df_3['Voltage_aligned'] > 4.2) | (df_3['Voltage_aligned'] < 2.7)] = np.nan
        df_3['SOC_aligned'] = df_2['SOC_aligned']

        # centering
        centering_value = (df_real['SOC_aligned'].loc[(np.argmin(np.abs(df_real['Voltage_aligned'] -
                                                                        np.min(df_3['Voltage_aligned'].loc[
                                                                                   ~df_3['Voltage_aligned'].isna()]))))]
                           - np.min(df_3['SOC_aligned'].loc[~df_3['Voltage_aligned'].isna()])
                           )
        emulated_full_cell_centered = df_3.copy()
        emulated_full_cell_centered['SOC_aligned'] = df_3['SOC_aligned'] + centering_value

        PE_out_centered = df_1.copy()
        PE_out_centered['SOC_aligned'] = df_1['SOC_aligned'] + centering_value

        NE_out_centered = df_2.copy()
        NE_out_centered['SOC_aligned'] = df_2['SOC_aligned'] + centering_value

        # Make new interpolation across SOC for full-cell error calculation
        min_soc_full_cell = np.min((np.min(df_3.loc[~df_3.Voltage_aligned.isna()].SOC_aligned),
                                    np.min(df_real.loc[~df_real.Voltage_aligned.isna()].SOC_aligned)))
        max_soc_full_cell = np.max((np.max(df_3.loc[~df_3.Voltage_aligned.isna()].SOC_aligned),
                                    np.max(df_real.loc[~df_real.Voltage_aligned.isna()].SOC_aligned)))
        soc_vec_full_cell = np.linspace(min_soc_full_cell, max_soc_full_cell, 1001)

        emulated_full_cell_interper = interp1d(emulated_full_cell_centered.SOC_aligned,
                                               emulated_full_cell_centered.Voltage_aligned, bounds_error=False)
        real_full_cell_interper = interp1d(df_real.SOC_aligned,
                                           df_real.Voltage_aligned, bounds_error=False)

        # Interpolate the emulated full-cell profile
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['SOC_aligned'] = soc_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(soc_vec_full_cell)

        # Interpolate the true full-cell profile
        df_real_interped = emulated_full_cell_interped.copy()
        df_real_interped['SOC_aligned'] = soc_vec_full_cell
        df_real_interped['Voltage_aligned'] = real_full_cell_interper(soc_vec_full_cell)

        return df_1, df_2, df_real_interped, emulated_full_cell_interped

    def _get_error_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        error_dis = distance.euclidean(
            df_real_interped['Voltage_aligned'].loc[
                (~df_real_interped['Voltage_aligned'].isna())
                & (~emulated_full_cell_interped['Voltage_aligned'].isna())],
            emulated_full_cell_interped['Voltage_aligned'].loc[
                (~df_real_interped['Voltage_aligned'].isna())
                & (~emulated_full_cell_interped['Voltage_aligned'].isna())]
                                  )

        error = error_dis + 0.01 * len(emulated_full_cell_interped['Voltage_aligned'].loc[
            emulated_full_cell_interped['Voltage_aligned'].isna()]
                                                 )

        return error

    def _impose_degradation(self,
                            pe_pristine=pd.DataFrame(),
                            ne_1_pristine=pd.DataFrame(),
                            ne_2_pristine_pos=pd.DataFrame(),
                            ne_2_pristine_neg=pd.DataFrame(),
                            lli=0.0, lam_pe=0.0, lam_ne=0.0, x_ne_2=0.0):
        pe_translation = 0
        pe_shrinkage = 0
        ne_translation = 0
        ne_shrinkage = 0

        # Blend negative electrodes
        ne_pristine = blend_electrodes(ne_1_pristine, ne_2_pristine_pos, ne_2_pristine_neg, x_ne_2)

        # Update degradation shifts for LLI
        ne_translation += lli

        # Update degradation shifts for LAM_PE
        upper_voltage_limit = self.UPPER_VOLTAGE
        pe_soc_setpoint = pe_pristine['SOC_aligned'].loc[
            np.argmin(np.abs(pe_pristine['Voltage_aligned']
                             - ne_pristine[
                                 'Voltage_aligned'] - upper_voltage_limit))]  # SOC at which the upper voltage
        # limit is hit
        pe_translation += lam_pe * pe_soc_setpoint / 100  # correction for shrinkage to ensure the locking of
        # the upper voltage SOC
        pe_shrinkage += lam_pe  # shrinkage of PE capacity due to LAM_PE

        #  Update degradation shifts for LAM_NE
        lower_voltage_limit = self.LOWER_VOLTAGE
        ne_soc_setpoint = ne_pristine['SOC_aligned'].loc[
            np.argmin((pe_pristine['Voltage_aligned']
                       - ne_pristine[
                           'Voltage_aligned'] - lower_voltage_limit))]  # SOC at which the lower voltage limit is hit
        ne_translation += lam_ne * ne_soc_setpoint / 100  # correction for shrinkage to ensure the locking of the
        # lower voltage SOC
        ne_shrinkage += lam_ne  # shrinkage of NE capacity due to LAM_NE

        # Update SOC vector for both electrodes according to their imposed degradation
        pe_pristine_shifted_by_deg = pe_pristine.copy()
        pe_pristine_shifted_by_deg['SOC_aligned'] = pe_pristine_shifted_by_deg['SOC_aligned'] * (
                    1 - pe_shrinkage / 100) + pe_translation

        ne_pristine_shifted_by_deg = ne_pristine.copy()
        ne_pristine_shifted_by_deg['SOC_aligned'] = ne_pristine_shifted_by_deg['SOC_aligned'] * (
                    1 - ne_shrinkage / 100) + ne_translation

        # Re-interpolate to align dataframes for differencing
        lower_soc = np.min((np.min(pe_pristine_shifted_by_deg['SOC_aligned']),
                            np.min(ne_pristine_shifted_by_deg['SOC_aligned'])))
        upper_soc = np.max((np.max(pe_pristine_shifted_by_deg['SOC_aligned']),
                            np.max(ne_pristine_shifted_by_deg['SOC_aligned'])))
        soc_vec = np.linspace(lower_soc, upper_soc, 1001)

        pe_pristine_interper = interp1d(pe_pristine_shifted_by_deg['SOC_aligned'],
                                        pe_pristine_shifted_by_deg['Voltage_aligned'], bounds_error=False)
        pe_degraded = pe_pristine_shifted_by_deg.copy()
        pe_degraded['SOC_aligned'] = soc_vec
        pe_degraded['Voltage_aligned'] = pe_pristine_interper(soc_vec)

        ne_pristine_interper = interp1d(ne_pristine_shifted_by_deg['SOC_aligned'],
                                        ne_pristine_shifted_by_deg['Voltage_aligned'], bounds_error=False)
        ne_degraded = ne_pristine_shifted_by_deg.copy()
        ne_degraded['SOC_aligned'] = soc_vec
        ne_degraded['Voltage_aligned'] = ne_pristine_interper(soc_vec)

        return pe_degraded, ne_degraded

    def get_dQdV_over_V_from_degradation_matching(self, x, *params):
        pe_out_centered, ne_out_centered, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_v3(x, *params)

        # Calculate dQdV from full cell profiles
        dq_dv_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                  columns=['dQdV']).ewm(alpha=x[-2]).mean()
        dq_dv_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV']).ewm(alpha=x[-1]).mean()

        # Include original data
        dq_dv_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dq_dv_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dq_dv_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dq_dv_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)

        v_dq_dv_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         dq_dv_real['dQdV'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         bounds_error=False, fill_value=0)
        v_soc_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       dq_dv_real['SOC_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        v_dq_dv_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                                 ~dq_dv_emulated['Voltage_aligned'].isna()],
                                             dq_dv_emulated['dQdV'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                             bounds_error=False, fill_value=0)
        v_soc_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                               ~dq_dv_emulated['Voltage_aligned'].isna()],
                                           dq_dv_emulated['SOC_aligned'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dq_dv_over_v_real = pd.DataFrame(v_dq_dv_interper_real(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_real['SOC_aligned'] = v_soc_interper_real(voltage_vec)
        dq_dv_over_v_real['Voltage_aligned'] = voltage_vec

        dq_dv_over_v_emulated = pd.DataFrame(v_dq_dv_interper_emulated(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_emulated['SOC_aligned'] = v_soc_interper_emulated(voltage_vec)
        dq_dv_over_v_emulated['Voltage_aligned'] = voltage_vec

        return (pe_out_centered,
                ne_out_centered,
                dq_dv_over_v_real,
                dq_dv_over_v_emulated,
                df_real_interped,
                emulated_full_cell_interped)

    def get_error_dQdV_over_V_from_degradation_matching(self, x, *params):
        try:
            (pe_out_centered, ne_out_centered,
             dq_dv_over_v_real, dq_dv_over_v_emulated,
             df_real_interped, emulated_full_cell_interped) = self.get_dQdV_over_V_from_degradation_matching(x, *params)
            error = distance.euclidean(dq_dv_over_v_real['dQdV'], dq_dv_over_v_emulated['dQdV']) + 0.01 * len(
                dq_dv_over_v_emulated['dQdV'].loc[dq_dv_over_v_emulated['dQdV'].isna()])
        except ValueError:
            error = 1000  # set an error for cases where calcuation fails
        return error

    def get_dVdQ_over_SOC_from_degradation_matching(self, x, *params):
        pe_out_centered, ne_out_centered, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_v3(x, *params)

        # Calculate dVdQ from full cell profiles
        dv_dq_over_soc_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'],
                                                       df_real_interped['SOC_aligned']),
                                           columns=['dVdQ']).ewm(alpha=x[-2]).mean()
        dv_dq_over_soc_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['SOC_aligned']),
            columns=['dVdQ']).ewm(alpha=x[-1]).mean()

        # Include original data
        dv_dq_over_soc_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dv_dq_over_soc_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dv_dq_over_soc_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dv_dq_over_soc_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return (pe_out_centered,
                ne_out_centered,
                dv_dq_over_soc_real,
                dv_dq_over_soc_emulated,
                df_real_interped,
                emulated_full_cell_interped)

    def get_error_dVdQ_over_SOC_from_degradation_matching(self, x, *params):
        try:
            (pe_out_centered, ne_out_centered,
             dv_dq_over_soc_real, dv_dq_over_soc_emulated,
             df_real_interped, emulated_full_cell_interped) = \
                self.get_dVdQ_over_SOC_from_degradation_matching(x, *params)
            error = distance.euclidean(dv_dq_over_soc_real['dVdQ'], dv_dq_over_soc_emulated['dVdQ']) + 0.01 * len(
                dv_dq_over_soc_emulated['dVdQ'].loc[dv_dq_over_soc_emulated['dVdQ'].isna()])
        except ValueError:
            error = 1000  # set an error for cases where calcuation fails
        return error

    def get_V_over_SOC_from_degradation_matching(self, x, *params):
        (PE_out_centered, NE_out_centered, real_aligned, emulated_aligned) = \
            self.halfcell_degradation_matching_v3(x, *params)

        min_soc_full_cell = np.min(real_aligned.loc[~real_aligned.Voltage_aligned.isna()].SOC_aligned)
        max_soc_full_cell = np.max(real_aligned.loc[~real_aligned.Voltage_aligned.isna()].SOC_aligned)

        soc_vec_full_cell = np.linspace(min_soc_full_cell, max_soc_full_cell, 1001)

        emulated_full_cell_interper = interp1d(
            emulated_aligned.SOC_aligned.loc[~real_aligned.Voltage_aligned.isna()],
            emulated_aligned.Voltage_aligned.loc[~real_aligned.Voltage_aligned.isna()],
            bounds_error=False)
        real_full_cell_interper = interp1d(real_aligned.SOC_aligned.loc[~real_aligned.Voltage_aligned.isna()],
                                           real_aligned.Voltage_aligned.loc[~real_aligned.Voltage_aligned.isna()],
                                           bounds_error=False)

        # Interpolate the emulated full-cell profile
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['SOC_aligned'] = soc_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(soc_vec_full_cell)

        # Interpolate the true full-cell profile
        df_real_interped = emulated_full_cell_interped.copy()
        df_real_interped['SOC_aligned'] = soc_vec_full_cell
        df_real_interped['Voltage_aligned'] = real_full_cell_interper(soc_vec_full_cell)
        return PE_out_centered, NE_out_centered, df_real_interped, emulated_full_cell_interped

    def _get_error_from_degradation_matching(self, x, *params):
        try:
            (pe_out_centered, ne_out_centered, real_aligned, emulated_aligned
             ) = self.get_V_over_SOC_from_degradation_matching(x, *params)
            error = (distance.euclidean(real_aligned.loc[~emulated_aligned.Voltage_aligned.isna()].values.ravel(),
                                        emulated_aligned.loc[~emulated_aligned.Voltage_aligned.isna()].values.ravel()) +
                     0.001 * len(emulated_aligned.loc[emulated_aligned.Voltage_aligned.isna()]))
        except ValueError:
            error = 100
        return error

    def halfcell_degradation_matching_v3(self, x, *params):
        lli = x[0]
        lam_pe = x[1]
        lam_ne = x[2]
        x_ne_2 = x[3]

        pe_pristine, ne_1_pristine, ne_2_pristine_pos, ne_2_pristine_neg, real_full_cell_with_degradation = params

        pe_out, ne_out = self._impose_degradation(pe_pristine, ne_1_pristine,
                                                  ne_2_pristine_pos, ne_2_pristine_neg,
                                                  lli, lam_pe,
                                                  lam_ne, x_ne_2)
        emulated_full_cell_with_degradation = pd.DataFrame()
        emulated_full_cell_with_degradation['SOC_aligned'] = pe_out['SOC_aligned'].copy()
        emulated_full_cell_with_degradation['Voltage_aligned'] = pe_out['Voltage_aligned'] - ne_out['Voltage_aligned']

        emulated_full_cell_with_degradation_centered = pd.DataFrame()

        emulated_full_cell_with_degradation_centered['Voltage_aligned'] = emulated_full_cell_with_degradation[
            'Voltage_aligned']
        centering_value = - np.min(emulated_full_cell_with_degradation['SOC_aligned'].loc[
                                       (~emulated_full_cell_with_degradation['Voltage_aligned'].isna())
                                   ])

        emulated_full_cell_with_degradation_centered['SOC_aligned'] = \
            (emulated_full_cell_with_degradation['SOC_aligned'] + centering_value)

        pe_out_centered = pe_out.copy()
        pe_out_centered['SOC_aligned'] = pe_out['SOC_aligned'] + centering_value

        ne_out_centered = ne_out.copy()
        ne_out_centered['SOC_aligned'] = ne_out['SOC_aligned'] + centering_value

        # Interpolate full profiles across same SOC range
        min_soc = np.min(
            real_full_cell_with_degradation['SOC_aligned'].loc[
                ~real_full_cell_with_degradation['Voltage_aligned'].isna()])
        max_soc = np.max(
            real_full_cell_with_degradation['SOC_aligned'].loc[
                ~real_full_cell_with_degradation['Voltage_aligned'].isna()])
        emulated_interper = interp1d(emulated_full_cell_with_degradation_centered['SOC_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_centered['Voltage_aligned'].isna()],
                                     emulated_full_cell_with_degradation_centered['Voltage_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_centered['Voltage_aligned'].isna()],
                                     bounds_error=False)
        real_interper = interp1d(
            real_full_cell_with_degradation['SOC_aligned'].loc[
                ~real_full_cell_with_degradation['Voltage_aligned'].isna()],
            real_full_cell_with_degradation['Voltage_aligned'].loc[
                ~real_full_cell_with_degradation['Voltage_aligned'].isna()],
            bounds_error=False)

        soc_vec = np.linspace(min_soc, max_soc, 1001)

        emulated_aligned = pd.DataFrame()
        emulated_aligned['SOC_aligned'] = soc_vec
        emulated_aligned['Voltage_aligned'] = emulated_interper(soc_vec)

        real_aligned = pd.DataFrame()
        real_aligned['SOC_aligned'] = soc_vec
        real_aligned['Voltage_aligned'] = real_interper(soc_vec)

        return pe_out_centered, ne_out_centered, real_aligned, emulated_aligned

    def halfcell_degradation_matching_v2(self, x, *params):
        lli = x[0]
        lam_pe = x[1]
        lam_ne = x[2]

        pe_pristine, ne_pristine, real_full_cell_with_degradation = params

        pe_out, ne_out = self._impose_degradation(pe_pristine, ne_pristine, lli, lam_pe, lam_ne)
        emulated_full_cell_with_degradation = pd.DataFrame()
        emulated_full_cell_with_degradation['SOC_aligned'] = pe_out['SOC_aligned'].copy()
        emulated_full_cell_with_degradation['Voltage_aligned'] = pe_out['Voltage_aligned'] - ne_out['Voltage_aligned']

        emulated_full_cell_with_degradation_centered = pd.DataFrame()

        emulated_full_cell_with_degradation_centered['Voltage_aligned'] = emulated_full_cell_with_degradation[
            'Voltage_aligned']
        emulated_full_cell_with_degradation_centered['Voltage_aligned'].loc[
            (emulated_full_cell_with_degradation_centered['Voltage_aligned'] > 4.2) |
            (emulated_full_cell_with_degradation_centered['Voltage_aligned'] < 2.7)] = np.nan

        centering_value = (
                real_full_cell_with_degradation['SOC_aligned'].loc[
                    (np.argmin(np.abs(real_full_cell_with_degradation['Voltage_aligned']
                                      - np.min(emulated_full_cell_with_degradation['Voltage_aligned'].loc[
                                                   ~emulated_full_cell_with_degradation['Voltage_aligned'].isna()]))))
                ]
                - np.min(emulated_full_cell_with_degradation['SOC_aligned'].loc[
                             ~emulated_full_cell_with_degradation['Voltage_aligned'].isna()])
        )

        emulated_full_cell_with_degradation_centered['SOC_aligned'] = \
            (emulated_full_cell_with_degradation['SOC_aligned'] + centering_value)

        pe_out_centered = pe_out.copy()
        pe_out_centered['SOC_aligned'] = pe_out['SOC_aligned'] + centering_value

        ne_out_centered = ne_out.copy()
        ne_out_centered['SOC_aligned'] = ne_out['SOC_aligned'] + centering_value

        # Interpolate full profiles across same SOC range
        min_soc = np.min(
            (np.min(emulated_full_cell_with_degradation_centered['SOC_aligned'].loc[
                        ~emulated_full_cell_with_degradation_centered['Voltage_aligned'].isna()]),
             np.min(real_full_cell_with_degradation['SOC_aligned'].loc[
                        ~real_full_cell_with_degradation['Voltage_aligned'].isna()])
             )
        )
        max_soc = np.max(
            (np.max(emulated_full_cell_with_degradation_centered['SOC_aligned'].loc[
                        ~emulated_full_cell_with_degradation_centered['Voltage_aligned'].isna()]),
             np.max(real_full_cell_with_degradation['SOC_aligned'].loc[
                        ~real_full_cell_with_degradation['Voltage_aligned'].isna()]))
        )
        emulated_interper = interp1d(emulated_full_cell_with_degradation_centered['SOC_aligned'],
                                     emulated_full_cell_with_degradation_centered['Voltage_aligned'],
                                     bounds_error=False)
        real_interper = interp1d(real_full_cell_with_degradation['SOC_aligned'],
                                 real_full_cell_with_degradation['Voltage_aligned'],
                                 bounds_error=False)

        soc_vec = np.linspace(min_soc, max_soc, 1001)

        emulated_aligned = pd.DataFrame()
        emulated_aligned['SOC_aligned'] = soc_vec
        emulated_aligned['Voltage_aligned'] = emulated_interper(soc_vec)

        real_aligned = pd.DataFrame()
        real_aligned['SOC_aligned'] = soc_vec
        real_aligned['Voltage_aligned'] = real_interper(soc_vec)

        return pe_out_centered, ne_out_centered, real_aligned, emulated_aligned

    def get_voltage_curves_for_cell(self, processed_cycler_run, cycle_type='rpt_0.2C'):
        # Filter down to only cycles of cycle_type
        diag_type_cycles = processed_cycler_run.diagnostic_interpolated.loc[
            processed_cycler_run.diagnostic_interpolated['cycle_type'] == cycle_type]

        # Loop across cycles
        cycle_indexes = diag_type_cycles['cycle_index'].unique()
        for i in cycle_indexes:
            diag_type_cycle_i = diag_type_cycles.loc[diag_type_cycles.cycle_index == i]
            x_charge = diag_type_cycle_i.charge_energy.loc[diag_type_cycles['step_type'] == 0]
            y_charge = diag_type_cycle_i.voltage.loc[diag_type_cycles['step_type'] == 0]

        return x_charge, y_charge

    def intracell_wrapper_init(self,
                               cell_struct,
                               initial_matching_bounds=None
                               ):
        """
        Wrapper function that calls all of the functions to get the initial values for the cell
        before fitting all of the
        Args:
             cell_struct (beep.structure): dataframe to determine whether
                charging or discharging
             initial_matching_bounds (tuple): Bounds for fitting parameters
        Returns:
            (bool): True if step is the charge state specified.
        """
        if initial_matching_bounds is None:
            initial_matching_bounds = ((0.8, 1.2), (-20.0, 20.0), (1, 1), (0.1, 0.1), (0.1, 0.1))

        real_cell_initial_charge_profile_aligned, real_cell_initial_charge_profile = \
            self.process_beep_cycle_data_for_initial_halfcell_analysis(cell_struct)
        opt_result_halfcell_initial_matching = differential_evolution(self._get_error_from_halfcell_initial_matching,
                                                                      initial_matching_bounds,
                                                                      args=(real_cell_initial_charge_profile_aligned,
                                                                            self.pe_pristine, self.ne_1_pristine,
                                                                            self.ne_2_pristine_pos,
                                                                            self.ne_2_pristine_neg),
                                                                      strategy='best1bin', maxiter=1000,
                                                                      popsize=15, tol=0.001, mutation=0.5,
                                                                      recombination=0.7, seed=1,
                                                                      callback=None, disp=False, polish=True,
                                                                      init='latinhypercube', atol=0,
                                                                      updating='deferred', workers=-1, constraints=())
        (PE_pristine_matched,
         NE_pristine_matched,
         df_real_interped,
         emulated_full_cell_interped) = self.halfcell_initial_matching_v2(opt_result_halfcell_initial_matching.x,
                                                                          real_cell_initial_charge_profile_aligned,
                                                                          self.pe_pristine,
                                                                          self.ne_1_pristine,
                                                                          self.ne_2_pristine_pos,
                                                                          self.ne_2_pristine_neg)
        return (real_cell_initial_charge_profile_aligned,
                real_cell_initial_charge_profile,
                PE_pristine_matched,
                NE_pristine_matched)

    def intracell_values_wrapper(self,
                                 cycle_index,
                                 cell_struct,
                                 real_cell_initial_charge_profile_aligned,
                                 real_cell_initial_charge_profile,
                                 PE_pristine_matched,
                                 NE_pristine_matched,
                                 degradation_bounds=None
                                 ):

        if degradation_bounds is None:
            degradation_bounds = ((-10, 50),  # LLI
                                  (-10, 50),  # LAM_PE
                                  (-10, 50),  # LAM_NE
                                  (1, 1),  # (-1,1) x_NE_2
                                  (0.1, 0.1),  # (0.01,0.1)
                                  (0.1, 0.1),  # (0.01,0.1)
                                  )

        real_cell_candidate_charge_profile_aligned = self.process_beep_cycle_data_for_candidate_halfcell_analysis(
            cell_struct,
            real_cell_initial_charge_profile_aligned,
            real_cell_initial_charge_profile,
            cycle_index)

        degradation_optimization_result = differential_evolution(self._get_error_from_degradation_matching,
                                                                 degradation_bounds,
                                                                 args=(PE_pristine_matched,
                                                                       NE_pristine_matched,
                                                                       self.ne_2_pristine_pos,
                                                                       self.ne_2_pristine_neg,
                                                                       real_cell_candidate_charge_profile_aligned
                                                                       ),
                                                                 strategy='best1bin', maxiter=100000,
                                                                 popsize=15, tol=0.001, mutation=0.5,
                                                                 recombination=0.7,
                                                                 seed=1,
                                                                 callback=None, disp=False, polish=True,
                                                                 init='latinhypercube',
                                                                 atol=0, updating='deferred', workers=-1,
                                                                 constraints=()
                                                                 )

        (PE_out_centered,
         NE_out_centered,
         dVdQ_over_SOC_real,
         dVdQ_over_SOC_emulated,
         df_real_interped,
         emulated_full_cell_interped) = self.get_dQdV_over_V_from_degradation_matching(
            degradation_optimization_result.x,
            PE_pristine_matched,
            NE_pristine_matched,
            self.ne_2_pristine_pos,
            self.ne_2_pristine_neg,
            real_cell_candidate_charge_profile_aligned)
        #
        (PE_upper_voltage, PE_lower_voltage, PE_upper_SOC, PE_lower_SOC, PE_mass,
         NE_upper_voltage, NE_lower_voltage, NE_upper_SOC, NE_lower_SOC, NE_mass,
         SOC_upper, SOC_lower, Li_mass) = get_halfcell_voltages(PE_out_centered, NE_out_centered)
        #
        LLI = degradation_optimization_result.x[0]
        LAM_PE = degradation_optimization_result.x[1]
        LAM_NE = degradation_optimization_result.x[2]
        x_NE_2 = degradation_optimization_result.x[3]
        alpha_real = degradation_optimization_result.x[4]
        alpha_emulated = degradation_optimization_result.x[5]

        loss_dict = {cycle_index: [LLI, LAM_PE, LAM_NE, x_NE_2, alpha_real, alpha_emulated,
                                   PE_upper_voltage, PE_lower_voltage, PE_upper_SOC, PE_lower_SOC, PE_mass,
                                   NE_upper_voltage, NE_lower_voltage, NE_upper_SOC, NE_lower_SOC, NE_mass,
                                   Li_mass
                                   ]
                     }
        profiles_dict = {cycle_index: real_cell_candidate_charge_profile_aligned}

        return loss_dict, profiles_dict


def blend_electrodes(electrode_1, electrode_2_pos, electrode_2_neg, x_2):
    """
    Inputs:
    electrode_1: Primary material in electrode, typically Gr. DataFrame supplied with SOC evenly spaced and voltage.
    electrode_2: Secondary material in electrode, typically Si. DataFrame supplied with SOC evenly spaced and
        voltage as an additional column.
    x_2: Fraction of electrode_2 material. Supplied as scalar value.
    """
    if electrode_2_pos.empty:
        df_blended = electrode_1
        return df_blended

    if electrode_2_neg.empty:
        electrode_2 = electrode_2_pos
        x_2 = np.abs(x_2)
    elif x_2 > 0:
        electrode_2 = electrode_2_pos
    else:
        electrode_2 = electrode_2_neg
        x_2 = np.abs(x_2)

    electrode_1_interper = interp1d(electrode_1['Voltage_aligned'], electrode_1['SOC_aligned'], bounds_error=False,
                                    fill_value='extrapolate')
    electrode_2_interper = interp1d(electrode_2['Voltage_aligned'], electrode_2['SOC_aligned'], bounds_error=False,
                                    fill_value='extrapolate')

    voltage_vec = np.linspace(np.min((np.min(electrode_1['Voltage_aligned']),
                                      np.min(electrode_2['Voltage_aligned']))),
                              np.max((np.max(electrode_1['Voltage_aligned']),
                                      np.max(electrode_2['Voltage_aligned']))),
                              1001)
    electrode_1_voltage_aligned = pd.DataFrame(electrode_1_interper(voltage_vec), columns=['SOC'])
    electrode_2_voltage_aligned = pd.DataFrame(electrode_2_interper(voltage_vec), columns=['SOC'])
    electrode_1_voltage_aligned['Voltage'] = voltage_vec
    electrode_2_voltage_aligned['Voltage'] = voltage_vec

    df_blend_voltage_aligned = pd.DataFrame(
        (1 - x_2) * electrode_1_voltage_aligned['SOC'] + x_2 * electrode_2_voltage_aligned['SOC'], columns=['SOC'])
    df_blend_voltage_aligned['Voltage'] = electrode_1_voltage_aligned.merge(electrode_2_voltage_aligned,
                                                                            on='Voltage')['Voltage']

    df_blended_interper = interp1d(df_blend_voltage_aligned['SOC'], df_blend_voltage_aligned['Voltage'],
                                   bounds_error=False)
    soc_vec = np.linspace(0, 100, 1001)

    df_blended = pd.DataFrame(df_blended_interper(soc_vec), columns=['Voltage_aligned'])
    df_blended['SOC_aligned'] = soc_vec

    # Modify NE to fully span 100% SOC within its valid voltage window
    df_blended_soc_mod_interper = interp1d(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()],
                                           df_blended['Voltage_aligned'].loc[~df_blended['Voltage_aligned'].isna()],
                                           bounds_error=False)
    soc_vec = np.linspace(np.min(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()]),
                          np.max(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()]),
                          1001)
    df_blended_soc_mod = pd.DataFrame(df_blended_soc_mod_interper(soc_vec), columns=['Voltage_aligned'])
    df_blended_soc_mod['SOC_aligned'] = soc_vec / np.max(soc_vec) * 100
    return df_blended_soc_mod


def get_halfcell_voltages(pe_out_centered, ne_out_centered):
    pe_minus_ne_centered = pd.DataFrame(pe_out_centered['Voltage_aligned'] - ne_out_centered['Voltage_aligned'],
                                        columns=['Voltage_aligned'])
    pe_minus_ne_centered['SOC_aligned'] = pe_out_centered['SOC_aligned']

    soc_upper = pe_minus_ne_centered.iloc[
        np.argmin(np.abs(pe_minus_ne_centered.Voltage_aligned - 4.20))].SOC_aligned
    soc_lower = pe_minus_ne_centered.iloc[
        np.argmin(np.abs(pe_minus_ne_centered.Voltage_aligned - 2.70))].SOC_aligned

    pe_upper_voltage = pe_out_centered.loc[pe_out_centered.SOC_aligned == soc_upper].Voltage_aligned.values[0]
    pe_lower_voltage = pe_out_centered.loc[pe_out_centered.SOC_aligned == soc_lower].Voltage_aligned.values[0]

    pe_upper_soc = ((pe_out_centered.loc[pe_out_centered.Voltage_aligned == pe_upper_voltage]['SOC_aligned'] -
                     np.min(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()])) / (
                            np.max(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()]) -
                            np.min(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()]))
                    ).values[0] * 100
    pe_lower_soc = ((pe_out_centered.loc[pe_out_centered.Voltage_aligned == pe_lower_voltage]['SOC_aligned'] -
                     np.min(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()])) / (
                            np.max(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()]) -
                            np.min(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()]))
                    ).values[0] * 100
    pe_mass = np.max(pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()]) - np.min(
        pe_out_centered['SOC_aligned'].loc[~pe_out_centered['Voltage_aligned'].isna()])

    ne_upper_voltage = ne_out_centered.loc[ne_out_centered.SOC_aligned == soc_upper].Voltage_aligned.values[0]
    ne_lower_voltage = ne_out_centered.loc[ne_out_centered.SOC_aligned == soc_lower].Voltage_aligned.values[0]
    ne_upper_soc = ((ne_out_centered.loc[ne_out_centered.Voltage_aligned == ne_upper_voltage]['SOC_aligned'] -
                     np.min(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()])) / (
                            np.max(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()]) -
                            np.min(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()]))
                    ).values[0] * 100
    ne_lower_soc = ((ne_out_centered.loc[ne_out_centered.Voltage_aligned == ne_lower_voltage]['SOC_aligned'] -
                     np.min(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()])) / (
                            np.max(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()]) -
                            np.min(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()]))
                    ).values[0] * 100
    ne_mass = np.max(ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()]) - np.min(
        ne_out_centered['SOC_aligned'].loc[~ne_out_centered['Voltage_aligned'].isna()])

    li_mass = np.max(pe_minus_ne_centered['SOC_aligned'].loc[~pe_minus_ne_centered.Voltage_aligned.isna()]) - np.min(
        pe_minus_ne_centered['SOC_aligned'].loc[~pe_minus_ne_centered.Voltage_aligned.isna()])

    return (pe_upper_voltage, pe_lower_voltage, pe_upper_soc, pe_lower_soc, pe_mass,
            ne_upper_voltage, ne_lower_voltage, ne_upper_soc, ne_lower_soc, ne_mass,
            soc_upper, soc_lower, li_mass)


def plot_voltage_curves_for_cell(processed_cycler_run,
                                 cycle_type='rpt_0.2C',
                                 x_var='capacity',
                                 step_type=0,
                                 fig_size_inches=(6, 4)):
    # Plot a series of voltage profiles over cycles for this cell
    fig = plt.figure()
    # Filter down to only cycles of cycle_type
    diag_type_cycles = processed_cycler_run.diagnostic_interpolated.loc[
        processed_cycler_run.diagnostic_interpolated['cycle_type'] == cycle_type]

    # Loop across cycles
    cycle_indexes = diag_type_cycles['cycle_index'].unique()
    step_type_list = ['charge', 'discharge']
    for i in cycle_indexes:
        diag_type_cycle_i = diag_type_cycles.loc[diag_type_cycles.cycle_index == i]
        x_charge = diag_type_cycle_i[step_type_list[step_type] + '_' + x_var].loc[
            diag_type_cycles['step_type'] == step_type]
        y_charge = diag_type_cycle_i.voltage.loc[diag_type_cycles['step_type'] == step_type]
        plt.plot(x_charge, y_charge,
                 label='cycle ' + str(i) + ' ' + step_type_list[step_type],
                 c=cm.winter((i - np.min(cycle_indexes)) / (np.max(cycle_indexes) - np.min(cycle_indexes))),
                 figure=fig)

    fig.set_size_inches(fig_size_inches)
    return fig
