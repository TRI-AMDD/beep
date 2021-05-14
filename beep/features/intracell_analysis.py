import json
import numpy as np
import pandas as pd
from monty.serialization import loadfn, dumpfn
from glob import glob
from datetime import datetime
# import yaml
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import pickle
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from scipy.spatial import distance
# from tvregdiff import TVRegDiff

FEATURE_DIR = os.path.dirname(__file__)


class IntracellAnalysis:
    def __init__(self, pe_pristine_file, ne_pristine_file, cycle_type='rtp_0.2C', step_type=0):
        """
        Invokes BeepFeatures object

        Args:
            name (str): predictor object name.
            X (pandas.DataFrame): features in DataFrame format.
            metadata (dict): information about the conditions, data
                and code used to produce features

        """

        if not os.path.split(pe_pristine_file)[0]:
            self.PE_pristine = pd.read_csv(os.path.join(FEATURE_DIR, pe_pristine_file),
                                           usecols=['SOC_aligned', 'Voltage_aligned'])
        else:
            self.PE_pristine = pd.read_csv(os.path.join(pe_pristine_file),
                                           usecols=['SOC_aligned', 'Voltage_aligned'])

        if not os.path.split(ne_pristine_file)[0]:
            self.NE_1_pristine = pd.read_csv(os.path.join(FEATURE_DIR, ne_pristine_file),
                                             usecols=['SOC_aligned', 'Voltage_aligned'])
        else:
            self.NE_1_pristine = pd.read_csv(os.path.join(ne_pristine_file),
                                             usecols=['SOC_aligned', 'Voltage_aligned'])
        # NE_2_pristine_pos = pd.read_csv('Si_aligned.csv')
        # NE_2_pristine_neg = pd.read_csv('Gr_02C_aligned.csv')

        self.NE_2_pristine_pos = pd.DataFrame()
        self.NE_2_pristine_neg = pd.DataFrame()

        self.cycle_type = cycle_type
        self.step_type = step_type
        self.upper_voltage = 4.2
        self.lower_voltage = 2.7
        self.threshold = 4.84 * 0.8

    def process_beep_cycle_data_for_candidate_halfcell_analysis(self,
                                                                cell_struct,
                                                                real_cell_initial_charge_profile_aligned,
                                                                real_cell_initial_charge_profile,
                                                                cycle_index):
        """

        Inputs:
        diag_type_cycles: beep cell_struct.diagnostic_interpolated filtered to one diagnostic type
        real_cell_initial_charge_profile_aligned:
        cycle_index:

        Outputs
        real_cell_initial_charge_profile_aligned: a dataframe containing columns SOC_aligned (evenly spaced) and Voltage_aligned
        """
        diag_type_cycles = cell_struct.diagnostic_data.loc[cell_struct.diagnostic_data['cycle_type'] == self.cycle_type]
        real_cell_candidate_charge_profile = diag_type_cycles.loc[
            (diag_type_cycles.cycle_index == cycle_index)
            & (diag_type_cycles.step_type == 0)  # step_type = 0 is charge, 1 is discharge
            & (diag_type_cycles.voltage < self.upper_voltage)
            & (diag_type_cycles.voltage > self.lower_voltage)][['voltage', 'charge_capacity']]

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
                                                               bounds_error=False)
        real_cell_candidate_charge_profile_aligned['Voltage_aligned'] = real_cell_candidate_charge_profile_interper(
            SOC_vec)
        real_cell_candidate_charge_profile_aligned['Voltage_aligned'].fillna(self.lower_voltage, inplace=True)
        real_cell_candidate_charge_profile_aligned['SOC_aligned'] = SOC_vec / np.max(
            real_cell_initial_charge_profile_aligned['SOC_aligned'].loc[
                ~real_cell_initial_charge_profile_aligned['Voltage_aligned'].isna()]) * 100

        return real_cell_candidate_charge_profile_aligned

    def process_beep_cycle_data_for_initial_halfcell_analysis(self,
                                                              cell_struct,
                                                              step_type=0):
        """
        This function creates the initial (un-degraded) voltage and soc profiles for the cell with columns
        interpolated on voltage and soc. This function works off of the

        Inputs
        cell_struct: beep cell_struct.diagnostic_interpolated filtered to one diagnostic type
        step_type: specifies whether the cell is charging or discharging. 0 is charge, 1 is discharge.

        Outputs
        real_cell_initial_charge_profile_aligned: a dataframe containing columns SOC_aligned (evenly spaced)
            and Voltage_aligned
        real_cell_initial_charge_profile: a dataframe containing columns Voltage (evenly spaced), capacity, and SOC
        """
        if step_type == 0:
            capacity_col = 'charge_capacity'
        else:
            capacity_col = 'discharge_capacity'

        diag_type_cycles = cell_struct.diagnostic_data.loc[cell_struct.diagnostic_data['cycle_type'] == self.cycle_type]
        soc_vec = np.linspace(0, 100.0, 1001)
        cycle_index_of_cycle_type = cell_struct.diagnostic_summary[
            cell_struct.diagnostic_summary.cycle_type == self.cycle_type].cycle_index.iloc[0]
        real_cell_initial_charge_profile = diag_type_cycles.loc[
            (diag_type_cycles.cycle_index == cycle_index_of_cycle_type)
            & (diag_type_cycles.step_type == step_type)  # step_type = 0 is charge, 1 is discharge
            & (diag_type_cycles.voltage < self.upper_voltage)
            & (diag_type_cycles.voltage > self.lower_voltage)][['voltage', capacity_col]]

        real_cell_initial_charge_profile['SOC'] = (
                (
                        real_cell_initial_charge_profile[capacity_col] -
                        np.min(real_cell_initial_charge_profile[capacity_col])
                ) /
                (
                       np.max(real_cell_initial_charge_profile[capacity_col]) -
                       np.min(real_cell_initial_charge_profile[capacity_col])
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
        dQdV_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                 columns=['dQdV'])
        dQdV_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV'])

        # Include original data
        dQdV_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dQdV_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dQdV_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dQdV_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return df_1, df_2, dQdV_real, dQdV_emulated, df_real_interped, emulated_full_cell_interped

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
        dQdV_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                 columns=['dQdV']).ewm(alpha=x[-2]).mean()
        dQdV_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV']).ewm(alpha=x[-1]).mean()

        # Include original data
        dQdV_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dQdV_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dQdV_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dQdV_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)

        V_dQdV_interper_real = interp1d(dQdV_real['Voltage_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                        dQdV_real['dQdV'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                        bounds_error=False, fill_value=0)
        V_SOC_interper_real = interp1d(dQdV_real['Voltage_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                       dQdV_real['SOC_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        V_dQdV_interper_emulated = interp1d(dQdV_emulated['Voltage_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                            dQdV_emulated['dQdV'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                            bounds_error=False, fill_value=0)
        V_SOC_interper_emulated = interp1d(dQdV_emulated['Voltage_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                           dQdV_emulated['SOC_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dQdV_over_V_real = pd.DataFrame(V_dQdV_interper_real(voltage_vec), columns=['dQdV']).fillna(0)
        dQdV_over_V_real['SOC'] = V_SOC_interper_real(voltage_vec)
        dQdV_over_V_real['Voltage'] = voltage_vec

        dQdV_over_V_emulated = pd.DataFrame(V_dQdV_interper_emulated(voltage_vec), columns=['dQdV']).fillna(0)
        dQdV_over_V_emulated['SOC'] = V_SOC_interper_emulated(voltage_vec)
        dQdV_over_V_emulated['Voltage'] = voltage_vec

        return df_1, df_2, dQdV_over_V_real, dQdV_over_V_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dQdV_over_V_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, dQdV_real, dQdV_emulated, df_real_interped, emulated_full_cell_interped = \
            self.get_dQdV_over_V_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dQdV_real['dQdV'], dQdV_emulated['dQdV']) + 0.01 * len(
            dQdV_emulated['dQdV'].loc[dQdV_emulated['dQdV'].isna()])
        return error

    def get_dVdQ_over_Q_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        # Calculate dVdQ from full cell profiles
        dVdQ_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'], df_real_interped['SOC_aligned']),
                                 columns=['dVdQ'])
        dVdQ_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['SOC_aligned']),
            columns=['dVdQ'])

        # Include original data
        dVdQ_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dVdQ_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dVdQ_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dVdQ_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return df_1, df_2, dVdQ_real, dVdQ_emulated, df_real_interped, emulated_full_cell_interped

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
        dVdQ_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                 columns=['dVdQ'])
        dVdQ_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dVdQ'])

        # Include original data
        dVdQ_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dVdQ_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dVdQ_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dVdQ_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)
        V_dVdQ_interper_real = interp1d(dVdQ_real['Voltage_aligned'].loc[~dVdQ_real['Voltage_aligned'].isna()],
                                        dVdQ_real['dVdQ'].loc[~dVdQ_real['Voltage_aligned'].isna()]
                                        , bounds_error=False, fill_value=0)
        V_SOC_interper_real = interp1d(dVdQ_real['Voltage_aligned'].loc[~dVdQ_real['Voltage_aligned'].isna()],
                                       dVdQ_real['SOC_aligned'].loc[~dVdQ_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        V_dVdQ_interp_emulated = interp1d(dVdQ_emulated['Voltage_aligned'].loc[
                                              ~dVdQ_emulated['Voltage_aligned'].isna()],
                                          dVdQ_emulated['dVdQ'].loc[~dVdQ_emulated['Voltage_aligned'].isna()],
                                          bounds_error=False, fill_value=0)
        V_SOC_interper_emulated = interp1d(dVdQ_emulated['Voltage_aligned'].loc[
                                               ~dVdQ_emulated['Voltage_aligned'].isna()],
                                           dVdQ_emulated['SOC_aligned'].loc[~dVdQ_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dVdQ_over_V_real = pd.DataFrame(V_dVdQ_interper_real(voltage_vec), columns=['dVdQ'])
        dVdQ_over_V_real['SOC'] = V_SOC_interper_real(voltage_vec)
        dVdQ_over_V_real['Voltage'] = voltage_vec

        dVdQ_over_V_emulated = pd.DataFrame(V_dVdQ_interp_emulated(voltage_vec), columns=['dVdQ'])
        dVdQ_over_V_emulated['SOC'] = V_SOC_interper_emulated(voltage_vec)
        dVdQ_over_V_emulated['Voltage'] = voltage_vec

        return df_1, df_2, dVdQ_over_V_real, dVdQ_over_V_emulated, df_real_interped, emulated_full_cell_interped

    def get_error_dVdQ_over_V_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, dVdQ_real, dVdQ_emulated, df_real_interped, emulated_full_cell_interped = \
            self.get_dVdQ_over_V_from_halfcell_initial_matching(x, *params)

        # Calculate distance between lines
        error = distance.euclidean(dVdQ_real['dVdQ'], dVdQ_emulated['dVdQ']) + 0.01 * len(
            dVdQ_emulated['dVdQ'].loc[dVdQ_emulated['dVdQ'].isna()])
        return error

    def blend_electrodes(self, electrode_1, electrode_2_pos, electrode_2_neg, x_2):
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
        electrode_1_Voltage_aligned = pd.DataFrame(electrode_1_interper(voltage_vec), columns=['SOC'])
        electrode_2_Voltage_aligned = pd.DataFrame(electrode_2_interper(voltage_vec), columns=['SOC'])
        electrode_1_Voltage_aligned['Voltage'] = voltage_vec
        electrode_2_Voltage_aligned['Voltage'] = voltage_vec

        df_blend_Voltage_aligned = pd.DataFrame(
            (1 - x_2) * electrode_1_Voltage_aligned['SOC'] + x_2 * electrode_2_Voltage_aligned['SOC'], columns=['SOC'])
        df_blend_Voltage_aligned['Voltage'] = electrode_1_Voltage_aligned.merge(electrode_2_Voltage_aligned,
                                                                                on='Voltage')['Voltage']

        df_blended_interper = interp1d(df_blend_Voltage_aligned['SOC'], df_blend_Voltage_aligned['Voltage'],
                                       bounds_error=False)
        SOC_vec = np.linspace(0, 100, 1001)

        df_blended = pd.DataFrame(df_blended_interper(SOC_vec), columns=['Voltage_aligned'])
        df_blended['SOC_aligned'] = SOC_vec

        # Modify NE to fully span 100% SOC within its valid voltage window
        df_blended_soc_mod_interper = interp1d(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()],
                                               df_blended['Voltage_aligned'].loc[~df_blended['Voltage_aligned'].isna()],
                                               bounds_error=False)
        SOC_vec = np.linspace(np.min(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()]),
                              np.max(df_blended['SOC_aligned'].loc[~df_blended['Voltage_aligned'].isna()]),
                              1001)
        df_blended_soc_mod = pd.DataFrame(df_blended_soc_mod_interper(SOC_vec), columns=['Voltage_aligned'])
        df_blended_soc_mod['SOC_aligned'] = SOC_vec / np.max(SOC_vec) * 100
        return df_blended_soc_mod

    def blend_electrodes_robust(self, NE_pristine_matched, NE_2_pos, NE_2_neg, x_NE_2):
        """

        """
        if NE_2_pos.empty:
            df_blended = NE_pristine_matched
            return df_blended

        if NE_2_neg.empty:
            df_blended = NE_pristine_matched
            return df_blended

        if x_NE_2 > 0:
            NE_2_pristine = NE_2_pos
        else:
            NE_2_pristine = NE_2_neg
            x_NE_2 = np.abs(x_NE_2)

        # match the two NE materials by SOC
        NE_pristine_matched_interper = interp1d(
            NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()]['SOC_aligned'],
            NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()]['Voltage_aligned'],
            bounds_error=False)
        NE_2_pristine_interper = interp1d(NE_2_pristine.loc[~NE_2_pristine.Voltage_aligned.isna()]['SOC_aligned'],
                                          NE_2_pristine.loc[~NE_2_pristine.Voltage_aligned.isna()]['Voltage_aligned'],
                                          bounds_error=False)

        SOC_vec = np.linspace(0, 100, 1001)

        NE_pristine_matched_len1001 = pd.DataFrame(SOC_vec, columns=['SOC_aligned'])
        NE_pristine_matched_len1001['Voltage_aligned'] = NE_pristine_matched_interper(SOC_vec)

        NE_2_pristine_interper_len1001 = pd.DataFrame(SOC_vec, columns=['SOC_aligned'])
        NE_2_pristine_interper_len1001['Voltage_aligned'] = NE_2_pristine_interper(SOC_vec)

        df_NE = self.blend_electrodes(NE_pristine_matched_len1001,
                                      NE_2_pristine_interper_len1001,
                                      pd.DataFrame(),
                                      x_NE_2)

        return df_NE


    def blend_electrodes_robust_v2(self, NE_pristine_matched, NE_2_pos, NE_2_neg, x_NE_2):
        """

        """
        if NE_2_pos.empty:
            df_blended = NE_pristine_matched
            return df_blended

        if NE_2_neg.empty:
            df_blended = NE_pristine_matched
            return df_blended

        if x_NE_2 > 0:
            NE_2_pristine = NE_2_pos
        else:
            NE_2_pristine = NE_2_neg
            x_NE_2 = np.abs(x_NE_2)

        # match the two NE materials by SOC
        SOC_vec = np.linspace(0, 100, 1001)

        NE_pristine_matched_0to100 = ((NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()][
                                           'SOC_aligned'] - np.min(
            NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()]['SOC_aligned'])) /
                                      (np.max(NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()][
                                                  'SOC_aligned']) - np.min(
                                          NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()][
                                              'SOC_aligned']))
                                      ) * 100
        NE_pristine_matched_interper = interp1d(NE_pristine_matched_0to100,
                                                NE_pristine_matched.loc[~NE_pristine_matched.Voltage_aligned.isna()][
                                                    'Voltage_aligned'],
                                                bounds_error=False)

        NE_2_pristine_interper = interp1d(NE_2_pristine.loc[~NE_2_pristine.Voltage_aligned.isna()]['SOC_aligned'],
                                          NE_2_pristine.loc[~NE_2_pristine.Voltage_aligned.isna()]['Voltage_aligned'],
                                          bounds_error=False)

        NE_pristine_matched_len1001 = pd.DataFrame(SOC_vec, columns=['SOC_aligned'])
        NE_pristine_matched_len1001['Voltage_aligned'] = NE_pristine_matched_interper(SOC_vec)

        NE_2_pristine_interper_len1001 = pd.DataFrame(SOC_vec, columns=['SOC_aligned'])
        NE_2_pristine_interper_len1001['Voltage_aligned'] = NE_2_pristine_interper(SOC_vec)

        df_NE_blended = self.blend_electrodes(NE_pristine_matched_len1001,
                                              NE_2_pristine_interper_len1001,
                                              pd.DataFrame(),
                                              x_NE_2)

        # restore blend back to original SOC span of NE_pristine_matched

        df_NE_blended_interper = interp1d(df_NE_blended['SOC_aligned'], df_NE_blended['Voltage_aligned'],
                                          bounds_error=False)  # intiializing interpolation across the blend

        len_nonNA_NE_pristine_matched = len(NE_pristine_matched.loc[
                                                ~NE_pristine_matched.Voltage_aligned.isna()])  # number of points
        # applicable to NE in NE_pristine_matched

        SOC_vec_prematching = np.linspace(np.min(df_NE_blended['SOC_aligned']),
                                          np.max(df_NE_blended['SOC_aligned']),
                                          len_nonNA_NE_pristine_matched)  # vector across blended NE with same number
        # of applicable points as original (NE_pristine_matched)

        df_NE_blended_matched = NE_pristine_matched.copy()

        df_NE_slice_for_matched = df_NE_blended_interper(SOC_vec_prematching)
        df_NE_blended_matched.at[(df_NE_blended_matched.loc[
            (~df_NE_blended_matched['Voltage_aligned'].isna())]).index, 'Voltage_aligned'] = df_NE_slice_for_matched

        return df_NE_blended_matched

    def halfcell_initial_matching_v2(self, x, *params):
        """
        Augments halfcell voltage profiles by scaling and translating them. Typically used in an optimization routine
        to fit the emulated full cell profile to a real cell profile.

        Inputs:
        x: an array of 2 or 3 parameters containing scale_ratio, offset, and optionally NE_2_x
        df_real: dataframe for the first diagnostic (pristine) of the real full cell. Columns for SOC (ev)
        df_pe: dataframe for the positive electrode. Columns for SOC (evenly spaced) and Voltage.
        df_ne_1: dataframe for the primary material in the negative electrode. Columns for SOC (evenly spaced)
            and Voltage.
        df_ne_2: dataframe for the secondary material in the negative electrode. Columns for SOC (evenly spaced)
            and Voltage. Supply empty DataFrame if not emulating a blend from two known elelctrodes.
        """

        df_real, df_pe, df_ne_1, df_ne_2_pos, df_ne_2_neg = params

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
            df_ne = self.blend_electrodes(df_ne_1, df_ne_2_pos, df_ne_2_neg, x_ne_2)  # _robust_v2

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
                                 df_1['Voltage_aligned']
                                 , bounds_error=False, fill_value=np.nan)
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

        ### centering
        centering_value = - np.min(df_3['SOC_aligned'].iloc[np.argmin(np.abs(df_3['Voltage_aligned'] - 2.7))])

        emulated_full_cell_centered = df_3.copy()
        emulated_full_cell_centered['SOC_aligned'] = df_3['SOC_aligned'] + centering_value

        pe_out_centered = df_1.copy()
        pe_out_centered['SOC_aligned'] = df_1['SOC_aligned'] + centering_value

        ne_out_centered = df_2.copy()
        ne_out_centered['SOC_aligned'] = df_2['SOC_aligned'] + centering_value

        ### Scaling

        emulated_full_cell_centered.loc[(emulated_full_cell_centered['Voltage_aligned'] > self.upper_voltage) | (
                    emulated_full_cell_centered['Voltage_aligned'] < self.lower_voltage)] = np.nan

        scaling_value = np.max(emulated_full_cell_centered['SOC_aligned'].loc[~emulated_full_cell_centered[
            'Voltage_aligned'].isna()])  # value to scale emulated back to 100% SOC

        emulated_full_cell_centered_scaled = emulated_full_cell_centered.copy()
        emulated_full_cell_centered_scaled['SOC_aligned'] = emulated_full_cell_centered['SOC_aligned'] / scaling_value * 100

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
                                           bounds_error=False, fill_value=(self.lower_voltage, self.upper_voltage))

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

        # Inputs:
        # x: an array of 4 or 5 parameters containing scale_PE, offset_PE,scale_PE, scale_NE, offset_NE, and optionally NE_2_x
        # df_real: dataframe for the first diagnostic (pristine) of the real full cell. Columns for SOC (ev)
        # df_PE: dataframe for the positive electrode. Columns for SOC (evenly spaced) and Voltage.
        # df_NE_1: dataframe for the primary material in the negative electrode. Columns for SOC (evenly spaced) and Voltage.
        # df_NE_2: dataframe for the secondary material in the negative electrode. Columns for SOC (evenly spaced) and Voltage. Supply empty DataFrame if not emulating a blend from two known elelctrodes.
        """

        df_real, df_PE, df_NE_1, df_NE_2 = params

        scale_PE = x[0]
        offset_PE = x[1]
        scale_NE = x[2]
        offset_NE = x[3]

        if df_NE_2.empty:
            # one-material anode
            df_NE = pd.DataFrame()
            df_NE['Voltage_aligned'] = df_NE_1['Voltage_aligned']
            df_NE['SOC_aligned'] = df_NE_1['SOC_aligned']
        else:
            # blended anode
            NE_2_x = x[4]  # fraction of NE_2
            df_NE = pd.DataFrame()
            df_NE['Voltage_aligned'] = (NE_2_x * df_NE_2['Voltage_aligned'] + (1 - NE_2_x) * df_NE_1['Voltage_aligned'])
            df_NE['SOC_aligned'] = df_NE_1['SOC_aligned']

        # shifted cathode
        shifted_PE = df_PE.copy()
        shifted_PE['SOC_aligned'] = shifted_PE['SOC_aligned'] * scale_PE + offset_PE

        # shifted anode
        shifted_NE = df_NE.copy()
        shifted_NE['SOC_aligned'] = shifted_NE['SOC_aligned'] * scale_NE + offset_NE

        # Interpolate across the max and min SOC of the half-cell dfs
        df_1 = shifted_PE.copy()
        df_2 = shifted_NE.copy()
        min_SOC = np.min((np.min(df_1['SOC_aligned']), np.min(df_2['SOC_aligned'])))
        max_SOC = np.max((np.max(df_1['SOC_aligned']), np.max(df_2['SOC_aligned'])))
        SOC_vec = np.linspace(min_SOC, max_SOC, 1001)

        df_1_interper = interp1d(df_1['SOC_aligned'], df_1['Voltage_aligned'], bounds_error=False)
        df_1['SOC_aligned'] = SOC_vec.copy()
        df_1['Voltage_aligned'] = df_1_interper(SOC_vec)

        df_2_interper = interp1d(df_2['SOC_aligned'], df_2['Voltage_aligned'], bounds_error=False)
        df_2['SOC_aligned'] = SOC_vec.copy()
        df_2['Voltage_aligned'] = df_2_interper(SOC_vec)

        # Calculate the full-cell profile
        df_3 = pd.DataFrame()
        df_3['Voltage_aligned'] = df_1['Voltage_aligned'].subtract(df_2['Voltage_aligned'])
        df_3.loc[(df_3['Voltage_aligned'] > 4.2) | (df_3['Voltage_aligned'] < 2.7)] = np.nan
        df_3['SOC_aligned'] = df_2['SOC_aligned']

        ### centering
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

        ###

        # Make new interpolation across SOC for full-cell error calculation
        min_SOC_full_cell = np.min((np.min(df_3.loc[~df_3.Voltage_aligned.isna()].SOC_aligned),
                                    np.min(df_real.loc[~df_real.Voltage_aligned.isna()].SOC_aligned)))
        max_SOC_full_cell = np.max((np.max(df_3.loc[~df_3.Voltage_aligned.isna()].SOC_aligned),
                                    np.max(df_real.loc[~df_real.Voltage_aligned.isna()].SOC_aligned)))
        SOC_vec_full_cell = np.linspace(min_SOC_full_cell, max_SOC_full_cell, 1001)

        emulated_full_cell_interper = interp1d(emulated_full_cell_centered.SOC_aligned,
                                               emulated_full_cell_centered.Voltage_aligned, bounds_error=False)
        real_full_cell_interper = interp1d(df_real.SOC_aligned,
                                           df_real.Voltage_aligned, bounds_error=False)

        # Interpolate the emulated full-cell profile
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['SOC_aligned'] = SOC_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(SOC_vec_full_cell)

        # Interpolate the true full-cell profile
        df_real_interped = emulated_full_cell_interped.copy()
        df_real_interped['SOC_aligned'] = SOC_vec_full_cell
        df_real_interped['Voltage_aligned'] = real_full_cell_interper(SOC_vec_full_cell)

        return df_1, df_2, df_real_interped, emulated_full_cell_interped

    def _get_error_from_halfcell_initial_matching(self, x, *params):
        df_1, df_2, df_real_interped, emulated_full_cell_interped = self.halfcell_initial_matching_v2(x, *params)

        error = distance.euclidean(df_real_interped['Voltage_aligned'],
                                   emulated_full_cell_interped['Voltage_aligned']) + 0.01 * len(
            emulated_full_cell_interped['Voltage_aligned'].loc[emulated_full_cell_interped['Voltage_aligned'].isna()])

        return error

    def _impose_degradation(self,
                            pe_pristine=pd.DataFrame(),
                            ne_1_pristine=pd.DataFrame(),
                            ne_2_pristine_pos=pd.DataFrame(),
                            ne_2_pristine_neg=pd.DataFrame(),
                            LLI=0.0, LAM_PE=0.0, LAM_NE=0.0, x_NE_2=0.0):
        pe_translation = 0
        pe_shrinkage = 0
        ne_translation = 0
        ne_shrinkage = 0

        # Blend negative electrodes
        ne_pristine = self.blend_electrodes(ne_1_pristine, ne_2_pristine_pos, ne_2_pristine_neg, x_NE_2)

        # Update degradation shifts for LLI
        ne_translation += LLI

        # Update degradation shifts for LAM_PE
        upper_voltage_limit = self.upper_voltage
        pe_soc_setpoint = pe_pristine['SOC_aligned'].loc[
            np.argmin(np.abs(pe_pristine['Voltage_aligned']
                             - ne_pristine[
                                 'Voltage_aligned'] - upper_voltage_limit))]  # SOC at which the upper voltage
        # limit is hit
        pe_translation += LAM_PE * pe_soc_setpoint / 100  # correction for shrinkage to ensure the locking of
        # the upper voltage SOC
        pe_shrinkage += LAM_PE  # shrinkage of PE capacity due to LAM_PE

        #  Update degradation shifts for LAM_NE
        lower_voltage_limit = self.lower_voltage
        ne_soc_setpoint = ne_pristine['SOC_aligned'].loc[
            np.argmin((pe_pristine['Voltage_aligned']
                       - ne_pristine[
                           'Voltage_aligned'] - lower_voltage_limit))]  # SOC at which the lower voltage limit is hit
        ne_translation += LAM_NE * ne_soc_setpoint / 100  # correction for shrinkage to ensure the locking of the
        # lower voltage SOC
        ne_shrinkage += LAM_NE  # shrinkage of NE capacity due to LAM_NE

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

    def get_halfcell_voltages(self, pe_out_centered, ne_out_centered):
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

    def get_dQdV_over_V_from_degradation_matching(self, x, *params):
        PE_out_centered, NE_out_centered, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_v3(x, *params)

        # Calculate dQdV from full cell profiles
        dQdV_real = pd.DataFrame(np.gradient(df_real_interped['SOC_aligned'], df_real_interped['Voltage_aligned']),
                                 columns=['dQdV']).ewm(alpha=x[-2]).mean()
        dQdV_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['SOC_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV']).ewm(alpha=x[-1]).mean()

        # Include original data
        dQdV_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dQdV_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dQdV_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dQdV_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over V
        voltage_vec = np.linspace(2.7, 4.2, 1001)

        V_dQdV_interper_real = interp1d(dQdV_real['Voltage_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                        dQdV_real['dQdV'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                        bounds_error=False, fill_value=0)
        V_SOC_interper_real = interp1d(dQdV_real['Voltage_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                       dQdV_real['SOC_aligned'].loc[~dQdV_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, 100))

        V_dQdV_interper_emulated = interp1d(dQdV_emulated['Voltage_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                            dQdV_emulated['dQdV'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                            bounds_error=False, fill_value=0)
        V_SOC_interper_emulated = interp1d(dQdV_emulated['Voltage_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                           dQdV_emulated['SOC_aligned'].loc[~dQdV_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0, 100))

        dQdV_over_V_real = pd.DataFrame(V_dQdV_interper_real(voltage_vec), columns=['dQdV']).fillna(0)
        dQdV_over_V_real['SOC_aligned'] = V_SOC_interper_real(voltage_vec)
        dQdV_over_V_real['Voltage_aligned'] = voltage_vec

        dQdV_over_V_emulated = pd.DataFrame(V_dQdV_interper_emulated(voltage_vec), columns=['dQdV']).fillna(0)
        dQdV_over_V_emulated['SOC_aligned'] = V_SOC_interper_emulated(voltage_vec)
        dQdV_over_V_emulated['Voltage_aligned'] = voltage_vec

        return PE_out_centered, NE_out_centered, dQdV_over_V_real, dQdV_over_V_emulated, df_real_interped, emulated_full_cell_interped


    def get_error_dQdV_over_V_from_degradation_matching(self, x, *params):
        try:
            (PE_out_centered, NE_out_centered,
             dQdV_over_V_real, dQdV_over_V_emulated,
             df_real_interped, emulated_full_cell_interped) = self.get_dQdV_over_V_from_degradation_matching(x, *params)
            error = distance.euclidean(dQdV_over_V_real['dQdV'], dQdV_over_V_emulated['dQdV']) + 0.01 * len(
                dQdV_over_V_emulated['dQdV'].loc[dQdV_over_V_emulated['dQdV'].isna()])
        except ValueError:
            error = 1000  # set an error for cases where calcuation fails
        return error


    def get_dVdQ_over_SOC_from_degradation_matching(self, x, *params):
        PE_out_centered, NE_out_centered, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_v3(x, *params)

        # Calculate dVdQ from full cell profiles
        dVdQ_over_SOC_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'],
                                                      df_real_interped['SOC_aligned']),
                                          columns=['dVdQ']).ewm(alpha=x[-2]).mean()
        dVdQ_over_SOC_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['SOC_aligned']),
            columns=['dVdQ']).ewm(alpha=x[-1]).mean()

        # Include original data
        dVdQ_over_SOC_real['SOC_aligned'] = df_real_interped['SOC_aligned']
        dVdQ_over_SOC_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dVdQ_over_SOC_emulated['SOC_aligned'] = emulated_full_cell_interped['SOC_aligned']
        dVdQ_over_SOC_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate over Q
        # ^^ already done in this case as standard data template is over Q

        return (PE_out_centered,
                NE_out_centered,
                dVdQ_over_SOC_real,
                dVdQ_over_SOC_emulated,
                df_real_interped,
                emulated_full_cell_interped)

    def get_error_dVdQ_over_SOC_from_degradation_matching(self, x, *params):
        try:
            (PE_out_centered, NE_out_centered,
             dVdQ_over_SOC_real, dVdQ_over_SOC_emulated,
             df_real_interped, emulated_full_cell_interped) = \
                self.get_dVdQ_over_SOC_from_degradation_matching(x, *params)
            error = distance.euclidean(dVdQ_over_SOC_real['dVdQ'], dVdQ_over_SOC_emulated['dVdQ']) + 0.01 * len(
                dVdQ_over_SOC_emulated['dVdQ'].loc[dVdQ_over_SOC_emulated['dVdQ'].isna()])
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
            (PE_out_centered, NE_out_centered, real_aligned, emulated_aligned
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

        emulated_full_cell_with_degradation_centered['SOC_aligned'] = (emulated_full_cell_with_degradation['SOC_aligned'] +
                                                                       centering_value)

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

        emulated_full_cell_with_degradation_centered['SOC_aligned'] = (emulated_full_cell_with_degradation['SOC_aligned'] +
                                                                       centering_value)

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

    def halfcell_degradation_matching(self, x, *params):
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

        emulated_full_cell_with_degradation_centered['SOC_aligned'] = (emulated_full_cell_with_degradation['SOC_aligned'] +
                                                                       centering_value)

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
        emulated_interper = interp1d(emulated_full_cell_with_degradation_centered['SOC_aligned']
                                     , emulated_full_cell_with_degradation_centered['Voltage_aligned'],
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

        # Calculate distance between lines
        # Dynamic time warping error metric
        alignment = dtw(emulated_aligned['Voltage_aligned'].dropna(),
                        real_aligned['Voltage_aligned'].dropna(),
                        keep_internals=True, open_begin=False, open_end=True, step_pattern='asymmetric')
        error = alignment.distance + 0.01 * len(emulated_aligned['Voltage_aligned'].loc[
                                                    emulated_aligned['Voltage_aligned'].isna()])

        return error, emulated_aligned, real_aligned, pe_out_centered, ne_out_centered

    def plot_voltage_curves_for_cell(self,
                                     processed_cycler_run,
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