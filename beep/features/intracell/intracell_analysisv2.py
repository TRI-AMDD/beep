import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.optimize import differential_evolution


class IntracellAnalysisV2:
    # IA constants
    FC_UPPER_VOLTAGE = 4.20
    FC_LOWER_VOLTAGE = 2.70
    NE_UPPER_VOLTAGE = 0.01
    NE_LOWER_VOLTAGE = 1.50
    PE_UPPER_VOLTAGE = 4.30
    PE_LOWER_VOLTAGE = 2.86
    THRESHOLD = 4.84 * 0.0

    def __init__(self,
                 pe_pristine_files_dict={},
                 pe_pristine_usecols=['Ecell/V','Capacity/mA.h',
                                      'SOC_aligned','c_rate',
                                      'BVV_c_rate','Voltage_aligned'],
                 ne_1_pristine_files_dict={},
                 ne_1_pristine_usecols=['Ecell/V','Capacity/mA.h',
                                      'SOC_aligned','c_rate',
                                      'BVV_c_rate','Voltage_aligned'],
                 Q_fc_nom=4.84,
                 C_nom=-0.2,
                 cycle_type='rpt_0.2C',
                 step_type=0,
                 error_type='V-Q',
                 error_weighting='uniform',
                 dvdq_bound=None,
                 ne_2pos_file=None,
                 ne_2neg_file=None
                 ):
        """
        Invokes the cell electrode analysis class. This is a class designed to fit the cell and electrode
        parameters in order to determine changes of electrodes within the full cell from only full cell cycling data.
        Args:
            pe_pristine_dict (str): dictionary for the half cell rate data of the pristine (uncycled) positive
                electrode. Keys are the rate and the entries are dataframes of Voltage vs. SOC.
            ne_1_pristine_dict (str): dictionary for the half cell rate data of the pristine (uncycled) negative
                electrode. Keys are the rate and the entris are dataframes of Voltage vs. SOC.
            cycle_type (str): type of diagnostic cycle for the fitting
            step_type (int): charge or discharge (0 for charge, 1 for discharge)
            error_type (str): defines which error metric is to be used 
            ne_2neg_file (str): file name of the data for the negative component of the anode
            ne_2pos_file (str): file name of the data for the positive component of the anode
        """
        
        self.pe_pristine_files_dict = pe_pristine_files_dict
        self.pe_pristine_dict = {}
        [self.pe_pristine_dict.update(
            {key:pd.read_csv(pe_pristine_files_dict[key],
                             usecols=pe_pristine_usecols)
                }) for key in pe_pristine_files_dict]
        
        self.ne_1_pristine_files_dict = ne_1_pristine_files_dict
        self.ne_1_pristine_dict = {}
        [self.ne_1_pristine_dict.update(
            {key:pd.read_csv(ne_1_pristine_files_dict[key],
                             usecols=ne_1_pristine_usecols)
                }) for key in ne_1_pristine_files_dict]
        
        
        
        self.Q_fc_nom = Q_fc_nom
        self.C_nom = C_nom

        if ne_2neg_file and ne_2pos_file:
            self.ne_2_pristine_pos = pd.read_csv(ne_2pos_file)
            self.ne_2_pristine_neg = pd.read_csv(ne_2neg_file)
        else:
            self.ne_2_pristine_pos = pd.DataFrame()
            self.ne_2_pristine_neg = pd.DataFrame()
            
        if step_type == 0:
            self.capacity_col = 'charge_capacity'
        else:
            self.capacity_col = 'discharge_capacity'

        self.cycle_type = cycle_type
        self.step_type = step_type
        self.error_type = error_type
        self.error_weighting = error_weighting
        self.dvdq_bound = dvdq_bound


    def process_beep_cycle_data_for_candidate_halfcell_analysis_ah(self,
                                                                cell_struct,
                                                                cycle_index):
        """
        Ingests BEEP structured cycling data and cycle_index and returns
                a Dataframe of evenly spaced capacity with corresponding voltage.
        
        Inputs:
        cell_struct (MaccorDatapath): BEEP structured cycling data
        cycle_index (int): cycle number at which to evaluate
        
        Outputs:
        real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned (evenly spaced)
                and Voltage_aligned
        """

        # filter the data down to the diagnostic type of interest
        diag_type_cycles = cell_struct.diagnostic_data.loc[cell_struct.diagnostic_data['cycle_type'] == self.cycle_type]
        real_cell_candidate_profile = diag_type_cycles.loc[
            (diag_type_cycles.cycle_index == cycle_index)
            & (diag_type_cycles.step_type == self.step_type)  # step_type = 0 is charge, 1 is discharge
            & (diag_type_cycles.voltage < self.FC_UPPER_VOLTAGE)
            & (diag_type_cycles[self.capacity_col] > 0)][['voltage', self.capacity_col]]
        
        # modify discharge data to match the V-Q directionality of charge data (increasing capacity w/ increasing voltage)
        if self.step_type == 1:
            real_cell_candidate_profile[self.capacity_col] = np.nanmax(real_cell_candidate_profile[self.capacity_col]) - real_cell_candidate_profile[self.capacity_col].copy()
#         real_cell_candidate_profile['voltage'] = np.flipud(real_cell_candidate_profile['voltage'].copy())
        
        
        # renaming capacity,voltage column
        real_cell_candidate_profile['Q'] =  real_cell_candidate_profile[self.capacity_col]
                                                     
        real_cell_candidate_profile['Voltage'] = real_cell_candidate_profile['voltage']
        real_cell_candidate_profile.drop('voltage', axis=1, inplace=True)
        
        # interpolate voltage along evenly spaced capacity axis
        Q_vec = np.linspace(0, np.max(real_cell_candidate_profile['Q']),1001)  

        real_cell_candidate_profile_aligned = pd.DataFrame()
        real_cell_candidate_profile_interper = interp1d(real_cell_candidate_profile['Q'],
                                                               real_cell_candidate_profile['Voltage'],
                                                               bounds_error=False,
                                                               fill_value=(self.FC_LOWER_VOLTAGE, self.FC_UPPER_VOLTAGE))
        real_cell_candidate_profile_aligned['Voltage_aligned'] = real_cell_candidate_profile_interper(
            Q_vec)

        real_cell_candidate_profile_aligned['Q_aligned'] = Q_vec

        return real_cell_candidate_profile_aligned
    
    def _impose_electrode_scale(self,
                            pe_pristine=pd.DataFrame(),
                            ne_1_pristine=pd.DataFrame(),
                            ne_2_pristine_pos=pd.DataFrame(),
                            ne_2_pristine_neg=pd.DataFrame(),
                            lli=0.0, Q_pe=0.0, Q_ne=0.0, x_ne_2=0.0):
        
        """
        Scales the reference electrodes according to specified capacities and 
        offsets their capacities according to lli. Blends negative electrode materials.
        
        Inputs:
        pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                electrode
        ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                electrode
        ne_2_pos (Dataframe): half cell data for the positive component of the anode
        ne_2_neg (Dataframe): half cell data for the negative component of the anode
        lli (float): Loss of Lithium Inventory - capacity of the misalignment between 
                cathode and anode zero-capacity
        Q_pe (float): capacity of the positive electrode (cathode)
        Q_ne (float): capacity of the negative electrode (anode)
        x_ne_2 (float): fraction of ne_2_pristine_pos or ne_2_pristine_neg 
                (positive or negative value, respectively) to ne_1_pristine
        
        Outputs:
        pe_degraded (Dataframe): positive electrode with imposed capacity 
                scale to emulate degradation
        ne_degraded (Dataframe): negative electrode with imposed capacity 
                scale and capacity offset to emulate degradation
        """

        # Blend negative electrodes
        ne_pristine = blend_electrodes(ne_1_pristine, ne_2_pristine_pos, ne_2_pristine_neg, x_ne_2)
        
        # rescaling pristine electrodes to Q_pe and Q_ne
        pe_Q_scaled = pe_pristine.copy()
        pe_Q_scaled['Q_aligned'] = (pe_Q_scaled['SOC_aligned']/np.nanmax(pe_Q_scaled['SOC_aligned']))*Q_pe
        ne_Q_scaled = ne_pristine.copy()
        ne_Q_scaled['Q_aligned'] = (ne_Q_scaled['SOC_aligned']/np.nanmax(ne_Q_scaled['SOC_aligned']))*Q_ne
        
        # translate pristine ne electrode with lli
        ne_Q_scaled['Q_aligned'] = ne_Q_scaled['Q_aligned'] + lli
        
        # Re-interpolate to align dataframes for differencing
        lower_Q = np.min((np.min(pe_Q_scaled['Q_aligned']),
                            np.min(ne_Q_scaled['Q_aligned'])))
        upper_Q = np.max((np.max(pe_Q_scaled['Q_aligned']),
                            np.max(ne_Q_scaled['Q_aligned'])))
        Q_vec = np.linspace(lower_Q, upper_Q, 1001)
        
        # Actually aligning the electrode Q's
        pe_pristine_interper = interp1d(pe_Q_scaled['Q_aligned'],
                                        pe_Q_scaled['Voltage_aligned'], bounds_error=False)
        pe_degraded = pe_Q_scaled.copy()
        pe_degraded['Q_aligned'] = Q_vec
        pe_degraded['Voltage_aligned'] = pe_pristine_interper(Q_vec)
        

        ne_pristine_interper = interp1d(ne_Q_scaled['Q_aligned'],
                                        ne_Q_scaled['Voltage_aligned'], bounds_error=False)
        ne_degraded = ne_Q_scaled.copy()
        ne_degraded['Q_aligned'] = Q_vec
        ne_degraded['Voltage_aligned'] = ne_pristine_interper(Q_vec)

        # Returning pe and ne degraded on an Ah basis
        return pe_degraded, ne_degraded
    
    def halfcell_degradation_matching_ah(self, x, *params):
        """
        Calls underlying functions to impose degradation through electrode 
        capacity scale and alignment through LLI. Modifies emulated full cell
        data to be within full cell voltage range and calibrates (zeros) capacity
        at the lowest permissible voltage. Interpolates real and emulated data onto
        a common capacity axis.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned

        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity

        df_real_aligned (Dataframe): capacity and voltage interpolated evenly across
                capacity for the real cell data
        emulated_full_cell_aligned (Dataframe): capacity and voltage interpolated evenly
                across capacity for the emulated cell data
        """
        
        lli = x[0]
        Q_pe = x[1]
        Q_ne = x[2]
        IR_coef_pe = x[3]
        IR_coef_ne = x[4]
        x_ne_2 = x[5]
        
        # Q_fc_nom (scalar) : nominal capacity of pristine cell
        # C_nom (scalar) : nominal C-rate of pristine cell
        Q_fc_nom = self.Q_fc_nom 
        C_nom = self.C_nom
        
        pe_pristine_dict, ne_1_pristine_dict, ne_2_pristine_pos, ne_2_pristine_neg, real_cell_candidate_profile_aligned= params
        
        if len(pe_pristine_dict) == 1:
            pe_pristine = pe_pristine_dict[list(pe_pristine_dict.keys())[0]]
        elif len(pe_pristine_dict) > 1:
            pe_at_I_eff, R_at_SOC_pe = (
                self._get_effective_electrode_rate_data(pe_pristine_dict, Q_pe, Q_fc_nom, C_nom, IR_coef_pe)
            )
            pe_pristine = pe_at_I_eff # the effective rate profile is now the reference profile at pristine state
        else:
            raise ValueError('No dictionary entries in pe_pristine_dict')
            
        if len(ne_1_pristine_dict) == 1:
            ne_1_pristine = ne_1_pristine_dict[list(ne_1_pristine_dict.keys())[0]]
        elif len(ne_1_pristine_dict) > 1:
            
            ne_at_I_eff, R_at_SOC_ne = (
                self._get_effective_electrode_rate_data( ne_1_pristine_dict, Q_ne, Q_fc_nom, C_nom, IR_coef_ne)
                                                        )
            ne_1_pristine = ne_at_I_eff # the effective rate profile is now the reference profile at pristine state
        else:
            raise ValueError('No dictionary entries in ne_pristine_dict')

        # output degraded ne and pe (on a AH basis, with electrode alignment (NaNs for voltage, when no capacity actually at the corresponding capacity index))
        pe_out, ne_out = self._impose_electrode_scale(pe_pristine, ne_1_pristine,
                                                  ne_2_pristine_pos, ne_2_pristine_neg,
                                                  lli, Q_pe,
                                                  Q_ne, x_ne_2)
        
        # PE - NE = full cell voltage
        emulated_full_cell_with_degradation = pd.DataFrame()
        emulated_full_cell_with_degradation['Q_aligned'] = pe_out['Q_aligned'].copy()
        emulated_full_cell_with_degradation['Voltage_aligned'] = pe_out['Voltage_aligned'] - ne_out['Voltage_aligned']
        
        # Replace emulated full cell values outside of voltage range with NaN
        emulated_full_cell_with_degradation['Voltage_aligned'].loc[
            emulated_full_cell_with_degradation['Voltage_aligned'] < self.FC_LOWER_VOLTAGE] = np.nan
        emulated_full_cell_with_degradation['Voltage_aligned'].loc[
            emulated_full_cell_with_degradation['Voltage_aligned'] > self.FC_UPPER_VOLTAGE] = np.nan
        
        ## Center the emulated full cell and half cell curves onto the same Q at which the real (degraded) capacity measurement started (self.FC_LOWER_VOLTAGE)
        emulated_full_cell_with_degradation_zeroed = pd.DataFrame()

        emulated_full_cell_with_degradation_zeroed['Voltage_aligned'] = emulated_full_cell_with_degradation[
            'Voltage_aligned'].copy()

        zeroing_value = emulated_full_cell_with_degradation['Q_aligned'].loc[
            np.nanargmin(emulated_full_cell_with_degradation['Voltage_aligned'])
                                                                            ]
    
        emulated_full_cell_with_degradation_zeroed['Q_aligned'] = \
            (emulated_full_cell_with_degradation['Q_aligned'].copy() - zeroing_value)

        pe_out_zeroed = pe_out.copy()
        pe_out_zeroed['Q_aligned'] = pe_out['Q_aligned'] - zeroing_value

        ne_out_zeroed = ne_out.copy()
        ne_out_zeroed['Q_aligned'] = ne_out['Q_aligned'] - zeroing_value

        # Interpolate full cell profiles across same Q range
        min_Q = np.min(
            real_cell_candidate_profile_aligned['Q_aligned'].loc[
                ~real_cell_candidate_profile_aligned['Voltage_aligned'].isna()])
        max_Q = np.max(
            (real_cell_candidate_profile_aligned['Q_aligned'].loc[
                ~real_cell_candidate_profile_aligned['Voltage_aligned'].isna()].max(),
            emulated_full_cell_with_degradation_zeroed['Q_aligned'].loc[
                ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()].max())
                        )
        emulated_interper = interp1d(emulated_full_cell_with_degradation_zeroed['Q_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()],
                                     emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()],
                                     bounds_error=False)
        real_interper = interp1d(
            real_cell_candidate_profile_aligned['Q_aligned'].loc[
                ~real_cell_candidate_profile_aligned['Voltage_aligned'].isna()],
            real_cell_candidate_profile_aligned['Voltage_aligned'].loc[
                ~real_cell_candidate_profile_aligned['Voltage_aligned'].isna()],
            bounds_error=False)

        Q_vec = np.linspace(min_Q, max_Q, 1001)

        emulated_aligned = pd.DataFrame()
        emulated_aligned['Q_aligned'] = Q_vec
        emulated_aligned['Voltage_aligned'] = emulated_interper(Q_vec)

        real_aligned = pd.DataFrame()
        real_aligned['Q_aligned'] = Q_vec
        real_aligned['Voltage_aligned'] = real_interper(Q_vec)

        return pe_out_zeroed, ne_out_zeroed, real_aligned, emulated_aligned

    def get_dQdV_over_V_from_degradation_matching_ah(self, x, *params):
        """
        This function imposes degradation scaling ,then outputs the dQdV representation of the emulated cell data.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned
        
        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity
        dq_dv_over_v_real (Dataframe): dQdV across voltage for the real cell data
        dq_dv_over_v_emulated (Dataframe): dQdV across voltage for the emulated cell data
        df_real_interped (Dataframe): capacity and voltage interpolated evenly across
                capacity for the real cell data
        emulated_full_cell_interped (Dataframe): capacity and voltage interpolated evenly
                across capacity for the emulated cell data
        """
        
        pe_out_zeroed, ne_out_zeroed, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_ah(x, *params)

        # Calculate dQdV from full cell profiles
        dq_dv_real = pd.DataFrame(np.gradient(df_real_interped['Q_aligned'], df_real_interped['Voltage_aligned']),
                                  columns=['dQdV']).ewm(0.1).mean()
        dq_dv_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Q_aligned'], emulated_full_cell_interped['Voltage_aligned']),
            columns=['dQdV']).ewm(0.1).mean()

        # Include original data
        dq_dv_real['Q_aligned'] = df_real_interped['Q_aligned']
        dq_dv_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dq_dv_emulated['Q_aligned'] = emulated_full_cell_interped['Q_aligned']
        dq_dv_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Interpolate dQdV and Q over V, aligns real and emulated over V
        voltage_vec = np.linspace(self.FC_LOWER_VOLTAGE, self.FC_UPPER_VOLTAGE, 1001)

        v_dq_dv_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         dq_dv_real['dQdV'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                         bounds_error=False, fill_value=0)
        v_Q_interper_real = interp1d(dq_dv_real['Voltage_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       dq_dv_real['Q_aligned'].loc[~dq_dv_real['Voltage_aligned'].isna()],
                                       bounds_error=False, fill_value=(0, np.max(df_real_interped['Q_aligned'])))

        v_dq_dv_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                                 ~dq_dv_emulated['Voltage_aligned'].isna()],
                                             dq_dv_emulated['dQdV'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                             bounds_error=False, fill_value=0)
        v_Q_interper_emulated = interp1d(dq_dv_emulated['Voltage_aligned'].loc[
                                               ~dq_dv_emulated['Voltage_aligned'].isna()],
                                        dq_dv_emulated['Q_aligned'].loc[~dq_dv_emulated['Voltage_aligned'].isna()],
                                           bounds_error=False, fill_value=(0,np.max(df_real_interped['Q_aligned'])))

        dq_dv_over_v_real = pd.DataFrame(v_dq_dv_interper_real(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_real['Q_aligned'] = v_Q_interper_real(voltage_vec)
        dq_dv_over_v_real['Voltage_aligned'] = voltage_vec

        dq_dv_over_v_emulated = pd.DataFrame(v_dq_dv_interper_emulated(voltage_vec), columns=['dQdV']).fillna(0)
        dq_dv_over_v_emulated['Q_aligned'] = v_Q_interper_emulated(voltage_vec)
        dq_dv_over_v_emulated['Voltage_aligned'] = voltage_vec

        return (pe_out_zeroed,
                ne_out_zeroed,
                dq_dv_over_v_real,
                dq_dv_over_v_emulated,
                df_real_interped,
                emulated_full_cell_interped)

    def get_dVdQ_over_Q_from_degradation_matching_ah(self, x, *params):
        """
        This function imposes degradation scaling ,then outputs the dVdQ representation of the emulated cell data.
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity
        dv_dq_real (Dataframe): dVdQ across capacity for the real cell data
        dv_dq_emulated (Dataframe): dVdQ across capacity for the emulated cell data
        df_real_interped (Dataframe): capacity and voltage interpolated evenly across
                capacity for the real cell data
        emulated_full_cell_interped (Dataframe): capacity and voltage interpolated evenly
                across capacity for the emulated cell data        
        """
        
        pe_out_zeroed, ne_out_zeroed, df_real_interped, emulated_full_cell_interped = \
            self.halfcell_degradation_matching_ah(x, *params)

        # Calculate dVdQ from full cell profiles
#         dv_dq_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'], df_real_interped['Q_aligned']),
#                                   columns=['dVdQ']).ewm(alpha=0.3).mean()
#         dv_dq_emulated = pd.DataFrame(
#             np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['Q_aligned']),
#             columns=['dVdQ']).ewm(alpha=0.3).mean()
        dv_dq_real = pd.DataFrame(np.gradient(df_real_interped['Voltage_aligned'], df_real_interped['Q_aligned']),
                                  columns=['dVdQ'])
        dv_dq_emulated = pd.DataFrame(
            np.gradient(emulated_full_cell_interped['Voltage_aligned'], emulated_full_cell_interped['Q_aligned']),
            columns=['dVdQ'])

        # Include original data
        dv_dq_real['Q_aligned'] = df_real_interped['Q_aligned']
        dv_dq_real['Voltage_aligned'] = df_real_interped['Voltage_aligned']

        dv_dq_emulated['Q_aligned'] = emulated_full_cell_interped['Q_aligned']
        dv_dq_emulated['Voltage_aligned'] = emulated_full_cell_interped['Voltage_aligned']

        # Q interpolation not needed, as interpolated over Q by default
       
        return (pe_out_zeroed,
                ne_out_zeroed,
                dv_dq_real,
                dv_dq_emulated,
                df_real_interped,
                emulated_full_cell_interped)
    
    def get_V_over_Q_from_degradation_matching_ah(self, x, *params):
        """
        This function imposes degradation scaling ,then outputs the V-Q representation of the emulated cell data.
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned
        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity
        df_real_interped (Dataframe): capacity and voltage interpolated evenly across
                capacity for the real cell data
        emulated_full_cell_interped (Dataframe): capacity and voltage interpolated evenly
                across capacity for the emulated cell data
        """
        (pe_out_zeroed, ne_out_zeroed, real_aligned, emulated_aligned) = \
            self.halfcell_degradation_matching_ah(x, *params)

        min_soc_full_cell = np.min(real_aligned.loc[~real_aligned.Voltage_aligned.isna()].Q_aligned)
        max_soc_full_cell = np.max((real_aligned.loc[~real_aligned.Voltage_aligned.isna()].Q_aligned.max(),
                                    emulated_aligned.loc[~emulated_aligned.Voltage_aligned.isna()].Q_aligned.max()))

        soc_vec_full_cell = np.linspace(min_soc_full_cell, max_soc_full_cell, 1001)

        emulated_full_cell_interper = interp1d(
            emulated_aligned.Q_aligned.loc[~emulated_aligned.Voltage_aligned.isna()],
            emulated_aligned.Voltage_aligned.loc[~emulated_aligned.Voltage_aligned.isna()],
            bounds_error=False)
        real_full_cell_interper = interp1d(real_aligned.Q_aligned.loc[~real_aligned.Voltage_aligned.isna()],
                                           real_aligned.Voltage_aligned.loc[~real_aligned.Voltage_aligned.isna()],
                                           bounds_error=False)

        # Interpolate the emulated full-cell profile
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['Q_aligned'] = soc_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(soc_vec_full_cell)

        # Interpolate the true full-cell profile
        df_real_interped = emulated_full_cell_interped.copy()
        df_real_interped['Q_aligned'] = soc_vec_full_cell
        df_real_interped['Voltage_aligned'] = real_full_cell_interper(soc_vec_full_cell)
        return pe_out_zeroed, ne_out_zeroed, df_real_interped, emulated_full_cell_interped
    
    
    def get_V_over_Q_from_degradation_matching_ah_no_real(self, x, *params):
        """
        This function imposes degradation scaling ,then outputs the V-Q representation of the emulated cell data, in the absence of real cell data.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned
                        
        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity
        emulated_full_cell_interped (Dataframe): capacity and voltage interpolated evenly
                across capacity for the emulated cell data
        
        """
        (pe_out_zeroed, ne_out_zeroed, emulated_aligned) = \
            self.halfcell_degradation_matching_ah_no_real(x, *params)

        min_Q_full_cell = np.min(emulated_aligned.loc[~emulated_aligned.Voltage_aligned.isna()].Q_aligned)
        max_Q_full_cell = np.max(emulated_aligned.loc[~emulated_aligned.Voltage_aligned.isna()].Q_aligned)

        Q_vec_full_cell = np.linspace(min_Q_full_cell, max_Q_full_cell, 1001)

        emulated_full_cell_interper = interp1d(
            emulated_aligned.Q_aligned.loc[~emulated_aligned.Voltage_aligned.isna()],
            emulated_aligned.Voltage_aligned.loc[~emulated_aligned.Voltage_aligned.isna()],
            bounds_error=False)

        # Interpolate the emulated full-cell profile
        emulated_full_cell_interped = pd.DataFrame()
        emulated_full_cell_interped['Q_aligned'] = Q_vec_full_cell
        emulated_full_cell_interped['Voltage_aligned'] = emulated_full_cell_interper(Q_vec_full_cell)

        return pe_out_zeroed, ne_out_zeroed, emulated_full_cell_interped
    
    def halfcell_degradation_matching_ah_no_real(self, x, *params):
        """
        Calls underlying functions to impose degradation through electrode 
        capacity scale and alignment through LLI. Modifies emulated full cell
        data to be within full cell voltage range and calibrates (zeros) capacity
        at the lowest permissible voltage.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned
                        
        Outputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity
        emulated_aligned (Dataframe): full cell data corresponding to the imposed degradation
        """
        lli = x[0]
        Q_pe = x[1]
        Q_ne = x[2]
        x_ne_2 = x[3]

        pe_pristine, ne_1_pristine, ne_2_pristine_pos, ne_2_pristine_neg = params

        pe_out, ne_out = self._impose_electrode_scale(pe_pristine, ne_1_pristine,
                                                  ne_2_pristine_pos, ne_2_pristine_neg,
                                                  lli, Q_pe,
                                                  Q_ne, x_ne_2) #outputs degraded ne and pe (on a AH basis, with electrode alignment (NaNs for voltage, when no overlap))
        
        emulated_full_cell_with_degradation = pd.DataFrame()
        emulated_full_cell_with_degradation['Q_aligned'] = pe_out['Q_aligned'].copy()
        emulated_full_cell_with_degradation['Voltage_aligned'] = pe_out['Voltage_aligned'] - ne_out['Voltage_aligned']
        
        # Replace emulated full cell values outside of voltage range with NaN
        emulated_full_cell_with_degradation['Voltage_aligned'].loc[
            emulated_full_cell_with_degradation['Voltage_aligned'] < self.FC_LOWER_VOLTAGE] = np.nan
        emulated_full_cell_with_degradation['Voltage_aligned'].loc[
            emulated_full_cell_with_degradation['Voltage_aligned'] > self.FC_UPPER_VOLTAGE] = np.nan


        ## Center the emulated full cell and half cell curves onto the same Q at which the real (degraded) capacity measurement started (self.FC_LOWER_VOLTAGE)
        emulated_full_cell_with_degradation_zeroed = pd.DataFrame()

        emulated_full_cell_with_degradation_zeroed['Voltage_aligned'] = emulated_full_cell_with_degradation[
            'Voltage_aligned']
        
        zeroing_value = emulated_full_cell_with_degradation['Q_aligned'].loc[
            np.nanargmin(emulated_full_cell_with_degradation['Voltage_aligned'])
                                                                            ]
        
        emulated_full_cell_with_degradation_zeroed['Q_aligned'] = \
            (emulated_full_cell_with_degradation['Q_aligned'] - zeroing_value)

        pe_out_zeroed = pe_out.copy()
        pe_out_zeroed['Q_aligned'] = pe_out['Q_aligned'] - zeroing_value

        ne_out_zeroed = ne_out.copy()
        ne_out_zeroed['Q_aligned'] = ne_out['Q_aligned'] - zeroing_value

        # Interpolate full profiles across same Q range
        min_Q = np.min(
            emulated_full_cell_with_degradation_zeroed['Q_aligned'].loc[
                ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()])
        max_Q = np.max(
            emulated_full_cell_with_degradation_zeroed['Q_aligned'].loc[
                ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()])
        
        emulated_interper = interp1d(emulated_full_cell_with_degradation_zeroed['Q_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()],
                                     emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].loc[
                                         ~emulated_full_cell_with_degradation_zeroed['Voltage_aligned'].isna()],
                                     bounds_error=False)

        Q_vec = np.linspace(min_Q, max_Q, 1001)

        emulated_aligned = pd.DataFrame()
        emulated_aligned['Q_aligned'] = Q_vec
        emulated_aligned['Voltage_aligned'] = emulated_interper(Q_vec)

        return pe_out_zeroed, ne_out_zeroed, emulated_aligned
    
    def _get_error_from_degradation_matching_ah(self, x, *params):
        """
        Wrapper function which selects the correct error sub routine and returns its error value.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        
        Outputs:
            error value (float) - output of the specified error sub function
        """
        error_type = self.error_type
        if error_type == 'V-Q':
            return self._get_error_from_degradation_matching_V_Q(x,*params)[0]
        elif error_type == 'dVdQ':
            return self._get_error_from_degradation_matching_dVdQ(x,*params)[0]
        elif error_type == 'dQdV':
            return self._get_error_from_degradation_matching_dQdV(x,*params)[0]
        else:
            return self._get_error_from_degradation_matching_V_Q(x,*params)[0]
        
    def _get_error_from_degradation_matching_V_Q(self, x, *params):
        """
        Error function returning the mean standardized Euclidean distance of each point of the real curve to
                the closest value on the emulated curve in the V-Q representation.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        
        Outputs:
            error (float): output of the specified error sub function
            error_vector (array): vector containingEuclidean distance of each point of the real curve to
                the closest value on the emulated curve in the V-Q representation
            XA (Dataframe): real full cell data used for error analysis
            XB (Dataframe): emulated full cell  data used for error analysis
        """
        
        try:
            (pe_out_zeroed, ne_out_zeroed, real_aligned, emulated_aligned
             ) = self.get_V_over_Q_from_degradation_matching_ah(x, *params)
            
            XA = real_aligned.dropna()
            XB = emulated_aligned.dropna()
            error_matrix = distance.cdist(XA, XB, 'seuclidean')
            error_vector = error_matrix.min(axis = 1)
            error = error_vector.mean()
        except ValueError:
            error = 100
            return error, None, None, None
        return error, error_vector, XA, XB
    
        # Pairwise euclidean from premade dQdV
    def _get_error_from_degradation_matching_dQdV(self, x, *params):
        """
        Error function returning the mean standardized Euclidean distance of each point of the real curve to
                the closest value on the emulated curve in the dQdV representation.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        
        Outputs:
            error (float): output of the specified error sub function
            error_vector (array): vector containing Euclidean distance of each point of the real curve to
                the closest value on the emulated curve in the dQdV representation
            XA (Dataframe): real full cell data used for error analysis
            XB (Dataframe): emulated full cell  data used for error analysis
        """
        
        try:
            # Call dQdV generating function
            (PE_out_zeroed,
            NE_out_zeroed,
            dQdV_over_v_real,
            dQdV_over_v_emulated,
            df_real_interped,
            emulated_full_cell_interped) = self.get_dQdV_over_V_from_degradation_matching_ah(x, *params)

            XA = dQdV_over_v_real[[ 'Voltage_aligned','dQdV']].dropna()
            XB = dQdV_over_v_emulated[['Voltage_aligned', 'dQdV']].dropna()
            error_matrix = distance.cdist(XA, XB, 'seuclidean')
            error_vector = error_matrix.min(axis = 1)
            error = error_vector.mean()
            
        except ValueError:
            error = 100
            return error, None, None, None
        return error, error_vector, XA, XB
    
    def _get_error_from_degradation_matching_dVdQ(self, x, *params):
        """
        Error function returning the mean standardized Euclidean distance of each point of the real curve to
                the closest value on the emulated curve in the dVdQ representation.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        
        Outputs:
            error (float): output of the specified error sub function
            error_vector (array): vector containing Euclidean distance of each point of the real curve to
                the closest value on the emulated curve in the dVdQ representation
            error_vector_weighted (array): error_vector multiplied by dQdV values as weighting
            XA (Dataframe): real full cell data used for error analysis
            XB (Dataframe): emulated full cell  data used for error analysis
        """
        
#         try:
        (PE_out_zeroed,
                NE_out_zeroed,
                dVdQ_over_Q_real,
                dVdQ_over_Q_emulated,
                df_real_interped,
                emulated_full_cell_interped) = self.get_dVdQ_over_Q_from_degradation_matching_ah(x, *params)

        XA = dVdQ_over_Q_real[[ 'Q_aligned','dVdQ']].dropna()
        XB = dVdQ_over_Q_emulated[['Q_aligned', 'dVdQ']].dropna()
            
        if self.dvdq_bound is not None:
            # down-select to values with dVdQ less than 0.8 V/Ahr to eliminate high-slope/high-valued region of dVdQ
            XA = XA.loc[XA.dVdQ < self.dvdq_bound]
            XB = XB.loc[XB.dVdQ < self.dvdq_bound]

        error_matrix = distance.cdist(XA, XB, 'seuclidean')
        error_vector_raw = error_matrix.min(axis = 1)

        # apply weighting scheme
        if self.error_weighting == 'dQdV':
            dQdV_over_v_real = self.get_dQdV_over_V_from_degradation_matching_ah(x, *params)[2]
            if self.dvdq_bound is not None:
                error_vector_weighted = np.multiply(
                    error_vector_raw,dQdV_over_v_real['dQdV'].loc[(XA.dVdQ < 0.8).index])
            elif self.dvdq_bound is None:
                error_vector_weighted = np.multiply(
                    error_vector_raw,dQdV_over_v_real['dQdV'])                    
        elif self.error_weighting == 'uniform':
            error_vector_weighted = error_vector_raw
        else:
            raise ValueError('Unknown weighting scheme provided. Check the value of "error_weighting".')

        error = error_vector_weighted.mean()
            
#         except ValueError:
#             print('ValueError')
#             error = 100
#             return error, None, None, None, None
        return error, error_vector_raw, error_vector_weighted, XA, XB
    
    def _get_error_from_synthetic_fitting_ah(self, x, *params):
        """
        Wrapper function which selects the correct error sub routine and returns its error value.
        This function is specific to fitting synthetic data rather than real cycling data.
        
        Inputs:
        x (list): [LLI, Q_pe, Q_ne, x_ne_2]
        *params:       
                pe_pristine (Dataframe): half cell data of the pristine (uncycled) positive
                        electrode
                ne_pristine (Dataframe): half cell data of the pristine (uncycled) negative
                        electrode
                ne_2_pos (Dataframe): half cell data for the positive component of the anode
                ne_2_neg (Dataframe): half cell data for the negative component of the anode
                real_cell_candidate_profile_aligned (Dataframe): columns Q_aligned 
                        (evenly spaced) and Voltage_aligned        
        
        Outputs:
            error value (float) - output of the specified error sub function
        """
        
        error_type = self.error_type
        
        try:
            if error_type == 'V-Q':
                return self._get_error_from_degradation_matching_V_Q(x, *params)[0]
            elif error_type == 'dVdQ':
                return self._get_error_from_degradation_matching_dVdQ(x, *params)[0]
            elif error_type == 'dQdV':
                return self._get_error_from_degradation_matching_dQdV(x, *params)[0]
            else:
                return self._get_error_from_degradation_matching_V_Q(x, *params)[0]
        except:
            print("Can't return error")
            return 100
        
    def intracell_values_wrapper_ah(self,
                                 cycle_index,
                                 cell_struct,
                                 degradation_bounds=None
                                 ):
        """
        Wrapper function to solve capacity sizing and offset of reference electrodes to real full cell cycle data.
        
        Inputs:
        cycle_index (int): the index of the cycle of interest of the structured real cycling data
        cell_struct (MaccorDatapath): BEEP structured cycling data

        Outputs:
        loss_dict (dict): dictionary with key of cycle index and entry of a list of
                error, LLI_opt, Q_pe_opt, Q_ne_opt, x_NE_2, Q_li
        profiles_dict (dict): dictionary with key of cycle index and entry of a dictionary
                containing various key/entry pairs of resulting from the fitting
        """

        if degradation_bounds is None:
            degradation_bounds = ((0, 2.0),  # LLI
                                  (2.5, 6.5),  # Q_pe (Q_pe must be > LLI)
                                  (2.5, 6.5),  # Q_ne
                                  (1,1), #IR coef pe
                                  (1,1), #IR coef ne
                                  (1, 1),  # (-1,1) x_NE_2
                                  )

        real_cell_candidate_profile_aligned = self.process_beep_cycle_data_for_candidate_halfcell_analysis_ah(
            cell_struct,
            cycle_index)
        
        

        degradation_optimization_result = differential_evolution(self._get_error_from_degradation_matching_ah,
                                                                 degradation_bounds,
                                                                 args=(self.pe_pristine_dict,
                                                                       self.ne_1_pristine_dict,
                                                                       self.ne_2_pristine_pos,
                                                                       self.ne_2_pristine_neg,
                                                                       real_cell_candidate_profile_aligned,
                                                                       ),
                                                                 strategy='best1bin', maxiter=1000,
                                                                 popsize=15, tol=0.01, mutation=0.5,
                                                                 recombination=0.7,
                                                                 seed=0,
                                                                 callback=None, disp=False, polish=True,
                                                                 init='latinhypercube',
                                                                 atol=0, updating='deferred', workers=-1,
                                                                 constraints=()
                                                                 )
#         print(degradation_optimization_result.x) #BVV
        (PE_out_zeroed,
         NE_out_zeroed,
         dVdQ_over_Q_real,
         dVdQ_over_Q_emulated,
         df_real_interped,
         emulated_full_cell_interped) = self.get_dVdQ_over_Q_from_degradation_matching_ah(
            degradation_optimization_result.x,
            self.pe_pristine_dict,
            self.ne_1_pristine_dict,
            self.ne_2_pristine_pos,
            self.ne_2_pristine_neg,
            real_cell_candidate_profile_aligned)
    
        (dQdV_over_v_real,
         dQdV_over_v_emulated) = self.get_dQdV_over_V_from_degradation_matching_ah(
                degradation_optimization_result.x,
                self.pe_pristine_dict,
                self.ne_1_pristine_dict,
                self.ne_2_pristine_pos,
                self.ne_2_pristine_neg,
                real_cell_candidate_profile_aligned)[2:4]
        #
        electrode_info_df = self.get_electrode_info_ah(PE_out_zeroed, NE_out_zeroed)
        #
        error = degradation_optimization_result.fun
        LLI_opt = degradation_optimization_result.x[0]
        Q_pe_opt = degradation_optimization_result.x[1]
        Q_ne_opt = degradation_optimization_result.x[2]
        IR_coef_pe_opt = degradation_optimization_result.x[3]
        IR_coef_ne_opt = degradation_optimization_result.x[4]
        x_NE_2 = degradation_optimization_result.x[5]
        
        loss_dict = {cycle_index: np.append([error, LLI_opt, Q_pe_opt, Q_ne_opt,IR_coef_pe_opt,IR_coef_ne_opt,
                                             x_NE_2],
                                             electrode_info_df.iloc[-1].values)
                     }
        
        
        error, error_vector_raw, error_vector_weighted, XA, XB = self._get_error_from_degradation_matching_dVdQ(
            [LLI_opt, Q_pe_opt, Q_ne_opt,IR_coef_pe_opt,IR_coef_ne_opt, x_NE_2],
            self.pe_pristine_dict,
            self.ne_1_pristine_dict,
            self.ne_2_pristine_pos,
            self.ne_2_pristine_neg,
            real_cell_candidate_profile_aligned)
        
        profiles_per_cycle_dict = {'NE_zeroed' : NE_out_zeroed, 
                                   'PE_zeroed' : PE_out_zeroed, 
                                   'dVdQ_over_Q_real': dVdQ_over_Q_real,
                                   'dVdQ_over_Q_emulated': dVdQ_over_Q_emulated,
                                   'dQdV_over_v_real':dQdV_over_v_real,
                                   'dQdV_over_v_emulated':dQdV_over_v_emulated,
                                   'df_real_interped': df_real_interped,
                                   'emulated_full_cell_interped': emulated_full_cell_interped ,
                                   'real_cell_candidate_profile_aligned': real_cell_candidate_profile_aligned,
                                   'error_vector_raw':error_vector_raw,
                                   'error_vector_weighted':error_vector_weighted,
                                   'XA':XA,
                                   'XB':XB
                                  }
        
        profiles_dict = {cycle_index: profiles_per_cycle_dict}

        return loss_dict, profiles_dict
    
    def solve_emulated_degradation(self,
                                 forward_simulated_profile,
                                 degradation_bounds=None
                                 ):
        
        """
        
        
        """

        if degradation_bounds is None:
            degradation_bounds = ((0, 2.0),  # LLI
                                  (2.5, 6.5),  # Q_pe
                                  (2.5, 6.5),  # Q_ne
                                  (1, 1),  # (-1,1) x_NE_2
                                  )

        degradation_optimization_result = differential_evolution(self._get_error_from_synthetic_fitting_ah,
                                                                 degradation_bounds,
                                                                 args=(self.pe_pristine_dict,
                                                                       self.ne_1_pristine_dict,
                                                                       self.ne_2_pristine_pos,
                                                                       self.ne_2_pristine_neg,
                                                                       forward_simulated_profile,
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

        return degradation_optimization_result
    
    def get_electrode_info_ah(self, pe_out_zeroed, ne_out_zeroed):
        """
        Calculates a variety of half-cell metrics at various positions in the full-cell profile.

        Inputs:
        pe_out_zeroed (Dataframe): cathode capacity and voltage columns scaled,
                offset, and aligned along capacity
        ne_out_zeroed (Dataframe): anode capacity and voltage columns scaled,
                offset, and aligned along capacity

        Outputs:
        electrode_info_df (Dataframe): dataframe containing a variety of half-cell metrics 
            at various positions in the emulated full-cell profile.

            pe_voltage_FC4p2V: voltage of the positive electrode (catahode) corresponding
                to the full cell at 4.2V
            ...
            pe_voltage_FC2p7V: voltage of the positive electrode (catahode) corresponding
                to the full cell at 2.7V

            pe_soc_FC4p2V: state of charge of the positive electrode corresponding
                to the full cell at 4.2V
            ...
            pe_soc_FC2p7V: state of charge of the positive electrode corresponding
                to the full cell at 2.7V

            ne_voltage_FC4p2V: voltage of the negative electrode (anode) corresponding
                to the full cell at 4.2V
            ...
            ne_voltage_FC2p7V: voltage of the negative electrode (anode) corresponding
                to the full cell at 2.7V

            ne_soc_FC4p2V: state of charge of the anode electrode corresponding
                to the full cell at 4.2V
            ...
            ne_soc_FC2p7V: state of charge of the anode electrode corresponding
                to the full cell at 2.7V

            Q_fc: capacity of the full cecll within the full cell voltage limits
            Q_pe: capacity of the cathode
            Q_ne: capacity of the anode [Ahr]
            Q_li
        """
        pe_minus_ne_zeroed = pd.DataFrame(pe_out_zeroed['Voltage_aligned'] - ne_out_zeroed['Voltage_aligned'],
                                            columns=['Voltage_aligned'])
        pe_minus_ne_zeroed['Q_aligned'] = pe_out_zeroed['Q_aligned']

        electrode_info_df = pd.DataFrame(index=[0])

        electrode_info_df['pe_voltage_FC4p2V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.2))].Voltage_aligned
        electrode_info_df['pe_voltage_FC4p1V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.1))].Voltage_aligned
        electrode_info_df['pe_voltage_FC4p0V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.0))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p9V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.9))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p8V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.8))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p7V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.7))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p6V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.6))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p5V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.5))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p4V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.4))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p3V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.3))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p2V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.2))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p1V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.1))].Voltage_aligned
        electrode_info_df['pe_voltage_FC3p0V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.0))].Voltage_aligned
        electrode_info_df['pe_voltage_FC2p9V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.9))].Voltage_aligned
        electrode_info_df['pe_voltage_FC2p8V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.8))].Voltage_aligned
        electrode_info_df['pe_voltage_FC2p7V'] = pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.7))].Voltage_aligned

        electrode_info_df['pe_soc_FC4p2V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.2))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.2V
        electrode_info_df['pe_soc_FC4p1V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.1))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.1V
        electrode_info_df['pe_soc_FC4p0V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.0))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.0V
        electrode_info_df['pe_soc_FC3p9V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.9))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.9V
        electrode_info_df['pe_soc_FC3p8V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.8))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.8V
        electrode_info_df['pe_soc_FC3p7V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.7))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.7V
        electrode_info_df['pe_soc_FC3p6V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.6))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.6V
        electrode_info_df['pe_soc_FC3p5V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.5))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.5V    
        electrode_info_df['pe_soc_FC3p4V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.4))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.4V
        electrode_info_df['pe_soc_FC3p3V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.3))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.3V    
        electrode_info_df['pe_soc_FC3p2V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.2))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.2V    
        electrode_info_df['pe_soc_FC3p1V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.1))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.1V    
        electrode_info_df['pe_soc_FC3p0V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.0))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.0V
        electrode_info_df['pe_soc_FC2p9V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.9))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.9V    
        electrode_info_df['pe_soc_FC2p8V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.8))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.8V        
        electrode_info_df['pe_soc_FC2p7V'] = ((pe_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.7))].Q_aligned - 
                            np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.7V

        electrode_info_df['ne_voltage_FC4p2V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.2))].Voltage_aligned
        electrode_info_df['ne_voltage_FC4p1V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.1))].Voltage_aligned
        electrode_info_df['ne_voltage_FC4p0V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.0))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p9V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.9))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p8V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.8))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p7V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.7))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p6V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.6))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p5V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.5))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p4V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.4))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p3V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.3))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p2V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.2))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p1V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.1))].Voltage_aligned
        electrode_info_df['ne_voltage_FC3p0V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.0))].Voltage_aligned
        electrode_info_df['ne_voltage_FC2p9V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.9))].Voltage_aligned
        electrode_info_df['ne_voltage_FC2p8V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.8))].Voltage_aligned
        electrode_info_df['ne_voltage_FC2p7V'] = ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.7))].Voltage_aligned

        electrode_info_df['ne_soc_FC4p2V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.2))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.2V    
        electrode_info_df['ne_soc_FC4p1V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.1))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.1V
        electrode_info_df['ne_soc_FC4p0V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.0))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 4.0V
        electrode_info_df['ne_soc_FC3p9V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.9))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.9V
        electrode_info_df['ne_soc_FC3p8V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.8))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3Q_aligned.8V
        electrode_info_df['ne_soc_FC3p7V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.7))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.7V
        electrode_info_df['ne_soc_FC3p6V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.6))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.6V
        electrode_info_df['ne_soc_FC3p5V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.5))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.5V
        electrode_info_df['ne_soc_FC3p4V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.4))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.4V
        electrode_info_df['ne_soc_FC3p3V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.3))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.3V
        electrode_info_df['ne_soc_FC3p2V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.2))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.2V
        electrode_info_df['ne_soc_FC3p1V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.1))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.1V
        electrode_info_df['ne_soc_FC3p0V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 3.0))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 3.0V
        electrode_info_df['ne_soc_FC2p9V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.9))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.9V
        electrode_info_df['ne_soc_FC2p8V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.8))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.8V
        electrode_info_df['ne_soc_FC2p7V'] = ((ne_out_zeroed.loc[np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 2.7))].Q_aligned - 
                            np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])) / (
                                np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) -
                                np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]))
                        ) # 2.7V

        electrode_info_df['Q_fc'] = pe_minus_ne_zeroed.loc[
            np.argmin(np.abs(pe_minus_ne_zeroed.Voltage_aligned - 4.20))].Q_aligned

        electrode_info_df['Q_pe'] = np.max(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()]) - np.min(pe_out_zeroed['Q_aligned'].loc[~pe_out_zeroed['Voltage_aligned'].isna()])

        electrode_info_df['Q_ne'] = np.max(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()]) - np.min(ne_out_zeroed['Q_aligned'].loc[~ne_out_zeroed['Voltage_aligned'].isna()])

        electrode_info_df['Q_li'] = np.max(pe_minus_ne_zeroed['Q_aligned'].loc[~pe_minus_ne_zeroed.Voltage_aligned.isna()]) - np.min(pe_minus_ne_zeroed['Q_aligned'].loc[~pe_minus_ne_zeroed.Voltage_aligned.isna()])
        

        return electrode_info_df
    
    def _get_effective_electrode_rate_data(self,electrode_rate_dict=dict(),
                                          Q_electrode=4.84,Q_fc_nom=4.84,
                                          C_nom=-0.2,IR_coef=1.0):
        """
        Inputs:
        electrode_rate_dict (dict): dictonary of rate data for an electrode. Keys are float representing C rate; entries are dataframe of rate data with columns Voltage_aligned and SOC_aligned. Sign of cathode rate keys should be same as full cell (positive for charge, negative for discharge). Sign of anode rate keys should be opposite of the full cell.
        Q_electrode (float): capacity of the electrode in Ahr
        Q_fc_nom (float): nominal capacity of full cell which was used to define the nominal C rate
        C_nom (float): nominal current rate in full cell (negative for discharge)
        IR_coef (float): scaling coefficient of overpotential term, used in fitting
        
        Outputs:
        electrode_at_I_eff (DataFrame): electrode data modified for effective rate, based on the supplied dictionary of rates and an overpotential scaling term.
        
        """
        
        I_eff = C_nom / (Q_electrode/Q_fc_nom)
        C_nom_sign = np.sign(list(electrode_rate_dict.keys())[0])
        
        R_at_SOC = []
        for SOC in electrode_rate_dict[list(electrode_rate_dict.keys())[0]]['SOC_aligned']:

            V_at_SOC_for_rates = []
            for rate in electrode_rate_dict.keys():
                V_at_SOC_for_rates = np.append(V_at_SOC_for_rates,
                                               electrode_rate_dict[rate]['Voltage_aligned'].loc[
                                                   electrode_rate_dict[rate]['SOC_aligned'] == SOC
                                               ])
            R_at_SOC = np.append(R_at_SOC,
                np.polynomial.polynomial.Polynomial.fit(
                list(electrode_rate_dict.keys()),
                V_at_SOC_for_rates,deg=2).deriv(m=1)(I_eff)
                                    )
        electrode_at_I_eff = pd.DataFrame()
        electrode_at_I_eff['SOC_aligned'] = electrode_rate_dict[
            list(electrode_rate_dict.keys())[0]]['SOC_aligned']
        
        closest_rate_key = list(electrode_rate_dict.keys())[
            np.argmin(np.abs(
                np.array(list(electrode_rate_dict.keys())) - I_eff))]
        electrode_at_I_eff['Voltage_aligned'] = electrode_rate_dict[
            closest_rate_key]['Voltage_aligned'] + IR_coef*(I_eff - (C_nom*C_nom_sign)*R_at_SOC)
        
        return electrode_at_I_eff, R_at_SOC

def blend_electrodes(electrode_1, electrode_2_pos, electrode_2_neg, x_2): ## this function needs revisited
    """
    Blends two electrode materials from their SOC-V profiles to form a blended electrode.
    
    Inputs:
    electrode_1: Primary material in electrode, typically Gr. DataFrame supplied with SOC evenly spaced and voltage.
    electrode_2: Secondary material in electrode, typically Si. DataFrame supplied with SOC evenly spaced and
        voltage as an additional column.
    x_2: Fraction of electrode_2 material's capacity (not mass). Supplied as scalar value.
    
    Outputs:
    df_blended_soc_mod (Dataframe): blended electrode with SOC_aligned and Voltage_aligned columns
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

