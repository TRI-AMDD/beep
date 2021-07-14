"""
Functions for identifying "knee" behavior in capacity fade trajectory.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from beep.structure.cli import auto_load_processed
from beep.featurize import DiagnosticProperties

def load_capacity_fade_from_run(run,
                                cycle_type = 'rpt_0.2C',
                                project_list = ["PreDiag", "PredictionDiagnostics"],
                                featurized_path = "/home/ec2-user/SageMaker/efs-readonly/features/"
                                ):
    
    seq_num = int(run[-25:-22])
    dataset_from_features = BeepDataset.from_features('batches_1_2_3_throughput',
                                                  project_list,
                                                  [DiagnosticProperties],
                                                  feature_dir=featurized_path)
    
    cell_and_cycle_df = dataset_from_features.data.loc[
    (dataset_from_features.data['cycle_type'] == cycle_type) & 
    (dataset_from_features.data['metric'] == 'discharge_capacity') &
    (dataset_from_features.data['seq_num'] == seq_num)]
    
    efc = (cell_and_cycle_df['initial_regular_throughput']*cell_and_cycle_df['normalized_regular_throughput']).values
    capacity = cell_and_cycle_df['fractional_metric'].values

    batch_df_one_cell = pd.DataFrame(np.vstack((capacity,efc)).T,columns=['fractional_metric','equivalent_full_cycles'])
    batch_df_one_cell['seq_num'] = seq_num
    batch_df_one_cell.set_index('seq_num',inplace=True)
    
    return batch_df_one_cell

def bacon_watts_heaviside(X,*params):
    """
    Function call to generate a set of two linear lines with a transition from one to the next.
    
    X: set of values for the line system construction (a0, af, xt, yt, xf)
         a0: y-value of initial capacity point
         af: y-value of final capacity point
         xt: x-value of transition between the two fitted linear lines
         yt: y-value of transition between the two fitted linear lines
         xf: set to 1 to find slope over normalized (0-1) axes
    
    params
    capacity: vector of values indicating a capacity metric
    efc: vector of values indicating the throughput/lifetime metric
    """
    
    a0,af,xt,yt,xf = X
    capacity,efc = params
    xf_scale = np.max(efc)

    y = np.heaviside(xt-efc/xf_scale,0.5)*(a0 + (yt-a0)/(xt-0)*(efc/xf_scale-0)) +\
        np.heaviside(efc/xf_scale-xt,0.5)*(af+(yt-af)/(xt-xf)*(efc/xf_scale-xf))
    return y

def get_error_bacon_watts_heaviside(X,*params):
    """
    Function call to generate a set of two linear lines with a transition from one to the next.
    
    X: set of values for the line system construction (a0, af, xt, yt, xf)
         a0: y-value of initial capacity point
         af: y-value of final capacity point
         xt: x-value of transition between the two fitted linear lines
         yt: y-value of transition between the two fitted linear lines
         xf: set to 1 to find slope over normalized (0-1) axes
    
    params
    capacity: vector of values indicating a capacity metric
    efc: vector of values indicating the throughput/lifetime metric
    """
    capacity,efc = params
    y = bacon_watts_heaviside(X,capacity,efc)
    error = np.sum((y-capacity)**2)
    return error

def get_knee_value(seq_num,batch_df,threshold=0.75):
    """
    Function call for obtaining descriptors of knee behavior of capacity fade for a given cell.
    This function calls underlying methods within a solver to fit two linear lines to the capacity fade.
    The current best indicator of a knee is a positive value in "angle_diff", indicating a concave (accelerating) capacity fade.
    Negative values of "angle_diff" indicate a convex (attentuating) capacity fade.
    
    Inputs
    seq_num (int): sequence number of battery cell, unique identifier
    batch_df (DataFrame): df with index as seq_num, columns 'fractional_metric' and 'equivalent_full_cycles'
    
    Outputs
    knee_df: one-row dataframe containing values that describe the knee behavior
         a0: y-value of initial capacity point
         af: y-value of final capacity point
         xt: x-value of transition between the two fitted linear lines
         yt: y-value of transition between the two fitted linear lines
         xf: set to 1 to find slope over normalized (0-1) axes
   xf_scale: x-value of final capacity value, used to rescale to full scale
 slope_diff: difference in the value of slope between the first and second fitted lines
 angle_diff: difference in value of angle (radians) between the first and second fitted lines
      error: squared error during the fitting routine. Values less than 1e-3 are typically acceptable
    """
    degradation_path_seq_num = batch_df.loc[seq_num]
    degradation_path_seq_num = degradation_path_seq_num.loc[degradation_path_seq_num['fractional_metric'] > threshold]
    
    a0 = degradation_path_seq_num['fractional_metric'].iloc[0]
    af = np.min(degradation_path_seq_num['fractional_metric'])
    xf = 1
    xf_scale = np.max(degradation_path_seq_num['equivalent_full_cycles'])
    bacon_watts_param_bounds = ((degradation_path_seq_num['fractional_metric'].iloc[0],degradation_path_seq_num['fractional_metric'].iloc[0]), # a0
                                (np.min(degradation_path_seq_num['fractional_metric']),np.min(degradation_path_seq_num['fractional_metric'])), # af
                                (0,1), # xt
                                (af,1), # yt
                                (1,1)) # xf
    opt_result = differential_evolution(get_error_bacon_watts_heaviside,
                                                 bacon_watts_param_bounds,
                                                 args=(degradation_path_seq_num['fractional_metric'],degradation_path_seq_num['equivalent_full_cycles']),
                                                 strategy='best1bin', maxiter=10000,
                                                 popsize=500, tol=0.1e-8, mutation=0.5, recombination=0.7, seed=1,
                                                 callback=None, disp=None, polish=True, init='latinhypercube', atol=1e-7, updating='deferred', workers=-1, constraints=())
    xt = opt_result.x[2]
    yt = opt_result.x[3]

    m1 = (yt-a0)/(xt)
    m2 = (af-yt)/(xf-xt)
    slope_diff = m1-m2
    angle_diff = np.arctan(m1) - np.arctan(m2)
    
    error = opt_result.fun
    
    knee_df = pd.DataFrame([[a0, af, xt, yt, xf, xf_scale, slope_diff, angle_diff, error]],
                           columns=['a0', 'af', 'xt', 'yt', 'xf', 'xf_scale', 'slope_diff', 'angle_diff', 'error'])
    return knee_df
