import json
import numpy as np
import pandas as pd
from monty.serialization import loadfn, dumpfn
from glob import glob
from datetime import datetime
from beep import structure, run_model, featurize
from beep.featurize import (
    RPTdQdVFeatures,
    HPPCResistanceVoltageFeatures,
    DiagnosticSummaryStats,
    HPPCRelaxationFeatures,
    DiagnosticProperties
)

# import yaml
from beep.dataset import BeepDataset
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import seaborn as sns
import os
# from throughput_dev_tri import get_threshold_targets
import pickle
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.optimize import differential_evolution
# from dtw import *
from IntracellAnalysis import *
# import ternary

def get_protocol_test_time_stats(protocol_params=pd.DataFrame({'charge_constant_current_1' : 0.2},index=[0]),
                                 project_list=["PreDiag", "PredictionDiagnostics"],
                                 processed_path="/home/ec2-user/SageMaker/efs-readonly/structure/",
                                 protocol_reference_file='PreDiag_parameters_20210302.csv',
                                 nominal_capacity=4.83,
                                 threshold=0.8,
                                 cycle_type='rpt_0.2C'):
    """
    This function retrieves test time information and computes summary statistics on that test time information.
    Test time information is extracted for cell's matching the protocol parameters provided in protocol_params
    and also have reached the fractional capacity defined by threshold for the diagnostic given by cycle_type.
    
    Inputs:
    protocol_params: DataFrame containing columns for the cycling parameters of interest to group upon. Only one row.
    project_list: list of project names to allow in search for repeated protocols.
    processed_path: path of procesed cell data.
    protocol_reference_file: csv file containing columns for cycling protocol information.
    
    Outputs:
    test_time_df: DataFrame containing  test time in seconds with seq_num as index.
    test_time_stats_df: DataFrame with one row containing summary statistics (mean, std) for the cells in test_time_df.
    """
    
    seq_num_list = get_seq_num_list_from_protocol_params(protocol_params=protocol_params,
                                     protocol_reference_file=protocol_reference_file,
                                     )
    test_time_df = get_test_time_df_from_seq_num_list(seq_num_list=seq_num_list,
                                                  nominal_capacity=nominal_capacity,
                                                  threshold=threshold,
                                                  cycle_type=cycle_type
                                                  )
    test_time_stats_df = pd.DataFrame({'mean':np.nanmean(test_time_df['test_time_to_eol']),
                                       'std':np.nanstd(test_time_df['test_time_to_eol'])},
                                       index=[0])
    return test_time_df, test_time_stats_df

def get_seq_num_list_from_protocol_params(protocol_params=pd.DataFrame(),
                                          protocol_reference_file='PreDiag_parameters_20210302.csv'):
    """
    This function collects a list of seq_num which have cycling protocols with parameters matching those provdided in protocol_params.
    
    Inputs:
    protocol_params: DataFrame containing columns for the cycling parameters of interest to group upon. Only one row.
    protocol_reference_file: csv file containing columns for cycling protocol information.
    
    Outputs:
    seq_num_list: list of seq_num that have cycling protocols with parameters matching those provdided in protocol_params.
    """    
    
    protocol_reference = pd.read_csv('PreDiag_parameters_20210302.csv').set_index('seq_num')
    seq_num_list = protocol_reference.groupby(by=protocol_params.columns.to_list()).get_group(tuple(protocol_params.values.ravel())).index.to_list()
    return seq_num_list

def get_test_time_df_from_seq_num_list(seq_num_list=[],
                                       nominal_capacity=4.83,
                                       threshold=0.8,
                                       cycle_type='rpt_0.2C',
                                       project_list=["PreDiag", "PredictionDiagnostics"],
                                       processed_path="/home/ec2-user/SageMaker/efs-readonly/structure/"):
    """
    This function extracts the test time endured until a provided threshold on fractional capacity is achieved,
    for a set of cells provided in seq_num_list.
    
    Inputs:
    seq_num_list: list of seq_num that have cycling protocols with parameters matching those provdided in protocol_params.
    nominal_capacity: nominal capacity of cell.
    threshold: fractional metric threshold for when the test time should be extracted.
    cycle_type: diagnostic type for which the threshold will be applied.
    project_list: list of project names to allow in search for repeated protocols.
    processed_path: path of procesed cell data.
    
    Outputs:
    test_time_df: DataFrame containing test time in seconds with seq_num as index.
    """    
    
    threshold_capacity = nominal_capacity*threshold
    
    test_time_list = []
    seq_num_list_finished = []
    for seq_num in seq_num_list:
        processed_run_list = [os.path.join(processed_path, f) 
                      for project_name in project_list 
                      for f in os.listdir(processed_path) 
                      if (os.path.isfile(os.path.join(processed_path, f)) and f.startswith(project_name)) and 'p2' not in f and project_name]
        
        assert [x for x in processed_run_list if int(x[-25:-22]) == seq_num], "unable to process cell "+str(seq_num)
        processed_run_for_seq_num = ([x for x in processed_run_list if int(x[-25:-22]) == seq_num])[0]
        cell_struct = loadfn(processed_run_for_seq_num)
        
        below_threshold_rows = cell_struct.diagnostic_summary.loc[
            (cell_struct.diagnostic_summary['discharge_capacity'] < threshold_capacity) & (cell_struct.diagnostic_summary.cycle_type == cycle_type)
                                                            ]
        if (len(below_threshold_rows) > 0):
            cycle_index_eol = cell_struct.diagnostic_summary.loc[
                (cell_struct.diagnostic_summary['discharge_capacity'] < threshold_capacity) & (cell_struct.diagnostic_summary.cycle_type == cycle_type)
                                                            ].iloc[0,:].cycle_index
        else:
            print('cell with seq_num '+str(seq_num)+' is in the designated protocol group, but has not reached the threshold.')
            continue
        test_time = np.nanmax(cell_struct.diagnostic_data.loc[
            cell_struct.diagnostic_data.cycle_index == cycle_index_eol]['test_time']
                              ) - np.nanmin(cell_struct.diagnostic_data['test_time'])
        test_time_list.append(test_time)
        seq_num_list_finished.append(seq_num)
    test_time_df = pd.DataFrame(test_time_list,index=seq_num_list_finished,columns=['test_time_to_eol'])
    
    return test_time_df

def get_unfinished_test_time_from_seq_num_list(seq_num_list=[],
                                         nominal_capacity=4.83,
                                         threshold=0.8,
                                         cycle_type='rpt_0.2C',
                                         project_list=["PreDiag", "PredictionDiagnostics"],
                                         processed_path="/home/ec2-user/SageMaker/efs-readonly/structure/"):
    """
    This function computes the total test time that has so far been endured for the cells provided in seq_num_list.
    
    Inputs:
    seq_num_list: list of seq_num that have cycling protocols with parameters matching those provdided in protocol_params.
    nominal_capacity: nominal capacity of cell.
    threshold: fractional metric threshold for when the test time should be extracted.
    cycle_type: diagnostic type for which the threshold will be applied.
    project_list: list of project names to allow in search for repeated protocols.
    processed_path: path of procesed cell data.
    
    Outputs:
    unfinished_test_time_df: DataFrame containing test time in seconds with seq_num as index.
    """    
    
    threshold_capacity = nominal_capacity*threshold
    
    test_time_list = []
    seq_num_list_finished = []
    for seq_num in seq_num_list:
        processed_run_list = [os.path.join(processed_path, f) 
                      for project_name in project_list 
                      for f in os.listdir(processed_path) 
                      if (os.path.isfile(os.path.join(processed_path, f)) and f.startswith(project_name)) and 'p2' not in f and project_name]
        
        assert [x for x in processed_run_list if int(x[-25:-22]) == seq_num], "unable to process cell "+str(seq_num)
        processed_run_for_seq_num = ([x for x in processed_run_list if int(x[-25:-22]) == seq_num])[0]
        cell_struct = loadfn(processed_run_for_seq_num)
        test_time_to_date = np.nanmax(cell_struct.diagnostic_data['test_time']) - np.nanmin(cell_struct.diagnostic_data['test_time'])
        test_time_list.append(test_time_to_date)
        
    unfinished_test_time_df = pd.DataFrame(test_time_list,index=seq_num_list,columns=['test_time_to_date'])
    return unfinished_test_time_df