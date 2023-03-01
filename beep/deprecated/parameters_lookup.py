from glob import glob
import os

import pandas as pd

from beep import logger
from beep import PROTOCOL_PARAMETERS_DIR

"""
Module for finding parameters for projects
"""


def get_project_sequence(path):
    """
    Returns project sequence for a given path

    Args:
        path (str): full project file path

    Returns:
        ([str]): list of project parts

    """
    root, file = os.path.split(path)
    file = file.split(".")[0]
    file_parts = file.split("_")
    return file_parts


def get_protocol_parameters(filepath, parameters_path):
    """
    Helper function to get the project parameters for a file given the filename

    Args:
        filepath (str): full path to the file
        parameters_path (str): location to look for parameter files

    Returns:
        pandas.DataFrame: single row DataFrame corresponding to the parameters for this file
        pandas.DataFrame: DataFrame with all of the parameter for the project

    """
    project_name_list = get_project_sequence(filepath)
    project_name = project_name_list[0]
    path = os.path.abspath(parameters_path)
    project_parameter_files = glob(os.path.join(path, project_name + "*"))
    assert len(project_parameter_files) <= 1, (
            "Found too many parameter files for: " + project_name
    )

    if len(project_parameter_files) == 1:
        df = pd.read_csv(project_parameter_files[0])
        parameter_row = df[df.seq_num == int(project_name_list[1])]
        if parameter_row.empty:
            logger.error("Unable to get project parameters for: %s", filepath)
            parameter_row = None
            df = None
    else:
        parameter_row = None
        df = None
    return parameter_row, df


def get_diagnostic_parameters(
        diagnostic_available, diagnostic_parameter_path, project_name
):
    """
    Interpolates data according to location and type of diagnostic
    cycles in the data

    Args:
        diagnostic_available (dict): dictionary with diagnostic_types as list,
            'length' of the diagnostic in cycles and location of the diagnostic
        diagnostic_parameter_path (str): full path to the location of the
            diagnostic parameter files
        project_name (str): name of the project to search with

    Returns:
        (list): containing upper and lower voltage limits for the
            diagnostic cycle

    """
    project_diag_files = glob(
        os.path.join(diagnostic_parameter_path, project_name + "*")
    )
    assert len(project_diag_files) <= 1, (
            "Found too many diagnostic parameter files for: " + project_name
    )

    # Find the voltage range for the diagnostic cycles
    if len(project_diag_files) == 1:
        df = pd.read_csv(project_diag_files[0])
        diag_row = df[
            df.diagnostic_parameter_set == diagnostic_available["parameter_set"]
            ]
        v_range = [
            diag_row["diagnostic_discharge_cutoff_voltage"].iloc[0],
            diag_row["diagnostic_charge_cutoff_voltage"].iloc[0],
        ]
    else:
        v_range = [2.7, 4.2]

    return v_range


def determine_structuring_parameters(
        self,
        v_range=None,
        resolution=1000,
        nominal_capacity=1.1,
        full_fast_charge=0.8,
        parameters_path=PROTOCOL_PARAMETERS_DIR,
):
    """
    Method for determining what values to use to convert raw run into processed run.


    Args:
        v_range ([float, float]): voltage range for interpolation
        resolution (int): resolution for interpolation
        nominal_capacity (float): nominal capacity for summary stats
        full_fast_charge (float): full fast charge for summary stats
        parameters_path (str): path to parameters files

    Returns:
        v_range ([float, float]): voltage range for interpolation
        resolution (int): resolution for interpolation
        nominal_capacity (float): nominal capacity for summary stats
        full_fast_charge (float): full fast charge for summary stats

    """
    if not parameters_path or not os.path.exists(parameters_path):
        raise FileNotFoundError(
            f"Parameters path {parameters_path} does not exist!")

    run_parameter, all_parameters = get_protocol_parameters(
        self.paths["raw"], parameters_path
    )

    # Logic for interpolation variables and diagnostic cycles
    diagnostic_available = False
    if run_parameter is not None:
        if {"capacity_nominal"}.issubset(run_parameter.columns.tolist()):
            nominal_capacity = run_parameter["capacity_nominal"].iloc[0]
        if {"discharge_cutoff_voltage", "charge_cutoff_voltage"}.issubset(
                run_parameter.columns):
            v_range = [
                all_parameters["discharge_cutoff_voltage"].min(),
                all_parameters["charge_cutoff_voltage"].max(),
            ]
        if {"diagnostic_type", "diagnostic_start_cycle",
            "diagnostic_interval"}.issubset(run_parameter.columns):
            if run_parameter["diagnostic_type"].iloc[0] == "HPPC+RPT":
                hppc_rpt = ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"]
                hppc_rpt_len = 5
                initial_diagnostic_at = [1, 1 + run_parameter[
                    "diagnostic_start_cycle"].iloc[0] + 1 * hppc_rpt_len]
                # Calculate the number of steps present for each cycle in the diagnostic as the pattern for
                # the diagnostic. If this pattern of steps shows up at the end of the file, that indicates
                # the presence of a final diagnostic
                diag_0_pattern = [len(self.raw_data[
                                          self.raw_data.cycle_index == x].step_index.unique())
                                  for x in range(initial_diagnostic_at[0],
                                                 initial_diagnostic_at[
                                                     0] + hppc_rpt_len)]
                diag_1_pattern = [len(self.raw_data[
                                          self.raw_data.cycle_index == x].step_index.unique())
                                  for x in range(initial_diagnostic_at[1],
                                                 initial_diagnostic_at[
                                                     1] + hppc_rpt_len)]
                # Find the steps present in the reset cycles for the first and second diagnostic
                diag_0_steps = set(self.raw_data[self.raw_data.cycle_index ==
                                                 initial_diagnostic_at[
                                                     0]].step_index.unique())
                diag_1_steps = set(self.raw_data[self.raw_data.cycle_index ==
                                                 initial_diagnostic_at[
                                                     1]].step_index.unique())
                diagnostic_starts_at = []
                for cycle in self.raw_data.cycle_index.unique():
                    steps_present = set(self.raw_data[
                                            self.raw_data.cycle_index == cycle].step_index.unique())
                    cycle_pattern = [len(self.raw_data[
                                             self.raw_data.cycle_index == x].step_index.unique())
                                     for x in
                                     range(cycle, cycle + hppc_rpt_len)]
                    if steps_present == diag_0_steps or steps_present == diag_1_steps:
                        diagnostic_starts_at.append(cycle)
                    # Detect final diagnostic if present in the data
                    elif cycle >= (
                            self.raw_data.cycle_index.max() - hppc_rpt_len - 1) and \
                            (
                                    cycle_pattern == diag_0_pattern or cycle_pattern == diag_1_pattern):
                        diagnostic_starts_at.append(cycle)

                diagnostic_available = {
                    "parameter_set":
                        run_parameter["diagnostic_parameter_set"].iloc[0],
                    "cycle_type": hppc_rpt,
                    "length": hppc_rpt_len,
                    "diagnostic_starts_at": diagnostic_starts_at,
                }

    return (
        v_range,
        resolution,
        nominal_capacity,
        full_fast_charge,
        diagnostic_available,
    )
