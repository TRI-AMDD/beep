"""
Module for finding parameters for projects
"""
import pandas as pd
from glob import glob
import os
from beep import logger


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
