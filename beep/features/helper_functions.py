#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""
Helper functions for generating features in beep.featurize module
All methods are currently lumped into this script.
"""
import os
import calendar

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis


from beep import PROTOCOL_PARAMETERS_DIR
from beep.utils import parameters_lookup


def list_minus(list1, list2):
    """
    this function takes in two lists and will return a list containing
    the values of list1 minus list2
    """
    result = []
    zip_object = zip(list1, list2)
    for list1_i, list2_i in zip_object:
        result.append(list1_i - list2_i)
    return result


def get_hppc_ocv_helper(cycle_hppc_0, step_num):
    """
    this helper function takes in a cycle and a step number
    and returns a list that stores the mean of the last five points of voltage in different
    step counter indexes (which is basically the soc window)
    """
    chosen1 = cycle_hppc_0[cycle_hppc_0.step_index == step_num]
    voltage1 = []
    step_index_counters = chosen1.step_index_counter.unique()[0:9]
    for i in range(len(step_index_counters)):
        df_i = chosen1.loc[chosen1.step_index_counter == step_index_counters[i]]
        voltage1.append(
            df_i["voltage"].iloc[-10].mean()
        )  # take the mean of the last 10 points of the voltage value
    return voltage1


def get_hppc_ocv(processed_cycler_run, diag_pos, parameters_path=PROTOCOL_PARAMETERS_DIR):
    """
    This function calculates the variance, min, mean, skew, kurtosis, sum and sum of squares 
    of ocv changes between hppc cycle specified by and the first one.

    Argument:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
            if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2.
    Returns:
            a dataframe with seven entries 
            ('var_ocv, min_ocv, mean_ocv, skew_ocv, kurtosis_ocv, sum_ocv, sum_square_ocv'):
    """

    hppc_ocv_features = pd.DataFrame()

    data = processed_cycler_run.diagnostic_data
    cycle_hppc = data.loc[data.cycle_type == "hppc"]
    cycle_hppc = cycle_hppc.loc[cycle_hppc.current.notna()]
    cycles = cycle_hppc.cycle_index.unique()

    cycle_hppc_0 = cycle_hppc.loc[cycle_hppc.cycle_index == cycles[0]]

    first_diagnostic_steps = get_step_index(processed_cycler_run,
                                            cycle_type="hppc",
                                            diag_pos=0,
                                            parameters_path=parameters_path
                                            )
    later_diagnostic_steps = get_step_index(processed_cycler_run,
                                            cycle_type="hppc",
                                            diag_pos=diag_pos,
                                            parameters_path=parameters_path
                                            )
    step_first = first_diagnostic_steps['hppc_long_rest']
    step_later = later_diagnostic_steps['hppc_long_rest']

    voltage_1 = get_hppc_ocv_helper(cycle_hppc_0, step_first)
    selected_diag_df = cycle_hppc.loc[cycle_hppc.cycle_index == cycles[diag_pos]]
    voltage_2 = get_hppc_ocv_helper(selected_diag_df, step_later)

    ocv = list_minus(voltage_1, voltage_2)

    hppc_ocv_features["var_ocv"] = [np.var(ocv)]
    hppc_ocv_features["min_ocv"] = [min(ocv)]
    hppc_ocv_features["mean_ocv"] = [np.mean(ocv)]
    hppc_ocv_features["skew_ocv"] = [skew(ocv)]
    hppc_ocv_features["kurtosis_ocv"] = [kurtosis(ocv, fisher=False, bias=False)]
    hppc_ocv_features["sum_ocv"] = [np.sum(np.absolute(ocv))]
    hppc_ocv_features["sum_square_ocv"] = [np.sum(np.square(ocv))]

    return hppc_ocv_features


def res_calc(chosen, soc, r_type):

    """
    This function calculates resistance based on different socs and differnet time scales in hppc cycles.
    Args:
        chosen(pd.DataFrame): a dataframe for a specific diagnostic cycle you are interested in.
        soc (int): step index counter corresponding to the soc window of interest - 0, 1, 2, 3, 4 ... 
        r_type (str): a string that indicates the time scale of the resistance you are calculating, e.g. 
        'r_c_0s', 'r_c_3s', 'r_c_end', 'r_d_0s', 'r_d_3s', 'r_d_end'
    Returns:
        charge/discharge resistance value (float) at this specific soc and time scale in hppc cycles
    """
    steps = chosen.step_index.unique()[1:6]
    counters = []
    for step in steps:
        counters.append(chosen[chosen.step_index == step].step_index_counter.unique().tolist())
    # for charge 
    if r_type[2] == 'c':
        # 40 s short rest 
        step_ocv = 2
        step_cur = 3
    # for discharge 
    if r_type[2] == 'd':
        # one hour long rest 
        step_ocv = 0
        step_cur = 1
    # if there is no dataframe for ocv or step cur, it means this step is skipped, so return None directly 
    try:
        chosen_step_ocv = chosen[(chosen.step_index_counter == counters[step_ocv][soc])]
        chosen_step_cur = chosen[chosen.step_index_counter == counters[step_cur][soc]]
    except IndexError:
        return None
    # since the data is voltage interpolated, so we want to sort the data based on time 
    chosen_step_ocv = chosen_step_ocv.sort_values(by='test_time')
    chosen_step_cur = chosen_step_cur.sort_values(by='test_time')
    # last data point of the rest is the ocv value 
    v_ocv = chosen_step_ocv.voltage.iloc[-1]
    # taking the average of the last 5 data points of the current 
    i_ocv = chosen_step_ocv.current.tail(5).mean()
    # now we look at the time scales   
    if r_type[4] == 'e':
        v_dis = chosen_step_cur.voltage.iloc[-1]
        i_dis = chosen_step_cur.current.iloc[-1]
        res = (v_dis - v_ocv)/(i_dis-i_ocv)
        return res
    else:
        if r_type[4] == '0':
            index = 0.001
        elif r_type[4] == '3':
            index = 3
        # test time is in the units of s 
        chosen_step_cur_index = chosen_step_cur[(chosen_step_cur.test_time - chosen_step_cur.test_time.min()) <= index]
        v_dis = chosen_step_cur_index.voltage.iloc[-1]
        i_dis = chosen_step_cur_index.current.iloc[-1]
        res = (v_dis - v_ocv)/(i_dis-i_ocv)
        return res


def get_resistance_soc_duration_hppc(processed_cycler_run, diag_pos):
    """
    This function calculates resistances based on different socs and differnet time scales for a targeted hppc cycle.
    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2
    Returns:
        a dataframe (single row)
        - and its 54 columns list all the possible resistance names 'r_c_0s_0', 'r_c_3s_0'...
            - r: resistance
            - c/d: state (charge or discharge)
            - timescale: 0s, 3s, or end of the cycle (resistance)
            - soc_index: an int indicating which soc window in HPPC e.g. 0, 1, 2,...
    """
    data = processed_cycler_run.diagnostic_data
    hppc_cycle = data.loc[data.cycle_type == 'hppc']
    hppc_cycle = hppc_cycle.loc[hppc_cycle.current.notna()]
    cycles = hppc_cycle.cycle_index.unique()
    # a list of strings to get charge/discharge resistances at different time scales
    names = ['r_c_0s', 'r_c_3s', 'r_c_end', 'r_d_0s', 'r_d_3s', 'r_d_end']
    output = pd.DataFrame()
    chosen = hppc_cycle[hppc_cycle.cycle_index == cycles[diag_pos]]
    # for each diagnostic cycle, we have a row conatins all the resistances 
    df_row = pd.DataFrame()
    for name in names:
        for j in range(9):
            # full name 
            f_name = name + '_' + str(j)
            df_row[f_name] = [res_calc(chosen, j, name)]
    output = output.append(df_row, ignore_index=True)
    return output


def get_dr_df(processed_cycler_run, diag_pos):
    """
    This function calculates resistance changes between a hppc cycle specified by and the first one under different
    pulse durations (1ms for ohmic resistance, 2s for charge transfer and the end of pulse for polarization resistance)
    and different state of charge.
    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2.
    Returns:
        a dataframe contains resistances changes normalized by the first diagnostic cycle value.
    """
    r_df_0 = get_resistance_soc_duration_hppc(processed_cycler_run, 0)
    r_df_i = get_resistance_soc_duration_hppc(processed_cycler_run, diag_pos)
    dr_df = (r_df_i - r_df_0) / r_df_0
    return dr_df


def get_V_I(df):
    """
    this helper functiion takes in a specific step in the first hppc cycle and gives you the voltage values as
    well as the current values after each step in the first cycle.
    """
    result = {}
    voltage = []
    current = []
    step_index_counters = df.step_index_counter.unique()[0:9]
    for i in range(len(step_index_counters)):
        df_i = df.loc[df.step_index_counter == step_index_counters[i]]
        voltage.append(df_i["voltage"].iloc[-1])  # the last point of the voltage value
        current.append(df_i["current"].mean())
    result["voltage"] = voltage
    result["current"] = current
    return result


def get_v_diff(processed_cycler_run, diag_pos, soc_window, parameters_path=PROTOCOL_PARAMETERS_DIR):
    """
    This method calculates the variance of voltage difference between a specified hppc cycle and the first
    one during a specified state of charge window.
    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2
        soc_window (int): step index counter corresponding to the soc window of interest.
    Returns:
        a dataframe that contains the variance of the voltage differences
    """

    result = pd.DataFrame()

    data = processed_cycler_run.diagnostic_data
    hppc_data = data.loc[data.cycle_type == "hppc"]
    cycles = hppc_data.cycle_index.unique()

    hppc_data_2 = hppc_data.loc[hppc_data.cycle_index == cycles[diag_pos]]
    hppc_data_1 = hppc_data.loc[hppc_data.cycle_index == cycles[0]]
    #     in case a final HPPC is appended in the end also with cycle number 2
    hppc_data_1 = hppc_data_1.loc[hppc_data_1.discharge_capacity < 8]

    step_ind_1 = get_step_index(processed_cycler_run, cycle_type="hppc", diag_pos=0, parameters_path=parameters_path)
    step_ind_2 = get_step_index(processed_cycler_run, cycle_type="hppc", diag_pos=1, parameters_path=parameters_path)

    hppc_data_2_d = hppc_data_2.loc[hppc_data_2.step_index == step_ind_2["hppc_discharge_to_next_soc"]]
    hppc_data_1_d = hppc_data_1.loc[hppc_data_1.step_index == step_ind_1["hppc_discharge_to_next_soc"]]
    step_counters_1 = hppc_data_1_d.step_index_counter.unique()
    step_counters_2 = hppc_data_2_d.step_index_counter.unique()

    if min(len(step_counters_1) - 1, len(step_counters_2) - 1) < soc_window:
        return None

    chosen_1 = hppc_data_1_d.loc[hppc_data_1_d.step_index_counter == step_counters_1[soc_window]]
    chosen_2 = hppc_data_2_d.loc[hppc_data_2_d.step_index_counter == step_counters_2[soc_window]]
    chosen_1 = chosen_1.loc[chosen_1.discharge_capacity.notna()]
    chosen_2 = chosen_2.loc[chosen_2.discharge_capacity.notna()]

    # Filter so that only comparing on the same interpolation
    chosen_2 = chosen_2[(chosen_1.discharge_capacity.min() < chosen_2.discharge_capacity) &
                        (chosen_1.discharge_capacity.max() > chosen_2.discharge_capacity)]

    V = chosen_1.voltage.values
    Q = chosen_1.discharge_capacity.values

    # Threshold between values so that non-strictly monotonic values are removed
    # 1e7 roughly corresponds to the resolution of a 24 bit ADC, higher precision
    # would be unphysical
    d_capacity_min = (np.max(Q) - np.min(Q)) / 1e7
    if not np.all(np.diff(Q) >= -d_capacity_min):
        # Assuming that Q needs to be strictly increasing with threshold
        index_of_repeated = np.where(np.diff(Q) >= -d_capacity_min)[0]
        Q = np.delete(Q, index_of_repeated, axis=0)
        V = np.delete(V, index_of_repeated, axis=0)

    f = interp1d(Q, V, kind="cubic", fill_value="extrapolate", assume_sorted=False)

    v_2 = chosen_2.voltage.tolist()
    v_1 = f(chosen_2.discharge_capacity).tolist()
    v_diff = list_minus(v_1, v_2)

    if abs(np.var(v_diff)) > 1:
        print("weird voltage")
        return None
    else:
        result["var_v_diff"] = [np.var(v_diff)] 
        result["min_v_diff"] = [min(v_diff)]
        result["mean_v_diff"] = [np.mean(v_diff)]
        result["skew_v_diff"] = [skew(v_diff)]
        result["kurtosis_v_diff"] = [kurtosis(v_diff, fisher=False, bias=False)]
        result["sum_v_diff"] = [np.sum(np.absolute(v_diff))]
        result["sum_square_v_diff"] = [np.sum(np.square(v_diff))]

        return result


# TODO: this is a linear fit, we should use something
#  from a library, e.g. numpy.polyfit
# The equation I am using is based on the linear part of the curve 
def d_curve_fitting(x, y):
    """
    This function fits given data x and y into a linear function.

    Argument:
           relevant data x and y.

    Returns:
            the slope of the curve.
    """

    def test(x, a, b):
        return a * x + b

    param, param_cov = curve_fit(test, x, y)

    a = param[0]

    return a


def get_diffusion_coeff(processed_cycler_run, diag_pos, parameters_path=PROTOCOL_PARAMETERS_DIR):
    """
    This method calculates diffusion coefficients at different soc for a specified hppc cycle.
    (NOTE: The slope is proportional to 1/sqrt(D), and D here is interdiffusivity)

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurrence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2

    Returns:
        a dataframe with 8 entries, slope at different socs.
    """

    data = processed_cycler_run.diagnostic_data
    hppc_cycle = data.loc[data.cycle_type == "hppc"]
    cycles = hppc_cycle.cycle_index.unique()
    diag_num = cycles[diag_pos]

    selected_diag_df = hppc_cycle.loc[hppc_cycle.cycle_index == diag_num]
    selected_diag_df = selected_diag_df.sort_values(by="test_time")

    counters = []

    step_ind = get_step_index(processed_cycler_run,
                              cycle_type="hppc",
                              diag_pos=diag_pos,
                              parameters_path=parameters_path
                              )

    steps = [step_ind["hppc_long_rest"],
             step_ind["hppc_discharge_pulse"],
             step_ind["hppc_short_rest"],
             step_ind["hppc_charge_pulse"],
             step_ind["hppc_discharge_to_next_soc"]]

    for step in steps:
        counters.append(
            selected_diag_df[selected_diag_df.step_index == step].step_index_counter.unique().tolist()
        )

    result = pd.DataFrame()

    for i in range(1, min(len(counters[1]), len(counters[2]), 9)):
        discharge = selected_diag_df.loc[selected_diag_df.step_index_counter == counters[1][i]]
        rest = selected_diag_df.loc[selected_diag_df.step_index_counter == counters[2][i]]
        rest.loc[:, "diagnostic_time"] = rest["test_time"] - rest["test_time"].min()
        t_d = discharge.test_time.max() - discharge.test_time.min()
        v = rest.voltage
        t = rest.diagnostic_time
        x = np.sqrt(t + t_d) - np.sqrt(t)
        y = v - v.min()
        a = d_curve_fitting(
            x[round(2 * len(x) / 3): len(x)], y[round(2 * len(x) / 3): len(x)]
        )
        result["D_" + str(i)] = [a]

    return result


def get_diffusion_features(processed_cycler_run, diag_pos):
    """
    This method calculates changes in diffusion coefficient between a specified hppc cycle and the first one at
    different state of charge.

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142...], <diag_pos>=0 would correspond to cycle_index 2.

    Returns:
        a dataframe contains 8 normalized slope changes.

    """
    df_0 = get_diffusion_coeff(processed_cycler_run, 0)
    df = get_diffusion_coeff(processed_cycler_run, diag_pos)
    result = (df - df_0)/(df_0)
    return result


def get_fractional_quantity_remaining(
    processed_cycler_run, metric="discharge_energy", diagnostic_cycle_type="rpt_0.2C"
):
    """
    Determine relative loss of <metric> in diagnostic_cycles of type <diagnostic_cycle_type> after 100 regular cycles


    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun): information about cycler run
        metric (str): column name to use for measuring degradation
        diagnostic_cycle_type (str): the diagnostic cycle to use for computing the amount of degradation

    Returns:
        a dataframe with cycle_index and corresponding degradation relative to the first measured value
    """
    summary_diag_cycle_type = processed_cycler_run.diagnostic_summary[
        (processed_cycler_run.diagnostic_summary.cycle_type == diagnostic_cycle_type)
        & (processed_cycler_run.diagnostic_summary.cycle_index > 100)
    ].reset_index()
    summary_diag_cycle_type = summary_diag_cycle_type[["cycle_index", metric]]
    summary_diag_cycle_type[metric] = (
        summary_diag_cycle_type[metric]
        / processed_cycler_run.diagnostic_summary[metric].iloc[0]
    )
    summary_diag_cycle_type.columns = ["cycle_index", "fractional_metric"]
    return summary_diag_cycle_type


def get_fractional_quantity_remaining_nx(
        processed_cycler_run,
        metric="discharge_energy",
        diagnostic_cycle_type="rpt_0.2C",
        parameters_path=PROTOCOL_PARAMETERS_DIR
):
    """
    Similar to get_fractional_quantity_remaining()
    Determine relative loss of <metric> in diagnostic_cycles of type <diagnostic_cycle_type>
    Also returns value of 'x', the discharge throughput passed by the first diagnostic
    and the value 'n' at each diagnostic

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun): information about cycler run
        metric (str): column name to use for measuring degradation
        diagnostic_cycle_type (str): the diagnostic cycle to use for computing the amount of degradation
        parameters_path (str): path to the parameters file for this run

    Returns:
        a dataframe with cycle_index, corresponding degradation relative to the first measured value, 'x',
        i.e. the discharge throughput passed by the first diagnostic
        and the value 'n' at each diagnostic, i.e. the equivalent scaling factor for lifetime using n*x
    """
    summary_diag_cycle_type = processed_cycler_run.diagnostic_summary[
        (processed_cycler_run.diagnostic_summary.cycle_type == diagnostic_cycle_type)
    ].reset_index()
    summary_diag_cycle_type = summary_diag_cycle_type[["cycle_index", "date_time_iso", metric]]

    # For the nx addition
    if 'energy' in metric:
        normalize_qty = 'discharge' + '_energy'
    else:
        normalize_qty = 'discharge' + '_capacity'

    normalize_qty_throughput = normalize_qty + '_throughput'
    regular_summary = processed_cycler_run.structured_summary.copy()
    regular_summary = regular_summary[regular_summary.cycle_index != 0]
    diagnostic_summary = processed_cycler_run.diagnostic_summary.copy()
    # TODO the line below should become superfluous
    regular_summary = regular_summary[
        ~regular_summary.cycle_index.isin(diagnostic_summary.cycle_index)]

    regular_summary.loc[:, normalize_qty_throughput] = regular_summary[normalize_qty].cumsum()
    diagnostic_summary.loc[:, normalize_qty_throughput] = diagnostic_summary[normalize_qty].cumsum()

    # Trim the cycle index in summary_diag_cycle_type to the max cycle in the regular cycles
    # (no partial cycles in the regular cycle summary) so that only full cycles are used for n
    summary_diag_cycle_type = summary_diag_cycle_type[summary_diag_cycle_type.cycle_index <=
                                                      regular_summary.cycle_index.max()]

    # Second gap in the regular cycles indicates the second set of diagnostics, bookending the
    # initial set of regular cycles.
    first_degradation_cycle = int(regular_summary.cycle_index[regular_summary.cycle_index.diff() > 1].iloc[0])
    last_initial_cycle = int(regular_summary.cycle_index[regular_summary.cycle_index <
                                                         first_degradation_cycle].iloc[-1])

    initial_regular_throughput = regular_summary[
            regular_summary.cycle_index == last_initial_cycle
        ][normalize_qty_throughput].values[0]

    summary_diag_cycle_type.loc[:, 'initial_regular_throughput'] = initial_regular_throughput

    summary_diag_cycle_type.loc[:, 'normalized_regular_throughput'] = summary_diag_cycle_type.apply(
        lambda x: (1 / initial_regular_throughput) *
        regular_summary[regular_summary.cycle_index < x['cycle_index']][normalize_qty_throughput].max(),
        axis=1
    )
    summary_diag_cycle_type['normalized_regular_throughput'].fillna(value=0, inplace=True)
    summary_diag_cycle_type.loc[:, 'normalized_diagnostic_throughput'] = summary_diag_cycle_type.apply(
        lambda x: (1 / initial_regular_throughput) *
        diagnostic_summary[diagnostic_summary.cycle_index < x['cycle_index']][normalize_qty_throughput].max(),
        axis=1
    )
    summary_diag_cycle_type['normalized_diagnostic_throughput'].fillna(value=0, inplace=True)
    # end of nx addition, calculate the fractional capacity compared to the first diagnostic cycle (reset)
    summary_diag_cycle_type.loc[:, metric] = (
        summary_diag_cycle_type[metric]
        / processed_cycler_run.diagnostic_summary[metric].iloc[0]
    )

    if "\\" in processed_cycler_run.metadata.protocol:
        protocol_name = processed_cycler_run.metadata.protocol.split("\\")[-1]
    else:
        _, protocol_name = os.path.split(processed_cycler_run.metadata.protocol)

    parameter_row, _ = parameters_lookup.get_protocol_parameters(protocol_name, parameters_path=parameters_path)

    summary_diag_cycle_type.loc[:, 'diagnostic_start_cycle'] = parameter_row['diagnostic_start_cycle'].values[0]
    summary_diag_cycle_type.loc[:, 'diagnostic_interval'] = parameter_row['diagnostic_interval'].values[0]

    # Calculate the epoch time stamp at each of the measurements for later comparison
    date_time_objs = pd.to_datetime(summary_diag_cycle_type["date_time_iso"])
    date_time_float = [
        calendar.timegm(t.timetuple()) if t is not pd.NaT else float("nan")
        for t in date_time_objs
    ]
    summary_diag_cycle_type.drop(columns=["date_time_iso"], inplace=True)
    summary_diag_cycle_type.loc[:, "epoch_time"] = date_time_float

    summary_diag_cycle_type.columns = ["cycle_index", "fractional_metric",
                                       "initial_regular_throughput", "normalized_regular_throughput",
                                       "normalized_diagnostic_throughput", "diagnostic_start_cycle",
                                       "diagnostic_interval", "epoch_time"]
    return summary_diag_cycle_type


def get_step_index(pcycler_run, cycle_type="hppc", diag_pos=0, parameters_path=PROTOCOL_PARAMETERS_DIR):
    """
        Gets the step indices of the diagnostic cycle which correspond to specific attributes

        Args:
            pcycler_run (beep.structure.ProcessedCyclerRun): processed data
            cycle_type (str): which diagnostic cycle type to evaluate
            diag_pos (int): which iteration of the diagnostic cycle to use (0 for first, 1 for second, -1 for last)

        Returns:
            dict: descriptive keys with step index as values
    """

    pulse_time = 120  # time in seconds used to decide if a current is a pulse or an soc change
    pulse_c_rate = 0.5  # c-rate to decide if a current is a discharge pulse
    rest_long_vs_short = 600  # time in seconds to decide if the rest is the long or short rest step
    soc_change_threshold = 0.05

    if "\\" in pcycler_run.metadata.protocol:
        protocol_name = pcycler_run.metadata.protocol.split("\\")[-1]
    else:
        _, protocol_name = os.path.split(pcycler_run.metadata.protocol)

    parameter_row, _ = parameters_lookup.get_protocol_parameters(protocol_name, parameters_path=parameters_path)

    step_indices_annotated = {}
    diag_data = pcycler_run.diagnostic_data
    cycles = diag_data.loc[diag_data.cycle_type == cycle_type]
    cycle = cycles[cycles.cycle_index == cycles.cycle_index.unique()[diag_pos]]

    if cycle_type == "hppc":
        for step in cycle.step_index.unique():
            cycle_step = cycle[(cycle.step_index == step)]
            median_crate = np.round(cycle_step.current.median() / parameter_row["capacity_nominal"].iloc[0], 2)
            mean_crate = np.round(cycle_step.current.mean() / parameter_row["capacity_nominal"].iloc[0], 2)
            remaining_time = cycle.test_time.max() - cycle_step.test_time.max()
            recurring = len(cycle_step.step_index_counter.unique()) > 1
            step_counter_duration = []
            for step_iter in cycle_step.step_index_counter.unique():
                cycle_step_iter = cycle_step[(cycle_step.step_index_counter == step_iter)]
                duration = cycle_step_iter.test_time.max() - cycle_step_iter.test_time.min()
                step_counter_duration.append(duration)
            median_duration = np.round(np.median(step_counter_duration), 0)

            if median_crate == 0.0:
                if median_duration > rest_long_vs_short:
                    step_indices_annotated["hppc_long_rest"] = step
                elif rest_long_vs_short >= median_duration > 0:
                    step_indices_annotated["hppc_short_rest"] = step
                else:
                    raise ValueError
            elif median_crate <= -pulse_c_rate and median_duration < pulse_time:
                step_indices_annotated["hppc_discharge_pulse"] = step
            elif median_crate >= pulse_c_rate and median_duration < pulse_time:
                step_indices_annotated["hppc_charge_pulse"] = step
            elif mean_crate != median_crate < 0 and remaining_time == 0.0 and not recurring:
                step_indices_annotated["hppc_final_discharge"] = step
            elif mean_crate == median_crate < 0 and remaining_time == 0.0 and not recurring:
                step_indices_annotated["hppc_final_discharge"] = step
            elif mean_crate == median_crate < 0 and abs(mean_crate * median_duration / 3600) > soc_change_threshold:
                step_indices_annotated["hppc_discharge_to_next_soc"] = step
            elif median_crate > 0 and median_duration > pulse_time:
                step_indices_annotated["hppc_charge_to_soc"] = step

    elif cycle_type == "rpt_0.2C" or cycle_type == "rpt_1C" or cycle_type == "rpt_2C" or cycle_type == "reset":
        for step in cycle.step_index.unique():
            cycle_step = cycle[(cycle.step_index == step)]
            median_crate = np.round(cycle_step.current.median() / parameter_row["capacity_nominal"].iloc[0], 2)
            if median_crate > 0:
                step_indices_annotated[cycle_type + "_charge"] = step
            elif median_crate < 0:
                step_indices_annotated[cycle_type + "_discharge"] = step
            else:
                raise ValueError
    else:
        raise NotImplementedError

    assert len(cycle.step_index.unique()) == len(step_indices_annotated.values())

    return step_indices_annotated


def check_diagnostic_validation(datapath):
    if not hasattr(datapath, "diagnostic_summary") & hasattr(
            datapath, "diagnostic_data"
    ):
        return False, "Datapath does not have diagnostic summary"
    if datapath.diagnostic_summary is None:
        return False, "Datapath does not have diagnostic summary"
    elif datapath.diagnostic_summary.empty:
        return False, "Datapath has empty diagnostic summary"
    else:
        return True, None
