#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""
Helper functions for generating features in beep.featurize module
All methods are currently lumped into this script.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lmfit import models
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# TODO: document these params
def isolate_dQdV_peaks(
    processed_cycler_run,
    diag_nr,
    charge_y_n,
    rpt_type,
    cwt_range,
    max_nr_peaks,
    half_peak_width=0.075,
):
    """
    Determine the number of cycles to reach a certain level of degradation

    Args:
        processed_cycler_run: processed_cycler_run (beep.structure.ProcessedCyclerRun): information about cycler run
        diag_nr (int): if 1 (default), takes dQdV of 1st RPT past the initial diagnostic
        charge_y_n (int): if 1 (default), takes charge dQdV, if 0, takes discharge dQdV
        rpt_type (str): string indicating which rpt to pick
        cwt_range (list, np.ndarray): range for scaling parameter to use in Continuous Wave Transform
            method - used for peak finding

    Returns:
        dataframe with Voltage and dQdV columns for charge or discharge curve in the rpt_type diagnostic cycle.
            The peaks will be isolated
    """

    rpt_type_data = processed_cycler_run.diagnostic_interpolated[
        (processed_cycler_run.diagnostic_interpolated.cycle_type == rpt_type)
    ]
    cycles = rpt_type_data.cycle_index.unique()

    # Take charge or discharge from cycle 'diag_nr'
    data = pd.DataFrame({"dQdV": [], "voltage": []})

    if charge_y_n == 1:
        data.dQdV = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr])
            & (rpt_type_data.step_type == 0)
        ].charge_dQdV.values
        data.voltage = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr])
            & (rpt_type_data.step_type == 0)
        ].voltage.values
    elif charge_y_n == 0:
        data.dQdV = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr])
            & (rpt_type_data.step_type == 1)
        ].discharge_dQdV.values
        data.voltage = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr])
            & (rpt_type_data.step_type == 1)
        ].voltage.values
        # Turn values to positive temporarily
        data.dQdV = -data.dQdV
    else:
        raise NotImplementedError("Charge_y_n must be either 0 or 1")

    # Remove NaN from x and y
    data = data.dropna()

    # Reset x and y to values without NaNs
    x = data.voltage
    y = data.dQdV

    # Remove strong outliers
    upper_limit = (
        y.sort_values().tail(round(0.01 * len(y))).mean() + y.sort_values().mean()
    )
    data = data[(y < upper_limit)]

    # Reset x and y to values without outliers
    x = data.voltage
    y = data.dQdV

    # Filter out the x values of the peaks only
    no_filter_data = data

    # Find peaks
    peak_indices = signal.find_peaks_cwt(y, cwt_range)[-max_nr_peaks:]

    peak_voltages = []
    peak_dQdVs = []
    filter_data = pd.DataFrame()
    for count, i in enumerate(peak_indices):
        temp_filter_data = no_filter_data[
            ((x > x.iloc[i] - half_peak_width) & (x < x.iloc[i] + half_peak_width))
        ]
        peak_voltages.append(x.iloc[i])
        peak_dQdVs.append(y.iloc[i])

        filter_data = filter_data.append(temp_filter_data)

    return filter_data, no_filter_data, peak_voltages, peak_dQdVs


def generate_model(spec):
    """
    Method that generates a model to fit the RPT data to for peak extraction, using spec dictionary

    Args:
        spec (dict): dictionary containing X, y model types.

    Returns:
        (lmfit.Model): composite model objects of lmfit Model class and a parameter object as defined in lmfit.

    """
    composite_model = None
    params = None
    x = spec["x"]
    y = spec["y"]
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec["model"]):
        prefix = f"m{i}_"

        # models is an lmfit object
        model = getattr(models, basis_func["type"])(prefix=prefix)
        if basis_func["type"] in [
            "GaussianModel",
            "LorentzianModel",
            "VoigtModel",
        ]:  # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint("sigma", min=1e-6, max=x_range)
            model.set_param_hint("center", min=x_min, max=x_max)
            model.set_param_hint("height", min=1e-6, max=1.1 * y_max)
            model.set_param_hint("amplitude", min=1e-6)

            default_params = {
                prefix + "center": x_min + x_range * np.random.randn(),
                prefix + "height": y_max * np.random.randn(),
                prefix + "sigma": x_range * np.random.randn(),
            }
        else:
            raise NotImplementedError(f'model {basis_func["type"]} not implemented yet')
        if "help" in basis_func:  # allow override of settings in parameter
            for param, options in basis_func["help"].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(
            **default_params, **basis_func.get("params", {})
        )
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def update_spec_from_peaks(
    spec, model_indices, peak_voltages, peak_dQdVs, peak_widths=(10,), basis_func=None,
):
    x = spec["x"]
    x_range = np.max(x) - np.min(x)

    for peak_voltage, peak_dQdV, model_index in zip(peak_voltages, peak_dQdVs, model_indices):
        model = spec["model"][model_index]

        if model["type"] in ["GaussianModel", "LorentzianModel", "VoigtModel"]:
            params = {
                "height": peak_dQdV,
                "sigma": x_range / len(x) * np.min(peak_widths),
                "center": peak_voltage,
            }
            if "params" in model:
                model.update(params)
            else:
                model["params"] = params
        else:
            basis_func = basis_func or "unknown"
            raise NotImplementedError(f'model {basis_func} not implemented yet')
    return


def generate_dQdV_peak_fits(
    processed_cycler_run,
    rpt_type,
    diag_nr,
    charge_y_n,
    plotting_y_n=0,
    max_nr_peaks=4,
    cwt_range=np.arange(10, 30),
):
    """
    Generate fits characteristics from dQdV peaks

    Args:
        processed_cycler_run: processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_nr: if 1, takes dQdV of 1st RPT past the initial diagnostic, 0 (default) is initial dianostic
        charge_y_n: if 1 (default), takes charge dQdV, if 0, takes discharge dQdV

    Returns:
        dataframe with Amplitude, mu and sigma of fitted peaks

    """
    # Uses isolate_dQdV_peaks function to filter out peaks and returns x(Volt) and y(dQdV) values from peaks

    data, no_filter_data, peak_voltages, peak_dQdVs = isolate_dQdV_peaks(
        processed_cycler_run,
        rpt_type=rpt_type,
        charge_y_n=charge_y_n,
        diag_nr=diag_nr,
        cwt_range=cwt_range,
        max_nr_peaks=max_nr_peaks,
        half_peak_width=0.07,
    )

    no_filter_x = no_filter_data.voltage
    no_filter_y = no_filter_data.dQdV

    # Setting spec for gaussian model generation

    x = data.voltage
    y = data.dQdV

    # Set construct spec using number of peaks
    model_types = []
    # TODO: i isn't being used here
    for i in np.arange(len(peak_dQdVs)):
        model_types.append({"type": "GaussianModel", "help": {"sigma": {"max": 0.1}}})

    spec = {"x": x, "y": y, "model": model_types}

    # Update spec using the found peaks
    update_spec_from_peaks(spec, np.arange(max_nr_peaks), peak_voltages, peak_dQdVs)
    if plotting_y_n:
        # TODO: not sure this works
        fig, ax = plt.subplots()
        ax.scatter(spec["x"], spec["y"], s=4)
        for i, peak_voltage in enumerate(peak_voltages):
            ax.axvline(x=peak_voltage, c="black", linestyle="dotted")
            ax.scatter(peak_voltage, peak_dQdVs[i], s=30, c="red")

    # Generate fitting model

    model, params = generate_model(spec)
    output = model.fit(spec["y"], params, x=spec["x"])
    if plotting_y_n:
        # Plot residuals
        # fig, gridspec = output.plot(data_kws={'markersize': 1})

        # Plot components
        ax.scatter(no_filter_x, no_filter_y, s=4)
        ax.set_xlabel("Voltage")

        if charge_y_n:
            ax.set_title(f"dQdV for charge diag cycle {diag_nr}")
            ax.set_ylabel("dQdV")
        else:
            ax.set_title(f"dQdV for discharge diag cycle {diag_nr}")
            ax.set_ylabel("- dQdV")

        components = output.eval_components()
        for i, model in enumerate(spec["model"]):
            ax.plot(spec["x"], components[f"m{i}_"])

    # Construct dictionary of peak fits
    peak_fit_dict = {}
    for i, model in enumerate(spec["model"]):
        prefix = f"m{i}_"
        peak_fit_dict[prefix + "Amp" + "_" + rpt_type + "_" + str(charge_y_n)] = [
            peak_dQdVs[i]
        ]
        peak_fit_dict[prefix + "Mu" + "_" + rpt_type + "_" + str(charge_y_n)] = [
            peak_voltages[i]
        ]

    # Make dataframe out of dict
    peak_fit_df = pd.DataFrame(peak_fit_dict)

    # Incorporate troughs of dQdV curve
    color_list = ["g", "b", "r", "k", "c"]
    for peak_nr in np.arange(0, len(peak_dQdVs) - 1):
        between_outer_peak_data = no_filter_data[
            (no_filter_data.voltage > peak_voltages[peak_nr])
            & (no_filter_data.voltage < peak_voltages[peak_nr + 1])
        ]
        pct = 0.05
        lowest_dQdV_pct_between_peaks = (
            between_outer_peak_data.dQdV.sort_values(ascending=False)
        ).tail(round(len(between_outer_peak_data.dQdV) * pct))

        if plotting_y_n:
            ax.axhline(
                y=lowest_dQdV_pct_between_peaks.mean(),
                color=color_list[peak_nr],
                linestyle="-",
            )
        # Add belly feature to dataframe
        peak_fit_df[
            f"trough_height_{peak_nr}_{rpt_type}_{charge_y_n}"
        ] = lowest_dQdV_pct_between_peaks.mean()

    return peak_fit_df


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


def get_hppc_ocv(processed_cycler_run, diag_pos):
    """
    This function calculates the variance of ocv changes between hppc cycle specified by and the first one.

    Argument:
            processed_cycler_run (beep.structure.ProcessedCyclerRun)
            diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
            if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2.
    Returns:
            a dataframe with one entry('variance of ocv'):
                the variance of the diag_num minus cycle 2 for OCV.
    """

    hppc_ocv_features = pd.DataFrame()

    data = processed_cycler_run.diagnostic_interpolated
    cycle_hppc = data.loc[data.cycle_type == "hppc"]
    cycle_hppc = cycle_hppc.loc[cycle_hppc.current.notna()]
    cycles = cycle_hppc.cycle_index.unique()

    cycle_hppc_0 = cycle_hppc.loc[cycle_hppc.cycle_index == cycles[0]]
    #     in case that cycle 2 correspond to two cycles one is real cycle 2, one is at the end
    cycle_hppc_0 = cycle_hppc_0.loc[cycle_hppc_0.test_time < 250000]

    step = 11
    step_later = 43

    voltage_1 = get_hppc_ocv_helper(cycle_hppc_0, step)
    chosen = cycle_hppc.loc[cycle_hppc.cycle_index == cycles[diag_pos]]
    voltage_2 = get_hppc_ocv_helper(chosen, step_later)

    dv = list_minus(voltage_1, voltage_2)

    var_dv = np.var(dv)

    hppc_ocv_features["variance of ocv"] = [var_dv]

    return hppc_ocv_features


def get_chosen_df(processed_cycler_run, diag_pos):
    """
    This function narrows your data down to a dataframe that contains only the diagnostic cycle number you
    are interested in.

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2.

    Returns:
        a datarame that only has the diagnostic cycle you are interested in, and there is a column called
        'diagnostic_time[h]' starting from 0 for this dataframe.
    """

    data = processed_cycler_run.diagnostic_interpolated
    hppc_cycle = data.loc[data.cycle_type == "hppc"]
    hppc_cycle = hppc_cycle.loc[hppc_cycle.current.notna()]
    cycles = hppc_cycle.cycle_index.unique()
    diag_num = cycles[diag_pos]

    chosen = hppc_cycle.loc[hppc_cycle.cycle_index == diag_num]
    chosen = chosen.sort_values(by="test_time")
    chosen["diagnostic_time"] = (chosen.test_time - chosen.test_time.min()) / 3600

    return chosen


def res_calc(chosen, diag_pos, soc, step_ocv, step_cur, index):
    """
    This function calculates resistances at different socs and a specific pulse duration for a specified hppc cycle.

    Args:
        chosen(pd.DataFrame): a dataframe for a specific diagnostic cycle you are interested in.
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2
        soc (int): step index counter corresponding to the soc window of interest.
        step_ocv (int): 0 corresponds to the 1h-rest, and 2 corresponds to the 40s-rest.
        step_cur (int): 1 is for discharge, and 3 is for charge.
        index (float or str): this will input a time scale for resistance (unit is second), e.g. 0.01, 5 or
        'last' which is the entire pulse duration.

    Returns:
        (a number) resistance at a specific soc in hppc cycles
    """

    counters = []

    if diag_pos == 0:
        steps = [11, 12, 13, 14, 15]
    else:
        steps = [43, 44, 45, 46, 47]

    for step in steps:
        counters.append(
            chosen[chosen.step_index == step].step_index_counter.unique().tolist()
        )

    if index == "last":
        index = -1
    else:
        start = chosen[
            (chosen.step_index_counter == counters[step_cur][soc])
        ].diagnostic_time.min()
        stop = start + index / 3600
        index = len(
            chosen[
                (chosen.step_index_counter == counters[step_cur][soc])
                & (chosen.diagnostic_time > start)
                & (chosen.diagnostic_time < stop)
            ]
        )

    if len(counters[step_ocv]) < soc - 1:
        return None

    v_ocv = chosen[(chosen.step_index_counter == counters[step_ocv][soc])].voltage.iloc[
        -1
    ]
    #     i_ocv = chosen[(chosen.step_index_counter == counters[step_ocv][soc])].current.tail(5).mean()
    v_dis = chosen[(chosen.step_index_counter == counters[step_cur][soc])].voltage.iloc[
        index
    ]
    i_dis = chosen[(chosen.step_index_counter == counters[step_cur][soc])].current.iloc[
        index
    ]
    res = (v_dis - v_ocv) / i_dis

    return res


def get_resistance_soc_duration_hppc(processed_cycler_run, diag_pos):
    """
    This function calculates resistances at different socs and different pulse durations for a specified hppc cycle.

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2

    Returns:
        a dataframe contains 54 resistances calculated at diag_pos.
        6 columns are ohmic, charge transfer and polarization resistances each both for charge and discharge;
        9 columns from 0 to 8, correspond to state of charge from high to low.
    """

    resistances = pd.DataFrame()

    chosen = get_chosen_df(processed_cycler_run, diag_pos)

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 0, 1, "last")
        resistance.append(res)
    resistances["discharge_pulse_last"] = resistance

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 0, 1, 0.001)
        resistance.append(res)
    resistances["discharge_pulse_0.001s"] = resistance

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 0, 1, 2)
        resistance.append(res)
    resistances["discharge_pulse_2s"] = resistance

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 2, 3, "last")
        resistance.append(res)
    resistances["charge_pulse_last"] = resistance

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 2, 3, 2)
        resistance.append(res)
    resistances["charge_pulse_2s"] = resistance

    resistance = []
    for i in range(9):
        res = res_calc(chosen, diag_pos, i, 2, 3, 0.001)
        resistance.append(res)
    resistances["charge_pulse_0.001s"] = resistance

    result = pd.DataFrame()
    result["ohmic_r_d"] = resistances["discharge_pulse_0.001s"]
    result["ohmic_r_c"] = resistances["charge_pulse_0.001s"]
    result["ct_r_d"] = (
        resistances["discharge_pulse_2s"] - resistances["discharge_pulse_0.001s"]
    )
    result["ct_r_c"] = (
        resistances["charge_pulse_2s"] - resistances["charge_pulse_0.001s"]
    )
    result["polar_r_d"] = (
        resistances["discharge_pulse_last"] - resistances["discharge_pulse_2s"]
    )
    result["polar_r_c"] = (
        resistances["charge_pulse_last"] - resistances["charge_pulse_2s"]
    )

    return result


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


def get_v_diff(processed_cycler_run, diag_pos, soc_window, baseline_step_index=15, measured_step_index=47):
    """
    This method calculates the variance of voltage difference between a specified hppc cycle and the first
    one during a specified state of charge window.

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2
        soc_window (int): step index counter corresponding to the soc window of interest.
        baseline_step_index (int): step index of the initial hppc cycle (TODO automatically get value in the method)
        measured_step_index (int): step index of the comparison hppc cycle (TODO automatically get value in the method)

    Returns:
        a dataframe that contains the variance of the voltage differences
    """

    result = pd.DataFrame()

    data = processed_cycler_run.diagnostic_interpolated
    hppc_data = data.loc[data.cycle_type == "hppc"]
    cycles = hppc_data.cycle_index.unique()

    hppc_data_2 = hppc_data.loc[hppc_data.cycle_index == cycles[diag_pos]]
    hppc_data_1 = hppc_data.loc[hppc_data.cycle_index == cycles[0]]
    #     in case a final HPPC is appended in the end also with cycle number 2
    hppc_data_1 = hppc_data_1.loc[hppc_data_1.discharge_capacity < 8]

    #         the discharge steps in later hppc cycles has a step number of 47
    #         but that in the initial hppc cycle has a step number of 15
    hppc_data_2_d = hppc_data_2.loc[hppc_data_2.step_index == measured_step_index]
    hppc_data_1_d = hppc_data_1.loc[hppc_data_1.step_index == baseline_step_index]
    step_counters_1 = hppc_data_1_d.step_index_counter.unique()
    step_counters_2 = hppc_data_2_d.step_index_counter.unique()
    chosen_1 = hppc_data_1_d.loc[
        hppc_data_1_d.step_index_counter == step_counters_1[soc_window]
    ]
    chosen_2 = hppc_data_2_d.loc[
        hppc_data_2_d.step_index_counter == step_counters_2[soc_window]
    ]
    chosen_1 = chosen_1.loc[chosen_1.discharge_capacity.notna()]
    chosen_2 = chosen_2.loc[chosen_2.discharge_capacity.notna()]

    V = chosen_1.voltage.values
    Q = chosen_1.discharge_capacity.values

    #  Round values so that non-strictly monotonic values are removed
    np.around(Q, decimals=6, out=Q)
    np.around(V, decimals=6, out=V)
    if not np.all(np.diff(Q) > 0):
        index_of_repeated = np.where(np.abs(np.diff(Q)) == 0)[0]
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
        result["var(v_diff)"] = [np.var(v_diff)]
        return result


# TODO: this is a linear fit, we should use something
#  from a library, e.g. numpy.polyfit
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


def get_diffusion_coeff(processed_cycler_run, diag_pos):
    """
    This method calculates diffusion coefficients at different soc for a specified hppc cycle.
    (NOTE: The slope is proportional to 1/sqrt(D), and D here is interdiffusivity)

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun)
        diag_pos (int): diagnostic cycle occurence for a specific <diagnostic_cycle_type>. e.g.
        if rpt_0.2C, occurs at cycle_index = [2, 37, 142, 244 ...], <diag_pos>=0 would correspond to cycle_index 2

    Returns:
        a dataframe with 8 entries, slope at different socs.
    """

    data = processed_cycler_run.diagnostic_interpolated
    hppc_cycle = data.loc[data.cycle_type == "hppc"]
    cycles = hppc_cycle.cycle_index.unique()
    diag_num = cycles[diag_pos]

    chosen = hppc_cycle.loc[hppc_cycle.cycle_index == diag_num]
    chosen = chosen.sort_values(by="test_time")

    counters = []

    if diag_pos == 0:
        steps = [11, 12, 13, 14, 15]
    else:
        steps = [43, 44, 45, 46, 47]

    for step in steps:
        counters.append(
            chosen[chosen.step_index == step].step_index_counter.unique().tolist()
        )

    result = pd.DataFrame()

    for i in range(1, min(len(counters[1]), 9)):
        discharge = chosen.loc[chosen.step_index_counter == counters[1][i]]
        rest = chosen.loc[chosen.step_index_counter == counters[2][i]]
        rest["diagnostic_time"] = rest.test_time - rest.test_time.min()
        t_d = discharge.test_time.max() - discharge.test_time.min()
        v = rest.voltage
        t = rest.diagnostic_time
        x = np.sqrt(t + t_d) - np.sqrt(t)
        y = v - v.min()
        a = d_curve_fitting(
            x[round(3 * len(x) / 4): len(x)], y[round(3 * len(x) / 4): len(x)]
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
        a dataframe contains 8 slope changes.

    """
    df_0 = get_diffusion_coeff(processed_cycler_run, 0)
    df = get_diffusion_coeff(processed_cycler_run, diag_pos)
    result = df_0.subtract(df)
    return result


# TODO: change decay argument to non-mutable type
def get_relaxation_times(voltage_data, time_data, decay_percentage=[0.5, 0.8, 0.99]):
    """
    This function takes in the voltage data and time data of a voltage relaxation curve
    and calculates out the time it takes to reach 50%, 80% and 99% of the OCV relaxation.

    Args:
        voltage_data(np.array): list of the voltage data in a voltage relaxation curve
        time_data(np.array)   : list of the time data corresponding to voltage data
        decay_percentage (list): list of thresholds to compute time constants for

    Returns:
        @time_array(np.array): list of time taken to reach percentage of total relaxation
                              where percentages are 50%, 80%, and 99% returned in that order.

    """

    # Scaling the voltage data to between 0-1
    final_voltage = voltage_data[-1]
    initial_voltage = voltage_data[0]
    scaled_voltage_data = (voltage_data - initial_voltage) / (
        final_voltage - initial_voltage
    )

    # shifting the time data to start at 0
    shifted_time_data = time_data - time_data[0]

    v_decay_inv = interp1d(scaled_voltage_data, shifted_time_data)

    # these are the decay percentages that will correspond to the time values extracted
    time_array = []

    for percent in decay_percentage:
        time_array.append(v_decay_inv(percent))

    return np.array(time_array)


def get_relaxation_features(processed_cycler_run, hppc_list=[0, 1], max_n_soc=10):
    """

    This function takes in the processed structure data and retrieves the fractional change in
    the time taken to reach 50%, 80% and 99% of the voltage decay between the first and
    the second HPPC cycles

    Args:
        processed_cycler_run (beep.structure.ProcessedCyclerRun): ProcessedCyclerRun object for the cell
            you want the diagnostic feature for.
        hppc_list (list): ordered list of length 2, specifying which two hppc cycles are used for computing
            fractional difference in relaxation time constants
        max_n_soc (int): max number of SOC windows expected for the hppc cycles

    Returns:
        @fracTimeArray(np.array): list of fractional difference in time taken to reach percentage of
            total relaxation between the first and second diagnostic cycle. It is organized such that
            the percentages 50%, 80%, and 99% correspond to a given column, and the rows are different
            SOCs of the HPPC starting at 0 with the highest SOC and going downwards.
    """

    total_time_array = []

    # chooses the first and the second diagnostic cycle
    for hppc_chosen in hppc_list:

        # Getting just the HPPC cycles
        hppc_diag_cycles = processed_cycler_run.diagnostic_interpolated[
            processed_cycler_run.diagnostic_interpolated.cycle_type == "hppc"
        ]

        # Getting unique and ordered cycle index list for HPPC cycles, and choosing the hppc cycle
        hppc_cycle_list = list(set(hppc_diag_cycles.cycle_index))
        hppc_cycle_list.sort()

        # Getting unique and ordered Regular Step List (Non-unique identifier)
        reg_step_list = hppc_diag_cycles[
            hppc_diag_cycles.cycle_index == hppc_cycle_list[hppc_chosen]
        ].step_index
        reg_step_list = list(set(reg_step_list))
        reg_step_list.sort()

        # The value of 1 for regular step corresponds to all of the relaxation curves in the hppc
        reg_step_relax = 1

        # Getting unique and ordered Step Counter List (unique identifier)
        step_count_list = hppc_diag_cycles[
            (hppc_diag_cycles.cycle_index == hppc_cycle_list[hppc_chosen])
            & (hppc_diag_cycles.step_index == reg_step_list[reg_step_relax])
        ].step_index_counter
        step_count_list = list(set(step_count_list))
        step_count_list.sort()

        all_time_array = np.nan * np.ones((max_n_soc-1, 3))

        # If there are more than 10 SOC windows truncate it to the last 9 elements.  Otherwise take
        # all the available ones from the second element onwards (first relaxation curve comes out
        # of a CV and must be ignored)
        if len(step_count_list) > max_n_soc:
            step_count_list = step_count_list[-(max_n_soc - 1):]
        else:
            step_count_list = step_count_list[1:]

        # gets all the times for a single SOC per loop
        for idx, step_idx_counter in enumerate(step_count_list):
            relax_curve_df = hppc_diag_cycles[
                (hppc_diag_cycles.cycle_index == hppc_cycle_list[hppc_chosen])
                & (hppc_diag_cycles.step_index_counter == step_idx_counter)
            ]

            time_array = get_relaxation_times(
                np.array(relax_curve_df.voltage), np.array(relax_curve_df.test_time)
            )
            all_time_array[idx][:] = time_array

        total_time_array.append(all_time_array)

    return total_time_array[1] / total_time_array[0]


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
    processed_cycler_run, metric="discharge_energy", diagnostic_cycle_type="rpt_0.2C"
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

    Returns:
        a dataframe with cycle_index, corresponding degradation relative to the first measured value, 'x',
        i.e. the discharge throughput passed by the first diagnostic
        and the value 'n' at each diagnostic, i.e. the equivalent scaling factor for lifetime using n*x
    """
    summary_diag_cycle_type = processed_cycler_run.diagnostic_summary[
        (processed_cycler_run.diagnostic_summary.cycle_type == diagnostic_cycle_type)
    ].reset_index()
    summary_diag_cycle_type = summary_diag_cycle_type[["cycle_index", metric]]

    # For the nx addition
    if 'energy' in metric:
        normalize_qty = 'discharge' + '_energy'
    else:
        normalize_qty = 'discharge' + '_capacity'

    normalize_qty_throughput = normalize_qty + '_throughput'
    regular_summary = processed_cycler_run.summary.copy()
    regular_summary[normalize_qty_throughput] = regular_summary[normalize_qty].cumsum()

    # Trim the cycle index in summary_diag_cycle_type to the max cycle in the regular cycles
    # (no partial cycles in the regular cycle summary) so that only full cycles are used for n
    summary_diag_cycle_type = summary_diag_cycle_type[summary_diag_cycle_type.cycle_index <=
                                                      regular_summary.cycle_index.max()]
    indices = summary_diag_cycle_type.cycle_index
    initial_throughput = regular_summary.loc[
        regular_summary.cycle_index == indices.iloc[0]
    ][normalize_qty_throughput].iloc[0]
    x = regular_summary.loc[
            regular_summary.cycle_index == indices.iloc[1]
        ][normalize_qty_throughput].iloc[0] - initial_throughput

    summary_diag_cycle_type['x'] = x
    summary_diag_cycle_type['n'] = (
            (1/x) * regular_summary[regular_summary.cycle_index.isin(indices)][normalize_qty_throughput]
    ).values

    # end of nx addition
    summary_diag_cycle_type[metric] = (
        summary_diag_cycle_type[metric]
        / processed_cycler_run.diagnostic_summary[metric].iloc[0]
    )

    summary_diag_cycle_type.columns = ["cycle_index", "fractional_metric", "x", "n"]
    return summary_diag_cycle_type
