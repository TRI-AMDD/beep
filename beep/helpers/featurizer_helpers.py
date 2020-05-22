#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""
Helper functions for generating features in beep.featurize module
All methods are currently lumped into this script.
"""


import pandas as pd
import numpy as np
import matplotlib as plt
from scipy import optimize, signal
from lmfit import models
from scipy.interpolate import interp1d


def isolate_dQdV_peaks(processed_cycler_run, diag_nr, charge_y_n, max_nr_peaks, rpt_type, half_peak_width=0.075):
    """
    Determine the number of cycles to reach a certain level of degradation

    Args:
        processed_cycler_run: processed_cycler_run (beep.structure.ProcessedCyclerRun): information about cycler run
        rpt_type: string indicating which rpt to pick
        charge_y_n: if 1 (default), takes charge dQdV, if 0, takes discharge dQdV
        diag_nr: if 1 (default), takes dQdV of 1st RPT past the initial diagnostic

    Returns:
        dataframe with Voltage and dQdV columns for charge or discharge curve in the rpt_type diagnostic cycle.
        The peaks will be isolated
    """

    rpt_type_data = processed_cycler_run.diagnostic_interpolated[(processed_cycler_run.diagnostic_interpolated.cycle_type == rpt_type)]
    cycles = rpt_type_data.cycle_index.unique()

    ## Take charge or discharge from cycle 'diag_nr'
    data = pd.DataFrame({'dQdV': [], 'voltage': []})

    if charge_y_n == 1:
        data.dQdV = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr]) & (rpt_type_data.step_type == 0)].charge_dQdV.values
        data.voltage = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr]) & (rpt_type_data.step_type == 0)].voltage.values
    elif charge_y_n == 0:
        data.dQdV = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr]) & (rpt_type_data.step_type == 1)].discharge_dQdV.values
        data.voltage = rpt_type_data[
            (rpt_type_data.cycle_index == cycles[diag_nr]) & (rpt_type_data.step_type == 1)].voltage.values
        # Turn values to positive temporarily
        data.dQdV = -data.dQdV
    else:
        raise NotImplementedError('Charge_y_n must be either 0 or 1')

    # Remove NaN from x and y
    data = data.dropna()

    # Reset x and y to values without NaNs
    x = data.voltage
    y = data.dQdV

    # Remove strong outliers
    upper_limit = y.sort_values().tail(round(0.01 * len(y))).mean() + y.sort_values().mean()
    data = data[(y < upper_limit)]

    # Reset x and y to values without outliers
    x = data.voltage
    y = data.dQdV

    # Filter out the x values of the peaks only
    no_filter_data = data

    # Find peaks
    peak_indices = signal.find_peaks_cwt(y, (10,))[-max_nr_peaks:]

    peak_voltages = {}
    peak_dQdVs = {}

    for count, i in enumerate(peak_indices):
        temp_filter_data = no_filter_data[((x > x.iloc[i] - half_peak_width) & (x < x.iloc[i] + half_peak_width))]
        peak_voltages[count] = x.iloc[i]
        peak_dQdVs[count] = y.iloc[i]

        if count == 0:
            filter_data = temp_filter_data
        else:
            filter_data = filter_data.append(temp_filter_data)

    return filter_data, no_filter_data, peak_voltages, peak_dQdVs


def generate_model(spec):
    """
    Method that generates a model to fit the hppc data to for peak extraction, using spec dictionary
    :param spec (dict): dictionary containing X, y model types.
    :return: composite model objects of lmfit Model class and a parameter object as defined in lmfit.
    """
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'

        #models is an lmfit object
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel',
                                  'VoigtModel']:  # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1 * y_max)
            model.set_param_hint('amplitude', min=1e-6)

            default_params = {
                prefix + 'center': x_min + x_range * np.random.randn(),
                prefix + 'height': y_max * np.random.randn(),
                prefix + 'sigma': x_range * np.random.randn()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params


def update_spec_from_peaks(spec, model_indices, peak_voltages, peak_dQdVs, peak_widths=(10,), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)

    for i, j, model_index in zip(peak_voltages, peak_dQdVs, model_indices):
        model = spec['model'][model_index]

        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'height': peak_dQdVs[j],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': peak_voltages[i]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
    return


def generate_dQdV_peak_fits(processed_cycler_run, rpt_type, diag_nr, charge_y_n, plotting_y_n=0, max_nr_peaks=4):
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

    data, no_filter_data, peak_voltages, peak_dQdVs = isolate_dQdV_peaks(processed_cycler_run, rpt_type=rpt_type, \
                                                                         charge_y_n=charge_y_n, diag_nr=diag_nr,
                                                                         max_nr_peaks=max_nr_peaks,
                                                                         half_peak_width=0.07)

    no_filter_x = no_filter_data.voltage
    no_filter_y = no_filter_data.dQdV

    ####### Setting spec for gaussian model generation

    x = data.voltage
    y = data.dQdV

    # Set construct spec using number of peaks
    model_types = []
    for i in np.arange(max_nr_peaks):
        model_types.append({'type': 'GaussianModel', 'help': {'sigma': {'max': 0.1}}})

    spec = {
        'x': x,
        'y': y,
        'model': model_types
    }

    # Update spec using the found peaks
    update_spec_from_peaks(spec, np.arange(max_nr_peaks), peak_voltages, peak_dQdVs)
    if plotting_y_n:
        fig, ax = plt.subplots()
        ax.scatter(spec['x'], spec['y'], s=4)
        for i in peak_voltages:
            ax.axvline(x=peak_voltages[i], c='black', linestyle='dotted')
            ax.scatter(peak_voltages[i], peak_dQdVs[i], s=30, c='red')

    #### Generate fitting model

    model, params = generate_model(spec)
    output = model.fit(spec['y'], params, x=spec['x'])
    if plotting_y_n:
        #         #Plot residuals
        #         fig, gridspec = output.plot(data_kws={'markersize': 1})

        ### Plot components

        ax.scatter(no_filter_x, no_filter_y, s=4)
        ax.set_xlabel('Voltage')

        if charge_y_n:
            ax.set_title(f'dQdV for charge diag cycle {diag_nr}')
            ax.set_ylabel('dQdV')
        else:
            ax.set_title(f'dQdV for discharge diag cycle {diag_nr}')
            ax.set_ylabel('- dQdV')

        components = output.eval_components()
        for i, model in enumerate(spec['model']):
            ax.plot(spec['x'], components[f'm{i}_'])

    # Construct dictionary of peak fits
    peak_fit_dict = {}
    for i, model in enumerate(spec['model']):
        best_values = output.best_values
        prefix = f'm{i}_'
        peak_fit_dict[prefix + "Amp"] = [peak_dQdVs[i]]
        peak_fit_dict[prefix + "Mu"] = [best_values[prefix + "center"]]
        peak_fit_dict[prefix + "Sig"] = [best_values[prefix + "sigma"]]

    # Make dataframe out of dict
    peak_fit_df = pd.DataFrame(peak_fit_dict)

    return peak_fit_df



def interp(df):
    '''
    this function takes in a data frame that we are interested in, and
    returns an interpolation function based on the discharge volatge and capacity
    '''
    V = df.voltage.values
    Q = df.discharge_capacity.values
    f = interp1d(Q, V, kind='cubic', fill_value="extrapolate")
    return f


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
        voltage1.append(df_i['voltage'].iloc[-10].mean())  # take the mean of the last 10 points of the voltage value
    return voltage1


def get_hppc_ocv(processed_cycler_run, diag_num):
    '''
    This function takes in cycling data for one cell and returns the variance of OCVs at different SOCs
    diag_num cyce minus first hppc cycle(cycle 2)
    Argument:
            processed_cycler_run(process_cycler_run object)
            diag_num(int): diagnostic cycle number at which you want to get the feature, such as 37 or 142
    Returns:
            a float
            the variance of the diag_num minus cycle 2 for OCV
    '''
    data = processed_cycler_run.diagnostic_interpolated
    cycle_hppc = data.loc[data.cycle_type == 'hppc']
    cycle_hppc = cycle_hppc.loc[cycle_hppc.current.notna()]
    step = 11
    step_later = 43
    cycle_hppc_0 = cycle_hppc.loc[cycle_hppc.cycle_index == 2]
    #     in case that cycle 2 correspond to two cycles one is real cycle 2, one is at the end
    cycle_hppc_0 = cycle_hppc_0.loc[cycle_hppc_0.test_time < 250000]
    voltage_1 = get_hppc_ocv_helper(cycle_hppc_0, step)
    chosen = cycle_hppc.loc[cycle_hppc.cycle_index == diag_num]
    voltage_2 = get_hppc_ocv_helper(chosen, step_later)
    dv = list_minus(voltage_1, voltage_2)
    return np.var(dv)


def get_hppc_r(processed_cycler_run, diag_num):
    '''
    This function takes in cycling data for one cell and returns the resistance at different SOCs with resistance at the
    first hppc cycle(cycle 2) deducted
    Argument:
            processed_cycler_run(process_cycler_run object)
            diag_num(int): diagnostic cycle number at which you want to get the feature, such as 37 or 142
    Returns:
            two floats
            the variance of the diag_num - cycle 2 for HPPC resistance for both charge and discharge
    '''
    data = processed_cycler_run.diagnostic_interpolated
    cycle_hppc = data.loc[data.cycle_type == 'hppc']
    cycle_hppc = cycle_hppc.loc[cycle_hppc.current.notna()]
    cycles = cycle_hppc.cycle_index.unique()
    if diag_num not in cycles:
        return None
    steps = [11, 12, 14]
    states = ['R', 'D', 'C']
    results_0 = {}
    results = {}
    resistance = {}
    dr_d = {}
    cycle_hppc_0 = cycle_hppc.loc[cycle_hppc.cycle_index == 2]
    #     in case that cycle 2 correspond to two cycles one is real cycle 2, one is at the end
    cycle_hppc_0 = cycle_hppc_0.loc[cycle_hppc_0.test_time < 250000]
    for i in range(len(steps)):
        chosen = cycle_hppc_0[cycle_hppc_0.step_index == steps[i]]
        state = states[i]
        result = get_V_I(chosen)
        results_0[state] = result
    results[2] = results_0
    steps_later = [43, 44, 46]
    #     step 43 is rest, 44 is discharge and 46 is charge, use the get ocv function to get the voltage values
    #     and calculate the over potential and thus the resistance change
    for i in range(1, len(cycles)):
        chosen = cycle_hppc[cycle_hppc.cycle_index == cycles[i]]
        results_s = {}
        for j in range(len(steps_later)):
            chosen_s = chosen[chosen.step_index == steps_later[j]]
            state = states[j]
            results_s[state] = get_V_I(chosen_s)
        results[cycles[i]] = results_s
    # calculate the resistance and compare the cycle evolution
    keys = list(results.keys())
    resistance['D'] = {}
    resistance['C'] = {}
    for i in range(len(keys)):
        d_v = results[keys[i]]['D']['voltage']  # discharge voltage for a cycle
        c_v = results[keys[i]]['C']['voltage']  # charge voltage for a cycle
        r_v = results[keys[i]]['R']['voltage']  # rest voltage for a cycle
        r_v_d = r_v[0:min(len(r_v), len(d_v))]  # in case the size is different
        d_v = d_v[0:min(len(r_v), len(d_v))]
        c_v = c_v[0:min(len(r_v), len(c_v))]
        r_v_c = r_v[0:min(len(r_v), len(c_v))]
        d_n = list(np.array(d_v) - np.array(r_v_d))  # discharge overpotential
        c_n = list(np.array(c_v) - np.array(r_v_c))  # charge overpotential
        resistance['D'][keys[i]] = np.true_divide(d_n, results[keys[i]]['D']['current'])
        resistance['C'][keys[i]] = np.true_divide(c_n, results[keys[i]]['C']['current'])
    resistance_d = resistance['D']
    resistance_c = resistance['C']
    dr_c = {}
    SOC = list(range(10, 100, 10))
    for i in range(1, len(keys)):
        resistance_d_i = resistance_d[keys[i]]
        resistance_d_0 = resistance_d[keys[0]]
        resistance_d_0 = resistance_d_0[0:min(len(resistance_d_i), len(resistance_d_0))]
        dr_d[keys[i]] = list(resistance_d_i - resistance_d_0)
    for i in range(1, len(keys)):
        resistance_c_i = resistance_c[keys[i]]
        resistance_c_0 = resistance_c[keys[0]]
        resistance_c_0 = resistance_c_0[0:min(len(resistance_c_i), len(resistance_c_0))]
        dr_c[keys[i]] = list(resistance_c_i - resistance_c_0)
    f2_d = np.var(dr_d[diag_num])
    f2_c = np.var(dr_c[diag_num])
    return f2_d, f2_c


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
        voltage.append(df_i['voltage'].iloc[-1])  # the last point of the voltage value
        current.append(df_i['current'].mean())
    result['voltage'] = voltage
    result['current'] = current
    return result


def get_v_diff(diag_num, processed_cycler_run, soc_window):
    """
    This function helps us get the feature of the variance of the voltage difference
    across a specific capacity window
    Argument:
            diag_num(int): diagnostic cycle number at which you want to get the feature, such as 37 or 142
            processed_cycler_run(process_cycler_run object)
            soc_window(int): let the function know which step_counter_index you want to look at
    Returns:
            a float
    """
    data = processed_cycler_run.diagnostic_interpolated
    hppc_data = data.loc[data.cycle_type == 'hppc']
    # the discharge steps in the hppc cycles step number 47
    hppc_data_2 = hppc_data.loc[hppc_data.cycle_index == diag_num]
    hppc_data_1 = hppc_data.loc[hppc_data.cycle_index == 2]
    #     in case a final HPPC is appended in the end also with cycle number 2
    hppc_data_1 = hppc_data_1.loc[hppc_data_1.discharge_capacity < 8]
    hppc_data_2_d = hppc_data_2.loc[hppc_data_2.step_index == 47]
    hppc_data_1_d = hppc_data_1.loc[hppc_data_1.step_index == 15]
    step_counters_1 = hppc_data_1_d.step_index_counter.unique()
    step_counters_2 = hppc_data_2_d.step_index_counter.unique()
    if (len(step_counters_1) < 8) or (len(step_counters_2) < 8):
        print('error')
        return None
    else:
        chosen_1 = hppc_data_1_d.loc[hppc_data_1_d.step_index_counter == step_counters_1[soc_window]]
        chosen_2 = hppc_data_2_d.loc[hppc_data_2_d.step_index_counter == step_counters_2[soc_window]]
        chosen_1 = chosen_1.loc[chosen_1.discharge_capacity.notna()]
        chosen_2 = chosen_2.loc[chosen_2.discharge_capacity.notna()]
        if len(chosen_1) == 0 or len(chosen_2) == 0:
            print('error')
            return None
        f = interp(chosen_2)
        v_1 = chosen_1.voltage.tolist()
        v_2 = f(chosen_1.discharge_capacity).tolist()
        v_diff = list_minus(v_1, v_2)
        if abs(np.var(v_diff)) > 1:
            print('weird voltage')
            return None
        else:
            return v_diff


def get_energy_fraction(diag_num, processed_cycler_run, remaining, metric, cycle_type, file):
    """
    This function can help us to get the dataframe that contains the diagnostic cycle index and energy fraction
    which we can use to predict energy fraction later
    Argument:
            diag_num(int): diagnostic cycle number at which you want to get the feature, such as 37 or 142
            processed_cycler_run(process_cycler_run object)
            remaining(float): how much enegry left, such as 0.95
            metric: such as discharge_energy
            cycle_type: such as rpt_0.2C
            file(str): a string that has the filename
    Returns:
            a dataframe that contains two columns cycle index and discharge energy fraction
            Diagnostic cycles after the diagnostic cycle at which we generate the feature
    """
    summary_diag = processed_cycler_run.diagnostic_summary[processed_cycler_run.diagnostic_summary.cycle_type == cycle_type]
    initial = summary_diag[metric].max()
    threshold = remaining * initial
    if summary_diag[metric].min() > threshold:
        print('havent degraded to' + str(remaining) + metric + file)
        return None
    y = summary_diag[summary_diag[metric] < threshold].cycle_index.min()
    if diag_num > y:
        print('degraded before diagnostic cycle' + str(diag_num) + file)
        return None
    df = summary_diag[summary_diag['cycle_index'] > diag_num + 1]
    result = pd.DataFrame()
    result['cycle_index'] = df['cycle_index']
    result['discharge_energy_fraction'] = df['discharge_energy'] / initial
    return result
