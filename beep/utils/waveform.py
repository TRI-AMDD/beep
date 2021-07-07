"""
Module for computing waveforms
"""
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve


def convert_velocity_to_power_waveform(waveform_file, velocity_units):
    """
    Helper function to perform model based conversion of velocity waveform into power waveform.

    For model description and parameters ref JECS, 161 (14) A2099-A2108 (2014)
    "Model-Based SEI Layer Growth and Capacity Fade Analysis for EV and PHEV Batteries and Drive Cycles"

    Args:
        waveform_file (str): file containing tab or comma delimited values of time and velocity
        velocity_units (str): units of velocity. Accept 'mph' or 'kmph' or 'mps'

    returns
    pd.DataFrame with two columns: time (sec) and power (W). Negative = Discharge
    """
    df = pd.read_csv(waveform_file, sep="\t", header=0)
    df.columns = ["t", "v"]

    if velocity_units == "mph":
        scale = 1600.0 / 3600.0
    elif velocity_units == "kmph":
        scale = 1000.0 / 3600.0
    elif velocity_units == "mps":
        scale = 1.0
    else:
        raise NotImplementedError

    df.v = df.v * scale

    # Define model constants
    m = 1500  # kg
    rolling_resistance_coef = 0.01  # rolling resistance coeff
    g = 9.8  # m/s^2
    theta = 0  # gradient in radians
    rho = 1.225  # kg/m^3
    drag_coef = 0.34  # Coeff of drag
    frontal_area = 1.75  # m^2
    v_wind = 0  # wind velocity in m/s

    # Power = Force * vel
    # Force = Rate of change of momentum + Rolling frictional force + Aerodynamic drag force

    # Method treats the time-series as is and does not interpolate on a uniform grid before computing gradient.
    power = (
        m * np.gradient(df.v, df.t)
        + rolling_resistance_coef * m * g * np.cos(theta * np.pi / 180)
        + 0.5 * rho * drag_coef * frontal_area * (df.v - v_wind) ** 2
    )

    power = -power * df.v  # positive power = charge

    return pd.DataFrame({"time": df.t, "power": power})


class RapidChargeWave:
    """
       Object to produce charging waveforms, with direct comparison between stepwise and
       smooth charging waveforms. Algorithms written by Patrick Asinger.

    """
    def __init__(self,
                 final_c_rate,
                 soc_initial,
                 soc_final,
                 max_c_rate,
                 min_c_rate
                 ):
        """
        Args:
            final_c_rate (float): charging rate for the final step
            soc_initial (float): estimated starting soc for the fast charging portion of the cycle
            soc_final (float): estimated soc to end the fast charging portion of the cycle
            max_c_rate (float): maximum charging rate for the charge
            min_c_rate (float): minimum charging rate for the charge

        """
        self.final_c_rate = final_c_rate
        self.soc_i = soc_initial
        self.soc_f = soc_final
        self.max_c_rate = max_c_rate
        self.min_c_rate = min_c_rate
        self.soc_points = 1000

    def get_currents_with_uniform_time_basis(self, charging_c_rates, mesh_points):
        """
        Function to re-interpolate all of the current values to a uniform time basis, with each time step 1 sec

        Args:
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            mesh_points (list): soc values for beginning and end of each of the charging windows

        returns
        np.array: array with the adjusted smooth current as a function of uniform time basis
        np.array: array with the multistep current as a function of the uniform time basis
        np.array: array with the corresponding uniformly spaced time values
        """
        assert np.all(np.diff(mesh_points)) > 0
        current_smooth, time_smooth, current_multistep, time_multistep = \
            self.get_input_currents_both_to_final_soc(charging_c_rates, mesh_points)
        end_time_smooth = int(np.round(np.max(time_smooth), 0))
        end_time_multistep = int(np.round(np.max(time_multistep), 0))
        assert end_time_smooth == end_time_multistep

        time_uniform = np.arange(0, end_time_smooth, 1)
        smooth_fill = (current_smooth[0], current_smooth[-1])

        interpolate_smooth = interpolate.interp1d(time_smooth,
                                                  current_smooth,
                                                  bounds_error=False,
                                                  fill_value=smooth_fill)
        current_smooth_uniform = interpolate_smooth(time_uniform)

        multistep_fill = (current_multistep[0], current_multistep[-1])
        interpolate_multistep = interpolate.interp1d(time_multistep,
                                                     current_multistep,
                                                     bounds_error=False,
                                                     fill_value=multistep_fill
                                                     )
        current_multistep_uniform = interpolate_multistep(time_uniform)

        return current_smooth_uniform, current_multistep_uniform, time_uniform

    def get_input_currents_both_to_final_soc(self, charging_c_rates, mesh_points):
        """
        Function to shift the charging rates for the smooth current to ensure that both the multistep and
        smooth charging functions reach the end in the same length of time.

        Args:
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            mesh_points (list): soc values for beginning and end of each of the charging windows

        returns
        np.array: array with the adjusted smooth current as a function of soc
        np.array: array with the corresponding time values
        np.array: array with the multistep current as a function of soc
        np.array: array with the corresponding time values
        """

        current_multistep, soc_vector = self.get_input_current_multistep_soc_as_x(charging_c_rates, mesh_points)

        time_multistep = self.get_time_vector_from_c_vs_soc(soc_vector, current_multistep)
        end_soc_time = time_multistep[soc_vector >= self.soc_f][0]

        offset_solved = fsolve(self.offset_value,
                               0.01,
                               maxfev=1000,
                               factor=0.1,
                               epsfcn=0.01,
                               args=(end_soc_time, charging_c_rates, mesh_points))

        charging_c_rates = [charging_c_rates[0]] + list(np.add(charging_c_rates[1:-1], offset_solved)) + \
                           [charging_c_rates[-1]]

        current_smooth_time_adjusted, soc_vector = self.get_input_current_smooth_soc_as_x(charging_c_rates, mesh_points)
        time_smooth_time_adjusted = self.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_time_adjusted)

        return current_smooth_time_adjusted, time_smooth_time_adjusted, current_multistep, time_multistep

    def offset_value(self, offset_test, *data):
        """
        Helper function shift the charging rates for the smooth current to ensure that both the multistep and
        smooth charging functions reach the end in the same length of time.

        Args:
            offset_test (float): Amount to shift all of the non-edge c rates in the smooth curve
            data (*args): ending time to hit, baseline charging rates to add to

        returns
        float: difference between the desired ending time and the actual ending time for the smooth curve
        """
        time_end_multistep, charging_c_rates, mesh_points = data
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)
        return time_end_multistep - self.shift_smooth_by_offset(offset_test,
                                                                charging_c_rates,
                                                                mesh_points)[0][soc_vector >= self.soc_f][0]

    def get_input_current_multistep_soc_as_x(self, charging_c_rates, mesh_points):
        """
        Helper function to generate a waveform with stepwise charging currents. Should return a vector
        of current values and vector of corresponding state of charge (soc) values.

        Args:
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            mesh_points (list): soc values for beginning and end of each of the charging windows

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)
        c_rate_i = np.zeros((len(mesh_points) - 1, len(soc_vector)))

        for i, elem in enumerate(mesh_points[0:-1]):
            soc1 = mesh_points[i]
            soc2 = mesh_points[i + 1]
            c_rate_i[i] = charging_c_rates[i] * np.heaviside(soc_vector - soc1, 0.5) * \
                np.heaviside(soc2 - soc_vector, 0.5)

            multistep_current_raw = sum(c_rate_i)

        c_rate_end = self.final_c_rate * np.heaviside(soc_vector - soc2, 0.5) * \
            np.heaviside(self.soc_f - soc_vector, 0.5)

        multistep_current = multistep_current_raw + c_rate_end

        return multistep_current, soc_vector

    def get_input_current_smooth_soc_as_x(self, charging_c_rates, mesh_points):
        """
        Helper function to generate a waveform with smoothly varying charging current. Should return a vector
        of current values and vector of corresponding state of charge (soc) values.

        Args:
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            mesh_points (list): soc values for beginning and end of each of the charging windows

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        mesh_points_mid = np.copy(mesh_points)

        for indx in range(len(mesh_points) - 1):
            mesh_points_mid[indx] = (mesh_points[indx] + mesh_points[indx + 1]) / 2

        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)

        mesh_points_mid = list([self.soc_i]) + list(mesh_points_mid) + list([self.soc_f * 1.01])
        mesh_points_mid = np.array(mesh_points_mid)

        charging_c_rate_start = charging_c_rates[0]
        rates = np.clip([charging_c_rate_start] + charging_c_rates + [charging_c_rates[-1]],
                        self.min_c_rate,
                        self.max_c_rate)

        interpolator = interpolate.PchipInterpolator(mesh_points_mid, rates, axis=0, extrapolate=0)

        charging_c_rate_soc1_end = interpolator.__call__(mesh_points[1])

        charging_c_rate_start = np.max(
            [charging_c_rates[0] - (charging_c_rate_soc1_end - charging_c_rates[0]), charging_c_rates[0]])
        charging_c_rate_start = np.min([self.max_c_rate, charging_c_rate_start])

        rates = np.clip([charging_c_rate_start] + charging_c_rates + [charging_c_rates[-1]],
                        self.min_c_rate,
                        self.max_c_rate)

        interpolator = interpolate.PchipInterpolator(mesh_points_mid, rates, axis=0, extrapolate=0)

        input_current = interpolator.__call__(soc_vector)
        input_current = np.nan_to_num(input_current, copy=False, nan=0)
        input_curent_smooth_soc_as_x = input_current

        return input_curent_smooth_soc_as_x, soc_vector

    def shift_smooth_by_offset(self, offset, charging_c_rates, mesh_points):
        """
        Helper function to shift the current values of the smooth charging curve to achieve the desired SOC sooner.

        Args:
            offset (float): Amount to shift all of the non-edge c rates in the smooth curve
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            mesh_points (list): soc values for beginning and end of each of the charging windows

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        shifted_c_rates = [charging_c_rates[0]] + list(np.add(charging_c_rates[1:-1], offset)) + [charging_c_rates[-1]]
        # shifted_c_rates = list(np.add(charging_c_rates, offset))

        current_smooth_shifted, soc_vector = self.get_input_current_smooth_soc_as_x(shifted_c_rates, mesh_points)

        time_smooth_shifted = self.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_shifted)

        return time_smooth_shifted, current_smooth_shifted

    @staticmethod
    def get_time_vector_from_c_vs_soc(soc_vector, current_soc_as_x):
        """
        Helper function to convert the soc vector to a time vector with the same datapoints

        Args:
            soc_vector (np.array): numpy array with
            current_soc_as_x (np.array): c-rates for all of the soc values

        returns
        np.array: array with the time corresponding to each of the current points
        """
        time_vector = np.cumsum((1 / current_soc_as_x) * np.gradient(soc_vector)) * 3600
        return time_vector
