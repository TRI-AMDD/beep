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
       smooth charging waveforms.

    """
    def __init__(self,
                 charging_c_rates,
                 above_80p_c_rate,
                 soc_initial,
                 soc_final,
                 ):
        """
        Args:
            charging_c_rates (list): c-rates for each of the charging steps. Each step is assumed to be an equal
                SOC portion of the charge, and the length of the list just needs to be at least 1
            above_80p_c_rate (float): charging rate for the final step
            soc_initial (float): estimated starting soc for the fast charging portion of the cycle
            soc_final (float): estimated soc to end the fast charging portion of the cycle

        """
        self.charging_c_rates = charging_c_rates
        self.final_c_rate = above_80p_c_rate
        self.soc_i = soc_initial
        self.soc_f = soc_final
        self.soc_points = 1000

    def get_input_currents_both_to_final_soc(self):
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)
        current_multistep, soc_vector = self.get_input_current_multistep_soc_as_x()

        time_multistep = self.get_time_vector_from_c_vs_soc(soc_vector, current_multistep)
        time_diff_80p = lambda offset: time_multistep[soc_vector > 0.8][0] - \
            self.shift_smooth_by_offset(offset)[0][soc_vector > 0.8][0]
        offset_solved = fsolve(time_diff_80p, 0.01, maxfev=1000, factor=0.1, epsfcn=0.01)

        self.charging_c_rates = list(np.add(self.charging_c_rates, offset_solved))

        current_smooth_time_adjusted, soc_vector = self.get_input_current_smooth_soc_as_x()
        time_smooth_time_adjusted = self.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_time_adjusted)

        return current_smooth_time_adjusted, time_smooth_time_adjusted, current_multistep, time_multistep

    def get_input_current_multistep_soc_as_x(self):
        """
        Helper function to generate a waveform with stepwise charging currents. Should return a vector
        of current values and vector of corresponding state of charge (soc) values.

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        mesh_points = np.linspace(self.soc_i, self.soc_f, len(self.charging_c_rates) + 1)
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)
        c_rate_i = np.zeros((len(mesh_points) - 1, len(soc_vector)))

        for i, elem in enumerate(mesh_points[0:-1]):
            soc1 = mesh_points[i]
            soc2 = mesh_points[i + 1]
            c_rate_i[i] = self.charging_c_rates[i] * np.heaviside(soc_vector - soc1, 0.5) * np.heaviside(soc2 - soc_vector, 0.5)

            multistep_current_raw = sum(c_rate_i)

        c_rate_end = self.final_c_rate * np.heaviside(soc_vector - soc2, 0.5) * np.heaviside(self.soc_f - soc_vector, 0.5)

        multistep_current = multistep_current_raw + c_rate_end

        return multistep_current, soc_vector

    def get_input_current_smooth_soc_as_x(self):
        """
        Helper function to generate a waveform with smoothly varying charging current. Should return a vector
        of current values and vector of corresponding state of charge (soc) values.

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)
        mesh_points = np.linspace(self.soc_i, self.soc_f, len(self.charging_c_rates) + 1)
        mesh_points_mid = mesh_points
        mesh_points_mid[0:-1] += 1 / (len(mesh_points_mid)) * 0.5
        # mesh_points_mid[-1] = mesh_points_mid[-1] * 1.01

        mesh_points_mid = list([self.soc_i]) + list(mesh_points_mid) + list([self.soc_f*1.01])
        # mesh_points_mid = np.array(mesh_points_mid)

        charging_c_rate_start = self.final_c_rate

        interpolator = interpolate.PchipInterpolator(
            mesh_points_mid,
            [charging_c_rate_start] + self.charging_c_rates + [self.final_c_rate] + [self.final_c_rate],
            axis=0,
            extrapolate=0)

        charging_c_rate_soc1_end = interpolator.__call__(mesh_points[1])

        charging_c_rate_start = np.max(
            [self.charging_c_rates[0] - (charging_c_rate_soc1_end - self.charging_c_rates[0]), self.final_c_rate])

        interpolator = interpolate.PchipInterpolator(
            mesh_points_mid,
            [charging_c_rate_start] + self.charging_c_rates + [self.final_c_rate] + [self.final_c_rate],
            axis=0,
            extrapolate=0)

        input_current = interpolator.__call__(soc_vector)
        input_current = np.nan_to_num(input_current, copy=False, nan=0)
        input_curent_smooth_soc_as_x = input_current

        return input_curent_smooth_soc_as_x, soc_vector

    def shift_smooth_by_offset(self, offset):
        """
        Helper function to shift the current values of the smooth charging curve to achieve the desired SOC sooner.

        Args:
            offset (float): amount to shift the current values

        returns
        np.array: array with the current as a function of soc
        np.array: array with the corresponding soc values
        """
        soc_vector = np.linspace(self.soc_i, self.soc_f, self.soc_points)

        shifted_c_rates = list(np.add(self.charging_c_rates, offset))

        self.charging_c_rates = shifted_c_rates

        current_smooth_shifted = self.get_input_current_smooth_soc_as_x()

        time_smooth_shifted = self.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_shifted)

        return time_smooth_shifted, current_smooth_shifted


    def get_time_vector_from_c_vs_soc(self, soc_vector, curent_soc_as_x):
        time_vector = np.cumsum((1 / curent_soc_as_x) * np.gradient(soc_vector)) * 3600
        return time_vector
