"""
Module for computing waveforms
"""
import pandas as pd
import numpy as np


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
