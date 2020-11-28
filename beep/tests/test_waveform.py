# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests related to generating waveform files"""

import os
import unittest
import json
import numpy as np
import datetime
import shutil
from copy import deepcopy

import pandas as pd
from beep.protocol import (
    PROCEDURE_TEMPLATE_DIR,
    SCHEDULE_TEMPLATE_DIR,
    BIOLOGIC_TEMPLATE_DIR,
)
from beep.generate_protocol import generate_protocol_files_from_csv
from beep.utils.waveform import convert_velocity_to_power_waveform, RapidChargeWave
import matplotlib.pyplot as plt

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class ChargeWaveformTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_input_current_soc_as_x(self):
        charging_c_rates = [0.7, 1.8, 1.5, 1.0]
        above_80p_c_rate = 0.5
        soc_initial = 0.05
        soc_final = 0.8

        charging = RapidChargeWave(above_80p_c_rate, soc_initial, soc_final)

        current_multistep_soc_as_x, soc_vector = charging.get_input_current_multistep_soc_as_x(charging_c_rates)
        self.assertEqual(np.round(np.mean(current_multistep_soc_as_x), 6), np.round(1.2492750000000001, 6))
        self.assertEqual(np.round(np.median(current_multistep_soc_as_x), 6), np.round(1.25, 6))
        self.assertEqual(np.round(np.max(current_multistep_soc_as_x), 6), np.round(1.8, 6))
        self.assertEqual(np.round(np.min(current_multistep_soc_as_x), 6), np.round(0.35, 6))

        current_smooth_soc_as_x, soc_vector = charging.get_input_current_smooth_soc_as_x(charging_c_rates)
        self.assertEqual(np.round(np.mean(current_smooth_soc_as_x), 6), np.round(1.224568, 6))
        self.assertEqual(np.round(np.median(current_smooth_soc_as_x), 6), np.round(1.297537, 6))
        self.assertEqual(np.round(np.max(current_smooth_soc_as_x), 6), np.round(1.8, 6))
        self.assertEqual(np.round(np.min(current_smooth_soc_as_x), 6), np.round(0.5, 6))

        plt.figure()
        plt.plot(soc_vector, current_smooth_soc_as_x)
        plt.plot(soc_vector, current_multistep_soc_as_x, linestyle='--')
        plt.xlim(0, 1.05)
        plt.ylim(0, 3)
        plt.xlabel('SOC')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC', 'CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_soc.png"))

    def test_get_input_current_time_as_x(self):
        charging_c_rates = [0.7, 1.8, 1.5, 1.0]
        above_80p_c_rate = 0.5
        soc_initial = 0.05
        soc_final = 0.8

        charging = RapidChargeWave(above_80p_c_rate, soc_initial, soc_final)

        current_multistep_soc_as_x, soc_vector = charging.get_input_current_multistep_soc_as_x(charging_c_rates)
        time_multistep = charging.get_time_vector_from_c_vs_soc(soc_vector, current_multistep_soc_as_x)
        self.assertEqual(np.round(np.mean(time_multistep), 6), np.round(1337.678584, 6))
        self.assertEqual(np.round(np.median(time_multistep), 6), np.round(1345.388245, 6))
        self.assertEqual(np.round(np.max(time_multistep), 3), 2472.235)

        current_smooth_soc_as_x, soc_vector = charging.get_input_current_smooth_soc_as_x(charging_c_rates)
        time_smooth = charging.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_soc_as_x)
        self.assertEqual(np.round(np.mean(time_smooth), 6), np.round(1371.222438, 6))
        self.assertEqual(np.round(np.median(time_smooth), 6), np.round(1378.791605, 6))
        self.assertEqual(np.round(np.max(time_smooth), 3), 2600.013)

        plt.figure()
        plt.plot(time_smooth, current_smooth_soc_as_x)
        plt.plot(time_multistep, current_multistep_soc_as_x, linestyle='--')
        # plt.xlim(0, 1.05)
        plt.ylim(0, 3)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_time.png"))

    def test_get_input_current_matching_time(self):
        charging_c_rates = [0.7, 1.8, 1.5, 1.0]
        above_80p_c_rate = 0.5
        soc_initial = 0.05
        soc_final = 0.8

        charging = RapidChargeWave(above_80p_c_rate, soc_initial, soc_final)

        current_smooth, time_smooth, current_multistep, time_multistep = \
            charging.get_input_currents_both_to_final_soc(charging_c_rates)
        self.assertEqual(np.round(np.max(time_smooth), 3), np.round(np.max(time_multistep), 3))

        plt.figure()
        plt.plot(time_smooth, current_smooth)
        plt.plot(time_multistep, current_multistep, linestyle='--')

        plt.ylim(0, 3)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_matching.png"))

    def test_get_input_current_uniform_time(self):
        charging_c_rates = [0.7, 1.8, 1.5, 1.0]
        above_80p_c_rate = 0.5
        soc_initial = 0.05
        soc_final = 0.8

        charging = RapidChargeWave(above_80p_c_rate, soc_initial, soc_final)
        current_smooth, current_multi, time_uniform = charging.get_currents_with_uniform_time_basis(charging_c_rates)
        soc_smooth = np.sum(current_smooth) / 3600
        soc_multistep = np.sum(current_multi) / 3600
        self.assertEqual(np.round(soc_smooth + soc_initial, 2), soc_final)
        self.assertEqual(np.round(soc_multistep + soc_initial, 2), soc_final)
        self.assertTrue(np.all(np.diff(time_uniform) == 1))

        plt.figure()
        plt.plot(time_uniform, current_smooth)
        plt.plot(time_uniform, current_multi, linestyle='--')

        plt.ylim(0, 3)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_uniform.png"))

