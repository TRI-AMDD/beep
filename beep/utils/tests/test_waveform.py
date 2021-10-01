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
import numpy as np
from beep.utils.waveform import convert_velocity_to_power_waveform, RapidChargeWave
import matplotlib.pyplot as plt

from beep.tests.constants import TEST_FILE_DIR


class ChargeWaveformTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_input_current_soc_as_x(self):
        charging_c_rates = [0.5, 2.5, 2.0, 0.2]
        soc_points = [0.05, 0.25, 0.65, 0.8]
        final_c_rate = charging_c_rates[-1]
        soc_initial = soc_points[0]
        soc_final = soc_points[-1]
        max_c_rate = 3.0
        min_c_rate = 0.2

        charging = RapidChargeWave(final_c_rate, soc_initial, soc_final, max_c_rate, min_c_rate)

        current_multistep_soc_as_x, soc_vector = charging.get_input_current_multistep_soc_as_x(charging_c_rates,
                                                                                               soc_points)
        self.assertEqual(np.round(np.mean(current_multistep_soc_as_x), 6), np.round(1.8648, 6))
        self.assertEqual(np.round(np.median(current_multistep_soc_as_x), 6), np.round(2.5, 6))
        self.assertEqual(np.round(np.max(current_multistep_soc_as_x), 6), np.round(2.5, 6))
        self.assertEqual(np.round(np.min(current_multistep_soc_as_x), 6), np.round(0.25, 6))

        current_smooth_soc_as_x, soc_vector = charging.get_input_current_smooth_soc_as_x(charging_c_rates, soc_points)
        self.assertEqual(np.round(np.mean(current_smooth_soc_as_x), 6), np.round(1.631819, 6))
        self.assertEqual(np.round(np.median(current_smooth_soc_as_x), 6), np.round(1.991661, 6))
        self.assertEqual(np.round(np.max(current_smooth_soc_as_x), 6), np.round(2.5, 6))
        self.assertEqual(np.round(np.min(current_smooth_soc_as_x), 6), np.round(0.2, 6))

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
        charging_c_rates = [0.5, 2.5, 2.0, 0.2]
        soc_points = [0.05, 0.25, 0.65, 0.8]
        final_c_rate = charging_c_rates[-1]
        soc_initial = soc_points[0]
        soc_final = soc_points[-1]
        max_c_rate = 3.0
        min_c_rate = 0.2

        charging = RapidChargeWave(final_c_rate, soc_initial, soc_final, max_c_rate, min_c_rate)

        current_multistep_soc_as_x, soc_vector = charging.get_input_current_multistep_soc_as_x(charging_c_rates,
                                                                                               soc_points)
        time_multistep = charging.get_time_vector_from_c_vs_soc(soc_vector, current_multistep_soc_as_x)
        self.assertEqual(np.round(np.mean(time_multistep), 6), np.round(1552.953655, 6))
        self.assertEqual(np.round(np.median(time_multistep), 6), np.round(1701.081081, 6))
        self.assertEqual(np.round(np.max(time_multistep), 3), 2296.358)

        current_smooth_soc_as_x, soc_vector = charging.get_input_current_smooth_soc_as_x(charging_c_rates, soc_points)
        time_smooth = charging.get_time_vector_from_c_vs_soc(soc_vector, current_smooth_soc_as_x)
        self.assertEqual(np.round(np.mean(time_smooth), 6), np.round(1503.457129, 6))
        self.assertEqual(np.round(np.median(time_smooth), 6), np.round(1644.678275, 6))
        self.assertEqual(np.round(np.max(time_smooth), 3), 2539.486)

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
        charging_c_rates = [0.5, 2.5, 2.0, 0.2]
        soc_points = [0.05, 0.25, 0.65, 0.8]
        final_c_rate = charging_c_rates[-1]
        soc_initial = soc_points[0]
        soc_final = soc_points[-1]
        max_c_rate = 3.0
        min_c_rate = 0.2

        charging = RapidChargeWave(final_c_rate, soc_initial, soc_final, max_c_rate, min_c_rate)

        current_smooth, time_smooth, current_multistep, time_multistep = \
            charging.get_input_currents_both_to_final_soc(charging_c_rates, soc_points)

        plt.figure()
        plt.plot(time_smooth, current_smooth)
        plt.plot(time_multistep, current_multistep, linestyle='--')

        plt.ylim(0, 3.5)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_matching.png"))

        self.assertEqual(np.round(np.max(time_smooth), 3), np.round(np.max(time_multistep), 3))
        self.assertLessEqual(np.max(current_smooth), max_c_rate)

    def test_dropping_current_matching_time(self):
        charging_c_rates = [2.5, 0.5, 2.0, 0.2]
        soc_points = [0.05, 0.25, 0.65, 0.8]
        final_c_rate = charging_c_rates[-1]
        soc_initial = soc_points[0]
        soc_final = soc_points[-1]
        max_c_rate = 3.0
        min_c_rate = 0.2

        charging = RapidChargeWave(final_c_rate, soc_initial, soc_final, max_c_rate, min_c_rate)

        current_smooth, time_smooth, current_multistep, time_multistep = \
            charging.get_input_currents_both_to_final_soc(charging_c_rates, soc_points)

        plt.figure()
        plt.plot(time_smooth, current_smooth)
        plt.plot(time_multistep, current_multistep, linestyle='--')

        plt.ylim(0, 4)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_matching_dropping.png"))

        self.assertLessEqual(np.max(current_smooth), 3)
        self.assertEqual(np.round(np.max(time_smooth), 3), np.round(np.max(time_multistep), 3))

    def test_get_input_current_uniform_time(self):
        charging_c_rates = [0.5, 2.5, 2.0, 0.2]
        soc_points = [0.05, 0.25, 0.65, 0.8]
        final_c_rate = charging_c_rates[-1]
        soc_initial = soc_points[0]
        soc_final = soc_points[-1]
        max_c_rate = 3.0
        min_c_rate = 0.2

        charging = RapidChargeWave(final_c_rate, soc_initial, soc_final, max_c_rate, min_c_rate)
        current_smooth, current_multi, time_uniform = charging.get_currents_with_uniform_time_basis(charging_c_rates,
                                                                                                    soc_points)
        soc_smooth = np.sum(current_smooth) / 3600
        soc_multistep = np.sum(current_multi) / 3600

        plt.figure()
        plt.plot(time_uniform, current_smooth)
        plt.plot(time_uniform, current_multi, linestyle='--')

        plt.ylim(0, 4)
        plt.xlabel('Time [sec]')
        plt.ylabel('C rate [h$^{-1}$]')
        plt.legend(['Smooth', 'Multistep CC'])
        plt.savefig(os.path.join(TEST_FILE_DIR, "rapid_charge_uniform.png"))

        self.assertEqual(np.round(soc_smooth + soc_initial, 2), soc_final)
        self.assertEqual(np.round(soc_multistep + soc_initial, 2), soc_final)
        self.assertTrue(np.all(np.diff(time_uniform) == 1))
