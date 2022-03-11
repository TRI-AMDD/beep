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
"""Unit tests related to cycler run data structures"""

import os
import unittest
import numpy as np
import pandas as pd

from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir

from beep.structure.base import BEEPDatapath, step_is_waveform_dchg, step_is_waveform_chg
from beep.structure.arbin import ArbinDatapath
from beep.structure.maccor import MaccorDatapath
from beep.structure.neware import NewareDatapath
from beep.structure.indigo import IndigoDatapath
from beep.structure.biologic import BiologicDatapath, get_cycle_index
from beep.structure.battery_archive import BatteryArchiveDatapath
from beep.tests.constants import TEST_FILE_DIR


class TestArbinDatapath(unittest.TestCase):
    """
    Tests specific to Arbin cyclers.
    """
    def setUp(self) -> None:
        self.bad_file = os.path.join(
            TEST_FILE_DIR, "2017-05-09_test-TC-contact_CH33.csv"
        )

        self.good_file = os.path.join(
            TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29.csv"
        )

        self.broken_file = os.path.join(
            TEST_FILE_DIR, "Talos_001385_NCR18650618003_CH33_truncated.csv"

        )

    # from RCRT.test_serialization
    def test_serialization(self):
        smaller_run = ArbinDatapath.from_file(self.bad_file)
        with ScratchDir("."):
            dumpfn(smaller_run, "smaller_cycler_run.json")
            resurrected = loadfn("smaller_cycler_run.json")
            self.assertIsInstance(resurrected, BEEPDatapath)
            self.assertIsInstance(resurrected.raw_data, pd.DataFrame)
            self.assertEqual(
                smaller_run.raw_data.voltage.to_list(), resurrected.raw_data.voltage.to_list()
            )
            self.assertEqual(
                smaller_run.raw_data.current.to_list(), resurrected.raw_data.current.to_list()
            )

    def test_from_file(self):
        ad = ArbinDatapath.from_file(self.good_file)
        self.assertEqual(ad.paths.get("raw"), self.good_file)
        self.assertEqual(ad.paths.get("metadata"), self.good_file.replace(".csv", "_Metadata.csv"))
        self.assertTupleEqual(ad.raw_data.shape, (251263, 15))

    # based on PCRT.test_from_arbin_insufficient_interpolation_length
    def test_from_arbin_insufficient_interpolation_length(self):
        rcycler_run = ArbinDatapath.from_file(self.broken_file)
        vrange, num_points, nominal_capacity, fast_charge, diag = rcycler_run.determine_structuring_parameters()
        # print(diag['parameter_set'])
        self.assertEqual(diag['parameter_set'], 'NCR18650-618')
        diag_interp = rcycler_run.interpolate_diagnostic_cycles(diag, resolution=1000, v_resolution=0.0005)
        self.assertAlmostEqual(diag_interp[(diag_interp.cycle_index == 1) &
                                           (diag_interp.step_index == 5)].charge_capacity.max(),
                               3.39608899, 3)
        self.assertAlmostEqual(diag_interp[(diag_interp.cycle_type == "hppc")].charge_capacity.max(),
                               3.4919972, 3)


class TestMaccorDatapath(unittest.TestCase):
    """
    Tests specific to Maccor cyclers.
    """

    def setUp(self) -> None:
        self.good_file = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000019_CH70.070")
        self.diagnostics_file = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000020_CH71.071")
        self.waveform_file = os.path.join(TEST_FILE_DIR, "test_drive_071620.095")
        self.broken_file = os.path.join(TEST_FILE_DIR, "PreDiag_000229_000229_truncated.034")
        self.timezone_file = os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_000109_tztest.010")
        self.timestamp_file = os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_000151_test.052")

        # Note: this is a legacy file
        self.diagnostic_interpolation_file = os.path.join(TEST_FILE_DIR,
                                                          "PredictionDiagnostics_000132_00004C_structure.json")

    # based on RCRT.test_ingestion_maccor
    # based on RCRT.test_timezone_maccor
    # based on RCRT.test_timestamp_maccor
    def test_from_file(self):

        for file in (self.good_file, self.timestamp_file, self.timezone_file):
            md = MaccorDatapath.from_file(file)
            self.assertEqual(md.paths.get("raw"), file)
            self.assertEqual(md.paths.get("metadata"), file)

            if file == self.good_file:
                self.assertTupleEqual(md.raw_data.shape, (11669, 40))
                self.assertEqual(70, md.metadata.channel_id)
            elif file == self.timestamp_file:
                self.assertTupleEqual(md.raw_data.shape, (333, 44))
                self.assertEqual(52, md.metadata.channel_id)
            else:
                self.assertTupleEqual(md.raw_data.shape, (2165, 44))
                self.assertEqual(10, md.metadata.channel_id)

            self.assertEqual(
                set(md.metadata.raw.keys()),
                {
                    "barcode",
                    "_today_datetime",
                    "start_datetime",
                    "filename",
                    "protocol",
                    "channel_id",
                },
            )

            # Quick test to see whether columns get recasted
            self.assertTrue(
                {
                    "data_point",
                    "cycle_index",
                    "step_index",
                    "voltage",
                    "current",
                    "charge_capacity",
                    "discharge_capacity",
                }
                < set(md.raw_data.columns)
            )

    # based on RCRT.test_quantity_sum_maccor
    def test_get_quantity_sum(self):
        md = MaccorDatapath.from_file(self.diagnostics_file)

        cycle_sign = np.sign(np.diff(md.raw_data["cycle_index"]))
        capacity_sign = np.sign(np.diff(md.raw_data["charge_capacity"]))
        self.assertTrue(
            np.all(capacity_sign >= -cycle_sign)
        )  # Capacity increases throughout cycle
        capacity_sign = np.sign(np.diff(md.raw_data["discharge_capacity"]))
        self.assertTrue(
            np.all(capacity_sign >= -cycle_sign)
        )  # Capacity increases throughout cycle

    # based on RCRT.test_whether_step_is_waveform
    def test_step_is_waveform(self):
        md = MaccorDatapath.from_file(self.waveform_file)
        df = md.raw_data
        self.assertTrue(df.loc[df.cycle_index == 6].
                        groupby("step_index").apply(step_is_waveform_dchg).any())
        self.assertFalse(df.loc[df.cycle_index == 6].
                        groupby("step_index").apply(step_is_waveform_chg).any())
        self.assertFalse(df.loc[df.cycle_index == 3].
                        groupby("step_index").apply(step_is_waveform_dchg).any())

    # based on RCRT.test_get_interpolated_waveform_discharge_cycles
    def test_interpolate_waveform_discharge_cycles(self):
        md = MaccorDatapath.from_file(self.waveform_file)
        all_interpolated = md.interpolate_cycles()
        all_interpolated = all_interpolated[(all_interpolated.step_type == "discharge")]
        self.assertTrue(all_interpolated.columns[0] == 'test_time')
        subset_interpolated = all_interpolated[all_interpolated.cycle_index==6]

        df = md.raw_data
        self.assertEqual(subset_interpolated.test_time.min(),
                         df.loc[(df.cycle_index == 6) &
                                (df.step_index == 33)].test_time.min())
        self.assertEqual(subset_interpolated[subset_interpolated.cycle_index == 6].shape[0], 1000)

    # based on RCRT.test_waveform_charge_discharge_capacity
    def test_waveform_charge_discharge_capacity(self):
        md = MaccorDatapath.from_file(self.waveform_file)
        df = md.raw_data
        cycle_sign = np.sign(np.diff(df["cycle_index"]))
        capacity_sign = np.sign(np.diff(df["charge_capacity"]))
        self.assertTrue(
            np.all(capacity_sign >= -cycle_sign)
        )  # Capacity increases throughout cycle
        capacity_sign = np.sign(np.diff(df["discharge_capacity"]))
        self.assertTrue(
            np.all(capacity_sign >= -cycle_sign)
        )

    # based on RCRT.test_get_interpolated_cycles_maccor
    def test_interpolate_cycles(self):
        md = MaccorDatapath.from_file(self.good_file)
        all_interpolated = md.interpolate_cycles(
            v_range=[3.0, 4.2], resolution=10000
        )

        self.assertSetEqual(set(all_interpolated.columns.tolist()),
                            {'voltage',
                             'test_time',
                             'discharge_capacity',
                             'discharge_energy',
                             'current',
                             'charge_capacity',
                             'charge_energy',
                             'internal_resistance',
                             'temperature',
                             'cycle_index',
                             'step_type'}
                            )
        interp2 = all_interpolated[
            (all_interpolated.cycle_index == 2)
            & (all_interpolated.step_type == "discharge")
            ].sort_values("discharge_capacity")
        interp3 = all_interpolated[
            (all_interpolated.cycle_index == 1)
            & (all_interpolated.step_type == "charge")
            ].sort_values("charge_capacity")

        self.assertTrue(interp3.current.mean() > 0)
        self.assertEqual(len(interp3.voltage), 10000)
        self.assertEqual(interp3.voltage.max(), np.float32(4.100838))
#        self.assertEqual(interp3.voltage.min(), np.float32(3.3334765)) # 3.437705 in python3.9
        np.testing.assert_almost_equal(
            interp3[
                interp3.charge_capacity <= interp3.charge_capacity.median()
                ].current.iloc[0],
            2.423209,
            decimal=6,
        )

        df = md.raw_data
        cycle_2 = df[df["cycle_index"] == 2]
        discharge = cycle_2[cycle_2.step_index == 12]
        discharge = discharge.sort_values("discharge_capacity")

        acceptable_error = 0.01
        acceptable_error_offest = 0.001
        voltages_to_check = [3.3, 3.2, 3.1]
        columns_to_check = [
            "voltage",
            "current",
            "discharge_capacity",
            "charge_capacity",
        ]
        for voltage_check in voltages_to_check:
            closest_interp2_index = interp2.index[
                (interp2["voltage"] - voltage_check).abs().min()
                == (interp2["voltage"] - voltage_check).abs()
                ]
            closest_interp2_match = interp2.loc[closest_interp2_index]
            closest_discharge_index = discharge.index[
                (discharge["voltage"] - voltage_check).abs().min()
                == (discharge["voltage"] - voltage_check).abs()
                ]
            closest_discharge_match = discharge.loc[closest_discharge_index]
            for column_check in columns_to_check:
                off_by = (
                        closest_interp2_match.iloc[0][column_check]
                        - closest_discharge_match.iloc[0][column_check]
                )
                self.assertLessEqual(np.abs(off_by),
                        np.abs(closest_interp2_match.iloc[0][column_check])
                        * acceptable_error
                        + acceptable_error_offest)

    # based on PCRT.test_from_maccor_insufficient_interpolation_length
    def test_from_maccor_insufficient_interpolation_length(self):
        md = MaccorDatapath.from_file(self.broken_file)
        vrange, num_points, nominal_capacity, fast_charge, diag = md.determine_structuring_parameters()
        self.assertEqual(diag['parameter_set'], 'Tesla21700')
        diag_interp = md.interpolate_diagnostic_cycles(diag, resolution=1000, v_resolution=0.0005)
        self.assertEqual(np.around(diag_interp[diag_interp.cycle_index == 1].charge_capacity.median(), 3),
                         np.around(0.6371558214610992, 3))

    # based on EISpectrumTest.test_from_maccor
    # todo: needs testing for the entire maccor object
    def test_eis(self):
        path = os.path.join(TEST_FILE_DIR, "maccor_test_file_4267-66-6519.EDA0001.041")
        d = MaccorDatapath.MaccorEIS.from_file(path)


class TestIndigoDatapath(unittest.TestCase):
    # based on RCRT.test_ingestion_indigo
    def test_from_file(self):
        indigo_file = os.path.join(TEST_FILE_DIR, "indigo_test_sample.h5")
        md = IndigoDatapath.from_file(indigo_file)
        self.assertTrue(
            {
                "data_point",
                "cycle_index",
                "step_index",
                "voltage",
                "temperature",
                "current",
                "charge_capacity",
                "discharge_capacity",
            }
            < set(md.raw_data.columns)
        )

        self.assertEqual(
            set(md.metadata.raw.keys()),
            {"indigo_cell_id", "_today_datetime", "start_datetime", "filename"},
        )


class TestBioLogicDatapath(unittest.TestCase):
    # based on RCRT.test_ingestion_biologic
    def test_from_csv(self):

        biologic_file = os.path.join(
            TEST_FILE_DIR, "raw", "test_loopsnewoutput_MB_CE1_short10k.csv"
        )
        dp = BiologicDatapath.from_file(biologic_file)

        self.assertTrue(
            {
                "cycle_index",
                "step_index",
                "voltage",
                "current",
                "discharge_capacity",
                "charge_capacity",
                "data_point",
                "charge_energy",
                "discharge_energy",
            }
            < set(dp.raw_data.columns),
        )

        self.assertEqual(
            {"_today_datetime", "filename", "barcode", "protocol", "channel_id"},
            set(dp.metadata.raw.keys()),
        )

        dp.structure(v_range=[3.0, 4.4])

        self.assertAlmostEqual(dp.structured_summary["charge_capacity"].tolist()[0], 1.4618750, 6)
        self.assertAlmostEqual(dp.structured_summary["discharge_capacity"].tolist()[0], 2.324598, 6)
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[0], "2021-05-05T22:36:22.757000+00:00")
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[1], "2021-05-06T09:44:45.604000+00:00")
        self.assertAlmostEqual(dp.raw_data["test_time"].min(), 0, 3)
        self.assertAlmostEqual(dp.raw_data["test_time"].max(), 102040.77, 3)
        # self.assertAlmostEqual(dp.structured_data["test_time"].min(), 13062.720560, 3)
        self.assertAlmostEqual(dp.structured_data["test_time"].min(), 101.089, 2)
        self.assertAlmostEqual(dp.structured_data["test_time"].max(), 101972.885, 3)

    def test_from_txt(self):
        biologic_file = os.path.join(
            TEST_FILE_DIR, "raw", "test_loopsnewoutput_MB_CE1_short10k.txt"
        )
        dp = BiologicDatapath.from_file(biologic_file)

        self.assertTrue(
            {
                "cycle_index",
                "step_index",
                "voltage",
                "current",
                "discharge_capacity",
                "charge_capacity",
                "data_point",
                "charge_energy",
                "discharge_energy",
            }
            < set(dp.raw_data.columns),
        )

        self.assertEqual(
            {"_today_datetime", "filename", "barcode", "protocol", "channel_id"},
            set(dp.metadata.raw.keys()),
        )
        dp.structure(v_range=[3.0, 4.4])

        self.assertAlmostEqual(dp.structured_summary["charge_capacity"].tolist()[0], 1.4618487, 6)
        self.assertAlmostEqual(dp.structured_summary["discharge_capacity"].tolist()[0], 2.324598, 6)
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[0], "2021-05-05T22:36:00+00:00")
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[1], "2021-05-06T09:44:22.848000+00:00")
        self.assertAlmostEqual(dp.raw_data["test_time"].min(), 0, 3)
        self.assertAlmostEqual(dp.raw_data["test_time"].max(), 102240.281, 3)
        #self.assertAlmostEqual(dp.structured_data["test_time"].min(), 13062.997, 3)
        self.assertAlmostEqual(dp.structured_data["test_time"].min(), 101.357, 2)
        self.assertAlmostEqual(dp.structured_data["test_time"].max(), 101972.886, 3)

    def test_from_formation_txt(self):
        biologic_file = os.path.join(
            TEST_FILE_DIR, "raw", "test_FormRegu_000100_CG1_Append_short.txt"
        )
        dp = BiologicDatapath.from_file(biologic_file)

        self.assertTrue(
            {
                "cycle_index",
                "step_index",
                "voltage",
                "current",
                "discharge_capacity",
                "charge_capacity",
                "data_point",
                "charge_energy",
                "discharge_energy",
            }
            < set(dp.raw_data.columns),
        )

        dp.structure(v_range=[1.0, 4.4])

        self.assertAlmostEqual(dp.structured_summary["charge_capacity"].tolist()[0], 0.2673875, 6)
        self.assertAlmostEqual(dp.structured_summary["discharge_capacity"].tolist()[0], 0.2573631, 6)
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[0], "2022-01-18T22:09:40.640000+00:00")
        self.assertEqual(dp.structured_summary["date_time_iso"].iloc[1], "2022-01-22T09:15:10.020000+00:00")
        self.assertAlmostEqual(dp.raw_data["test_time"].min(), 0, 3)
        self.assertAlmostEqual(dp.raw_data["test_time"].max(), 784864.55, 3)
        self.assertAlmostEqual(dp.structured_data["test_time"].min(), 259220.283, 3)
        self.assertAlmostEqual(dp.structured_data["test_time"].max(), 784853.102, 3)
        self.assertGreater(dp.structured_summary["discharge_capacity"].tolist()[4], 0)
        self.assertGreater(dp.structured_summary["discharge_capacity"].tolist()[20], 0)
        self.assertGreater(dp.structured_summary["discharge_capacity"].tolist()[40], 0)

    def test_add_cycle_index(self):

        biologic_file = os.path.join(
            TEST_FILE_DIR, "raw", "test_loopsnewoutput_MB_CE1_short10k.csv"
        )
        df = pd.read_csv(biologic_file, sep=";")
        ns_list = df["Ns"].tolist()
        loop_list = df["Loop"].tolist()
        biotest_file = os.path.join(TEST_FILE_DIR, "BioTest_000001.000.technique_1_cycle_rules.json")
        cycle_index = get_cycle_index(ns_list, biotest_file, loop_list=loop_list)
        c_i = pd.Series(cycle_index)
        self.assertListEqual([1, 2, 3], c_i.unique().tolist())

    def test_mapping_file(self):
        biologic_file = os.path.join(
            TEST_FILE_DIR, "raw", "test_loopsnewoutput_MB_CE1_short10k.txt"
        )
        biotest_file = os.path.join(TEST_FILE_DIR, "BioTest_000001.000.technique_1_cycle_rules.json")
        dp = BiologicDatapath.from_file(biologic_file, mapping_file=biotest_file)
        self.assertIn("cycle_index", dp.raw_data.columns)
        self.assertListEqual([1, 2, 3], dp.raw_data["cycle_index"].unique().tolist())

class TestNewareDatapath(unittest.TestCase):
    # based on RCRT.test_ingestion_neware
    def test_from_file(self):
        neware_file = os.path.join(TEST_FILE_DIR, "raw", "neware_test.csv")
        md = NewareDatapath.from_file(neware_file)
        self.assertEqual(md.raw_data.columns[22], "internal_resistance")
        self.assertTrue(md.raw_data["test_time"].is_monotonic_increasing)
        summary = md.summarize_cycles(nominal_capacity=4.7, full_fast_charge=0.8)
        self.assertEqual(summary["discharge_capacity"].head(5).round(4).tolist(),
                         [2.4393, 2.4343, 2.4255, 2.4221, 2.4210])
        self.assertEqual(summary[summary["cycle_index"] == 55]["discharge_capacity"].round(4).tolist(),
                         [2.3427])


class TestBatteryArchiveDatapath(unittest.TestCase):
    def test_from_file(self):
        ba_file = os.path.join(TEST_FILE_DIR,
                               "SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv")
        bd = BatteryArchiveDatapath.from_file(ba_file)

        self.assertEqual(bd.raw_data.columns[-1], "internal_resistance")
        self.assertEqual(bd.raw_data.columns[0], "date_time")
        self.assertTrue(bd.raw_data["test_time"].is_monotonic_increasing)

        summary = bd.summarize_cycles()
        self.assertAlmostEqual(summary["temperature_maximum"].loc[3], 16.832001, places=4)
        self.assertAlmostEqual(summary["charge_duration"].loc[4548], 5773.640137, places=4)


if __name__ == "__main__":
    unittest.main()
