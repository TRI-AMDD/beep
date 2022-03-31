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
import copy
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir

from beep.conversion_schemas import STRUCTURE_DTYPES
from beep.utils.s3 import download_s3_object
from beep.structure.base import BEEPDatapath
from beep.structure.base_eis import EIS, BEEPDatapathWithEIS
from beep.structure.maccor import MaccorDatapath
from beep.tests.constants import TEST_FILE_DIR, BIG_FILE_TESTS, SKIP_MSG
from beep.structure.cli import auto_load_processed
from beep import VALIDATION_SCHEMA_DIR


class TestBEEPDatapath(unittest.TestCase):
    """
    Tests common to all datapaths.
    """

    class BEEPDatapathChildTest(BEEPDatapath):
        """
        A test class representing any child of BEEPDatapath.
        """

        @classmethod
        def from_file(cls, path, index_col=0, **kwargs):
            return pd.read_csv(path, index_col=0, **kwargs)

    @classmethod
    def setUpClass(cls) -> None:

        # Use arbin memloaded inputs as source of non-diagnostic truth
        arbin_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_arbin_memloaded.csv")
        arbin_meta_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_arbin_metadata_memloaded.json")
        cls.data_nodiag = pd.read_csv(arbin_fname, index_col=0)
        cls.metadata_nodiag = loadfn(arbin_meta_fname)
        cls.datapath_nodiag = cls.BEEPDatapathChildTest(
            raw_data=cls.data_nodiag,
            metadata=cls.metadata_nodiag,
            paths={"raw": arbin_fname, "raw_metadata": arbin_meta_fname}
        )

        # Use maccor memloaded inputs as source of diagnostic truth
        maccor_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_w_diagnostic_memloaded.csv")
        maccor_meta_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_w_diagnostic_metadata_memloaded.json")
        cls.data_diag = pd.read_csv(maccor_fname, index_col=0)
        cls.metadata_diag = loadfn(maccor_meta_fname)
        cls.datapath_diag = cls.BEEPDatapathChildTest(
            raw_data=cls.data_diag,
            metadata=cls.metadata_diag,
            paths={"raw": maccor_fname, "raw_metadata": maccor_meta_fname}

        )

        # Use maccor paused memloaded inputs as source of paused run truth
        maccor_paused_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_paused_memloaded.csv")
        maccor_paused_meta_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_paused_metadata_memloaded.json")
        cls.data_paused = pd.read_csv(maccor_paused_fname, index_col=0)
        cls.metadata_paused = loadfn(maccor_paused_meta_fname)
        cls.datapath_paused = cls.BEEPDatapathChildTest(
            raw_data=cls.data_paused,
            metadata=cls.metadata_paused,
            paths={"raw": maccor_paused_fname, "raw_metadata": maccor_paused_meta_fname}
        )

        # Small maccor file with parameters
        maccor_small_params_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_parameterized_memloaded.csv")
        maccor_small_params_meta_fname = os.path.join(TEST_FILE_DIR,
                                                      "BEEPDatapath_maccor_parameterized_metadata_memloaded.json")
        cls.data_small_params = pd.read_csv(maccor_small_params_fname, index_col=0)
        cls.metadata_small_params = loadfn(maccor_small_params_meta_fname)
        cls.datapath_small_params = cls.BEEPDatapathChildTest(
            raw_data=cls.data_small_params,
            metadata=cls.metadata_small_params,
            paths={"raw": maccor_small_params_fname, "raw_metadata": maccor_small_params_meta_fname}
        )

        # Maccor with various diagnostics from memory
        # For testing determine_structuring_parameters
        maccor_diag_normal_fname = os.path.join(TEST_FILE_DIR, "BEEPDatapath_maccor_diagnostic_normal_memloaded.csv")
        maccor_diag_normal_meta_fname = os.path.join(TEST_FILE_DIR,
                                                     "BEEPDatapath_maccor_diagnostic_normal_metadata_memloaded.json")
        maccor_diag_normal_original_fname = os.path.join(TEST_FILE_DIR, "PreDiag_000287_000128short.092")
        cls.data_diag_normal = pd.read_csv(maccor_diag_normal_fname, index_col=0)
        cls.metadata_diag_normal = loadfn(maccor_diag_normal_meta_fname)
        cls.datapath_diag_normal = cls.BEEPDatapathChildTest(
            raw_data=cls.data_diag_normal,
            metadata=cls.metadata_diag_normal,
            paths={"raw": maccor_diag_normal_original_fname, "raw_metadata": maccor_diag_normal_meta_fname}
        )
        maccor_diag_misplaced_fname = os.path.join(TEST_FILE_DIR,
                                                   "BEEPDatapath_maccor_diagnostic_misplaced_memloaded.csv")
        maccor_diag_misplaced_meta_fname = os.path.join(
            TEST_FILE_DIR, "BEEPDatapath_maccor_diagnostic_misplaced_metadata_memloaded.json"
        )
        maccor_diag_misplaced_original_fname = os.path.join(TEST_FILE_DIR, "PreDiag_000412_00008Fshort.022")
        cls.data_diag_misplaced = pd.read_csv(maccor_diag_misplaced_fname, index_col=0)
        cls.metadata_diag_misplaced = loadfn(maccor_diag_misplaced_meta_fname)
        cls.datapath_diag_misplaced = cls.BEEPDatapathChildTest(
            raw_data=cls.data_diag_misplaced,
            metadata=cls.metadata_diag_misplaced,
            paths={"raw": maccor_diag_misplaced_original_fname, "raw_metadata": maccor_diag_misplaced_meta_fname}
        )

        cls.cycle_run_file = os.path.join(
            TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29_processed.json"
        )

        cls.diagnostic_available = {
            "type": "HPPC",
            "cycle_type": ["hppc"],
            "length": 1,
            "diagnostic_starts_at": [1],
        }

    def setUp(self) -> None:
        # Reset all datapaths after each run, avoiding colliions between tests
        # and avoiding reloading files
        for dp in [
            self.datapath_diag,
            self.datapath_nodiag,
            self.datapath_diag_normal,
            self.datapath_diag_misplaced,
            self.datapath_paused,
            self.datapath_small_params
        ]:
            dp.unstructure()

    def run_dtypes_check(self, summary):
        reg_dyptes = summary.dtypes.tolist()
        reg_columns = summary.columns.tolist()
        reg_dyptes = [str(dtyp) for dtyp in reg_dyptes]
        for indx, col in enumerate(reg_columns):
            self.assertEqual(reg_dyptes[indx], STRUCTURE_DTYPES["summary"][col])

    def test_abc(self):

        class BEEPDatapathChildBad(BEEPDatapath):
            def some_extra_method(self):
                return True

            # missing required method!
            # should fail

        # Ensure this bad child class has ABC error
        with self.assertRaises(TypeError):
            BEEPDatapathChildBad(raw_data=pd.DataFrame({}), metadata={"some": "metadata"})

    def test_unstructure(self):
        self.datapath_nodiag.structure()
        self.assertTrue(self.datapath_nodiag.is_structured)
        self.datapath_nodiag.unstructure()
        self.assertFalse(self.datapath_nodiag.is_structured)

    def test_serialization(self):
        truth_datapath = self.datapath_diag

        truth_datapath.structure()
        d = truth_datapath.as_dict()
        datapath_from_dict = self.BEEPDatapathChildTest.from_dict(d)

        # Test loading with and without compression, and with and without raw_data via omit_raw
        for fname_short in ("test_serialization.json", "test_serialization.json.gz"):
            for omit_raw in (True, False):

                fname = os.path.join(TEST_FILE_DIR, fname_short)
                truth_datapath.to_json_file(fname, omit_raw=omit_raw)
                datapath_from_json = self.BEEPDatapathChildTest.from_json_file(fname)

                for df_name in ("structured_data", "structured_summary", "diagnostic_data", "diagnostic_summary"):
                    df_truth = getattr(truth_datapath, df_name)
                    for datapath_test in (datapath_from_dict, datapath_from_json):
                        df_test = getattr(datapath_test, df_name)

                        if df_truth is None:
                            self.assertEqual(df_truth, df_test)
                        else:
                            self.assertTrue(isinstance(df_test, pd.DataFrame))
                            self.assertTrue(df_truth.equals(df_test))

                self.assertEqual(datapath_from_json.paths.get("structured"), fname)
                self.assertEqual(datapath_from_json.paths.get("raw"), self.datapath_diag.paths.get("raw"))

                if os.path.exists(fname):
                    os.remove(fname)

    # Test to address bug where schema_path is absolute but is created in a different environment
    def test_reloading_new(self):
        test_file = os.path.join(TEST_FILE_DIR, "PredictionDiagnostics_000107_0001B9_structure_short.json")
        struct = auto_load_processed(test_file)
        self.assertEqual(struct.schema,
                         os.path.join(VALIDATION_SCHEMA_DIR, "schema-maccor-2170.yaml"))

    # based on RCRT.test_serialization
    def test_serialization_legacy(self):
        test_file = os.path.join(
            TEST_FILE_DIR, "2017-12-04_4_65C-69per_6C_CH29_processed.json"
        )

        datapath = self.BEEPDatapathChildTest.from_json_file(test_file)
        self.assertTrue(isinstance(datapath.structured_summary, pd.DataFrame))
        self.assertTrue(isinstance(datapath.structured_data, pd.DataFrame))
        self.assertIsNone(datapath.metadata.barcode)
        self.assertIsNone(datapath.metadata.protocol)
        self.assertEqual(datapath.metadata.channel_id, 28)
        self.assertEqual(datapath.structured_summary.shape[0], 188)
        self.assertEqual(datapath.structured_summary.shape[1], 7)
        self.assertEqual(datapath.structured_data.shape[0], 188000),
        self.assertEqual(datapath.structured_data.shape[1], 7)
        self.assertAlmostEqual(datapath.structured_data["voltage"].loc[0], 2.8, places=5)
        self.assertAlmostEqual(datapath.structured_data["discharge_capacity"].loc[187999], 0.000083, places=6)
        self.assertEqual(datapath.paths.get("structured"), test_file)
        self.assertEqual(datapath.paths.get("raw"), None)

    # based on RCRT.test_get_interpolated_charge_step
    def test_interpolate_step(self):
        reg_cycles = [i for i in self.datapath_nodiag.raw_data.cycle_index.unique()]
        v_range = [2.8, 3.5]
        resolution = 1000

        for step_type in ("charge", "discharge"):
            interpolated_charge = self.datapath_nodiag.interpolate_step(
                v_range,
                resolution,
                step_type=step_type,
                reg_cycles=reg_cycles,
                axis="test_time",
            )
            lengths = [len(df) for index, df in interpolated_charge.groupby("cycle_index")]
            axis_1 = interpolated_charge[
                interpolated_charge.cycle_index == 5
            ].charge_capacity.to_list()
            axis_2 = interpolated_charge[
                interpolated_charge.cycle_index == 10
            ].charge_capacity.to_list()
            self.assertGreater(max(axis_1), max(axis_2))
            self.assertTrue(np.all(np.array(lengths) == 1000))

            if step_type == "charge":
                self.assertTrue(interpolated_charge["current"].mean() > 0)
            else:
                self.assertTrue(interpolated_charge["current"].mean() < 0)

    # based on RCRT.test_get_interpolated_discharge_cycles
    # based on RCRT.test_get_interpolated_charge_cycles
    # based on RCRT.test_interpolated_cycles_dtypes
    def test_interpolate_cycles(self):
        dp = self.datapath_nodiag
        all_interpolated = dp.interpolate_cycles()

        # Test discharge cycles
        dchg = all_interpolated[(all_interpolated.step_type == "discharge")]
        lengths = [len(df) for index, df in dchg.groupby("cycle_index")]
        self.assertTrue(np.all(np.array(lengths) == 1000))

        # Found these manually
        dchg = dchg.drop(columns=["step_type"])
        y_at_point = dchg.iloc[[1500]]
        x_at_point = dchg.voltage[1500]
        cycle_1 = dp.raw_data[dp.raw_data["cycle_index"] == 1]

        # Discharge step is 12
        discharge = cycle_1[cycle_1.step_index == 12]
        discharge = discharge.sort_values("voltage")

        # Get an interval between which one can find the interpolated value
        measurement_index = np.max(np.where(discharge.voltage - x_at_point < 0))
        interval = discharge.iloc[measurement_index: measurement_index + 2]
        interval = interval.drop(columns=["date_time_iso"])  # Drop non-numeric column

        # Test interpolation with a by-hand calculation of slope
        diff = np.diff(interval, axis=0)
        pred = interval.iloc[[0]] + diff * (
                    x_at_point - interval.voltage.iloc[0]) / (
                       interval.voltage.iloc[1] - interval.voltage.iloc[0]
               )
        pred = pred.reset_index()
        for col_name in y_at_point.columns:
            self.assertAlmostEqual(
                pred[col_name].iloc[0], y_at_point[col_name].iloc[0],
                places=2
            )

        # Test charge cycles
        chg = all_interpolated[(all_interpolated.step_type == "charge")]
        lengths = [len(df) for index, df in chg.groupby("cycle_index")]
        axis_1 = chg[chg.cycle_index == 5].charge_capacity.to_list()
        axis_2 = chg[chg.cycle_index == 10].charge_capacity.to_list()
        self.assertEqual(axis_1, axis_2)
        self.assertTrue(np.all(np.array(lengths) == 1000))
        self.assertTrue(chg["current"].mean() > 0)


        # Test dtypes
        cycles_interpolated_dyptes = all_interpolated.dtypes.tolist()
        cycles_interpolated_columns = all_interpolated.columns.tolist()
        cycles_interpolated_dyptes = [str(dtyp) for dtyp in cycles_interpolated_dyptes]
        for indx, col in enumerate(cycles_interpolated_columns):
            self.assertEqual(
                cycles_interpolated_dyptes[indx],
                STRUCTURE_DTYPES["cycles_interpolated"][col],
            )

    # based on RCRT.test_get_summary
    # based on RCRT.test_get_energy
    # based on RCRT.test_get_charge_throughput
    # based on RCRT.test_summary_dtypes
    def test_summarize_cycles(self):
        summary_diag = self.datapath_diag.summarize_cycles(nominal_capacity=4.7, full_fast_charge=0.8)
        self.assertTrue(
            set.issubset(
                {
                    "discharge_capacity",
                    "charge_capacity",
                    "dc_internal_resistance",
                    "temperature_maximum",
                    "temperature_average",
                    "temperature_minimum",
                    "date_time_iso",
                    "charge_throughput",
                    "energy_throughput",
                    "charge_energy",
                    "discharge_energy",
                    "energy_efficiency",
                    "CV_time",
                    "CV_current",
                    "CV_capacity"
                },
                set(summary_diag.columns),
            )
        )
        self.assertEqual(summary_diag["cycle_index"].tolist(), list(range(0, 13)))
        self.assertEqual(len(summary_diag.index), len(summary_diag["date_time_iso"]))
        self.assertEqual(summary_diag["paused"].max(), 0)
        self.assertEqual(summary_diag["CV_time"][1], np.float32(160111.796875))
        self.assertEqual(summary_diag["CV_current"][1], np.float32(0.4699016))
        self.assertEqual(summary_diag["CV_capacity"][1], np.float32(94.090355))
        self.run_dtypes_check(summary_diag)

        # incorporates test_get_energy and get_charge_throughput
        summary = self.datapath_nodiag.summarize_cycles(nominal_capacity=4.7, full_fast_charge=0.8)
        self.assertEqual(np.around(summary["charge_energy"][5], 6), np.around(3.7134638, 6))
        self.assertEqual(np.around(summary["energy_efficiency"][5], 7), np.around(np.float32(0.872866405753033), 7))
        self.assertEqual(summary["charge_throughput"][5], np.float32(6.7614093))
        self.assertEqual(summary["energy_throughput"][5], np.float32(23.2752363))
        self.run_dtypes_check(summary)

    # based on RCRT.test_determine_structuring_parameters
    def test_determine_structuring_parameters(self):
        (v_range, resolution, nominal_capacity, full_fast_charge, diagnostic_available) = \
            self.datapath_diag_normal.determine_structuring_parameters()
        diagnostic_available_test = {
            "parameter_set": "Tesla21700",
            "cycle_type": ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
            "length": 5,
            "diagnostic_starts_at": [
                1, 36, 141, 246, 351, 456, 561, 666, 771, 876, 981, 1086,
                1191,
                1296, 1401, 1506, 1611, 1716, 1821, 1926, 2031, 2136, 2241,
                2346,
                2451, 2556, 2661, 2766, 2871, 2976, 3081, 3186, 3291, 3396,
                3501,
                3606, 3628
            ]
        }

        self.assertEqual(v_range, [2.5, 4.2])
        self.assertEqual(resolution, 1000)
        self.assertEqual(nominal_capacity, 4.84)
        self.assertEqual(full_fast_charge, 0.8)
        self.assertEqual(diagnostic_available, diagnostic_available_test)

        (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        ) = self.datapath_diag_misplaced.determine_structuring_parameters()
        diagnostic_available_test = {
            "parameter_set": "Tesla21700",
            "cycle_type": ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
            "length": 5,
            "diagnostic_starts_at": [1, 36, 141, 220, 255]
        }
        self.assertEqual(v_range, [2.5, 4.2])
        self.assertEqual(resolution, 1000)
        self.assertEqual(nominal_capacity, 4.84)
        self.assertEqual(full_fast_charge, 0.8)
        self.assertEqual(diagnostic_available, diagnostic_available_test)

    # based on RCRT.test_get_interpolated_diagnostic_cycles
    def test_interpolate_diagnostic_cycles(self):
        d_interp = self.datapath_diag.interpolate_diagnostic_cycles(
            self.diagnostic_available, resolution=500
        )
        self.assertGreaterEqual(len(d_interp.cycle_index.unique()), 1)

        # Ensure step indices are partitioned and processed separately
        self.assertEqual(len(d_interp.step_index.unique()), 9)
        first_step = d_interp[
            (d_interp.step_index == 7) & (d_interp.step_index_counter == 1)
        ]
        second_step = d_interp[
            (d_interp.step_index == 7) & (d_interp.step_index_counter == 4)
        ]
        self.assertLess(first_step.voltage.diff().max(), 0.001)
        self.assertLess(second_step.voltage.diff().max(), 0.001)

    # based on RCRT.test_get_diagnostic_summary
    def test_summarize_diagnostic(self):
        diag_summary = self.datapath_diag.summarize_diagnostic(self.diagnostic_available)
        self.assertEqual(diag_summary["paused"].max(), 0)
        self.assertEqual(diag_summary["CV_time"][0], np.float32(125502.578125))
        self.assertEqual(diag_summary["CV_current"][0], np.float32(2.3499656))
        self.assertEqual(diag_summary["CV_capacity"][0], np.float32(84.70442))

    # based on RCRT.test_determine_paused
    def test_paused_intervals(self):
        paused = self.datapath_paused.paused_intervals
        self.assertEqual(paused.max(), 7201.0)

        not_paused = self.datapath_diag.paused_intervals
        self.assertEqual(not_paused.max(), 0.0)

    # based on RCRT.test_get_diagnostic
    # though it is based on maccor files this test is designed to
    # check structuring of diagnostic cycles
    @unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
    def test_get_diagnostic(self):
        maccor_file_w_parameters_s3 = {
            "bucket": "beep-sync-test-stage",
            "key": "big_file_tests/PreDiag_000287_000128.092"
        }

        maccor_file_w_parameters = os.path.join(
            TEST_FILE_DIR, "PreDiag_000287_000128.092"
        )

        download_s3_object(bucket=maccor_file_w_parameters_s3["bucket"],
                           key=maccor_file_w_parameters_s3["key"],
                           destination_path=maccor_file_w_parameters)

        md = MaccorDatapath.from_file(maccor_file_w_parameters)

        (
            v_range,
            resolution,
            nominal_capacity,
            full_fast_charge,
            diagnostic_available,
        ) = md.determine_structuring_parameters()

        self.assertEqual(nominal_capacity, 4.84)
        # self.assertEqual(v_range, [2.7, 4.2]) # This is an older assertion, value changed when
        # different cell types were added

        self.assertEqual(v_range, [2.5, 4.2])
        self.assertEqual(
            diagnostic_available["cycle_type"],
            ["reset", "hppc", "rpt_0.2C", "rpt_1C", "rpt_2C"],
        )
        diag_summary = md.summarize_diagnostic(diagnostic_available)

        reg_summary = md.summarize_cycles(diagnostic_available)
        self.assertEqual(len(reg_summary.cycle_index.tolist()), 230)
        self.assertEqual(reg_summary.cycle_index.tolist()[:10],
                         [0, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        # Check data types are being set correctly for diagnostic summary
        diag_dyptes = diag_summary.dtypes.tolist()
        diag_columns = diag_summary.columns.tolist()
        diag_dyptes = [str(dtyp) for dtyp in diag_dyptes]
        for indx, col in enumerate(diag_columns):
            self.assertEqual(
                diag_dyptes[indx], STRUCTURE_DTYPES["diagnostic_summary"][col]
            )

        self.assertEqual(
            diag_summary.cycle_index.tolist(),
            [1, 2, 3, 4, 5, 36, 37, 38, 39, 40, 141, 142, 143, 144, 145, 246,
             247],
        )
        self.assertEqual(
            diag_summary.cycle_type.tolist(),
            [
                "reset",
                "hppc",
                "rpt_0.2C",
                "rpt_1C",
                "rpt_2C",
                "reset",
                "hppc",
                "rpt_0.2C",
                "rpt_1C",
                "rpt_2C",
                "reset",
                "hppc",
                "rpt_0.2C",
                "rpt_1C",
                "rpt_2C",
                "reset",
                "hppc",
            ],
        )
        self.assertEqual(diag_summary.paused.max(), 0)
        diag_interpolated = md.interpolate_diagnostic_cycles(
            diagnostic_available, resolution=1000
        )

        # Check data types are being set correctly for interpolated data
        diag_dyptes = diag_interpolated.dtypes.tolist()
        diag_columns = diag_interpolated.columns.tolist()
        diag_dyptes = [str(dtyp) for dtyp in diag_dyptes]
        for indx, col in enumerate(diag_columns):
            self.assertEqual(
                diag_dyptes[indx],
                STRUCTURE_DTYPES["diagnostic_interpolated"][col]
            )

        # Provide visual inspection to ensure that diagnostic interpolation is being done correctly
        diag_cycle = diag_interpolated[
            (diag_interpolated.cycle_type == "rpt_0.2C")
            & (diag_interpolated.step_type == 1)
            ]
        self.assertEqual(diag_cycle.cycle_index.unique().tolist(), [3, 38, 143])
        plt.figure()
        plt.plot(diag_cycle.discharge_capacity, diag_cycle.voltage)
        plt.savefig(
            os.path.join(TEST_FILE_DIR, "discharge_capacity_interpolation.png"))
        plt.figure()
        plt.plot(diag_cycle.voltage, diag_cycle.discharge_dQdV)
        plt.savefig(
            os.path.join(TEST_FILE_DIR, "discharge_dQdV_interpolation.png"))

        self.assertEqual(len(diag_cycle.index), 3000)

        hppcs = diag_interpolated[
            (diag_interpolated.cycle_type == "hppc")
            & pd.isnull(diag_interpolated.current)
            ]
        self.assertEqual(len(hppcs), 0)

        hppc_dischg1 = diag_interpolated[
            (diag_interpolated.cycle_index == 37)
            & (diag_interpolated.step_type == 2)
            & (diag_interpolated.step_index_counter == 3)
            & ~pd.isnull(diag_interpolated.current)
            ]

        plt.figure()
        plt.plot(hppc_dischg1.test_time, hppc_dischg1.voltage)
        plt.savefig(os.path.join(TEST_FILE_DIR, "hppc_discharge_pulse_1.png"))
        self.assertEqual(len(hppc_dischg1), 176)

        hppc_dischg2 = diag_interpolated[
            (diag_interpolated.cycle_index == 37)
            & (diag_interpolated.step_type == 6)
            # & (diag_interpolated.step_index_counter == 3)
            & ~pd.isnull(diag_interpolated.current)
            ]
        print(hppc_dischg2.step_type.unique())
        self.assertAlmostEqual(hppc_dischg2.voltage.min(), hppc_dischg2.voltage.max(), 3)
        plt.figure()
        plt.plot(hppc_dischg2.test_time, hppc_dischg2.current)
        plt.savefig(os.path.join(TEST_FILE_DIR, "hppc_cv.png"))
        self.assertEqual(len(hppc_dischg2), 1000)

        # processed_cycler_run = cycler_run.to_processed_cycler_run()
        md.autostructure()
        self.assertNotIn(
            diag_summary.cycle_index.tolist(),
            md.structured_data.cycle_index.unique(),
        )
        self.assertEqual(
            reg_summary.cycle_index.tolist(),
            md.structured_summary.cycle_index.tolist(),
        )

        processed_cycler_run_loc = os.path.join(
            TEST_FILE_DIR, "processed_diagnostic.json"
        )
        # Dump to the structured file and check the file size
        # File size had to be incteased as datapath dump includes ALL data now
        dumpfn(md, processed_cycler_run_loc)
        proc_size = os.path.getsize(processed_cycler_run_loc)
        self.assertLess(proc_size, 260000000)

        # Reload the structured file and check for errors
        test = loadfn(processed_cycler_run_loc)
        self.assertIsInstance(test.diagnostic_summary, pd.DataFrame)
        diag_dyptes = test.diagnostic_summary.dtypes.tolist()
        diag_columns = test.diagnostic_summary.columns.tolist()
        diag_dyptes = [str(dtyp) for dtyp in diag_dyptes]
        for indx, col in enumerate(diag_columns):
            self.assertEqual(
                diag_dyptes[indx], STRUCTURE_DTYPES["diagnostic_summary"][col]
            )

        diag_dyptes = test.diagnostic_data.dtypes.tolist()
        diag_columns = test.diagnostic_data.columns.tolist()
        diag_dyptes = [str(dtyp) for dtyp in diag_dyptes]
        for indx, col in enumerate(diag_columns):
            self.assertEqual(
                diag_dyptes[indx],
                STRUCTURE_DTYPES["diagnostic_interpolated"][col]
            )

        self.assertEqual(test.structured_summary.cycle_index.tolist()[:10],
                         [0, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        plt.figure()

        single_charge = test.structured_data[
            (test.structured_data.step_type == "charge")
            & (test.structured_data.cycle_index == 25)
            ]
        self.assertEqual(len(single_charge.index), 1000)
        plt.plot(single_charge.charge_capacity, single_charge.voltage)
        plt.savefig(
            os.path.join(
                TEST_FILE_DIR, "charge_capacity_interpolation_regular_cycle.png"
            )
        )

        os.remove(processed_cycler_run_loc)

    # based on PCRT.test_from_raw_cycler_run_parameters
    def test_structure(self):
        self.datapath_small_params.structure()
        self.assertEqual(self.datapath_small_params.metadata.barcode, "0001BC")
        self.assertEqual(self.datapath_small_params.metadata.protocol, "PredictionDiagnostics_000109.000")
        self.assertEqual(self.datapath_small_params.metadata.channel_id, 10)
        self.assertTrue(self.datapath_small_params.is_structured)

    def test_autostructure(self):
        self.datapath_diag_normal.autostructure()
        self.assertEqual(3448, len(self.datapath_diag_normal.structured_summary))
        self.assertEqual(4957, self.datapath_diag_normal.structured_summary.paused.iloc[0])

        self.assertEqual("reset", self.datapath_diag_normal.diagnostic_summary.cycle_type.iloc[0])
        self.assertEqual(4.711819, np.round(self.datapath_diag_normal.diagnostic_summary.discharge_capacity.iloc[0], 6))

    # based on PCRT.test_get_cycle_life
    def test_get_cycle_life(self):
        datapath = self.BEEPDatapathChildTest.from_json_file(self.cycle_run_file)
        self.assertEqual(datapath.get_cycle_life(30, 0.99), 82)
        self.assertEqual(datapath.get_cycle_life(40, 0.0), 189)

    # based on PCRT.test_data_types_old_processed
    def test_data_types_old_processed(self):
        datapath = self.BEEPDatapathChildTest.from_json_file(self.cycle_run_file)

        all_summary = datapath.structured_summary
        reg_dyptes = all_summary.dtypes.tolist()
        reg_columns = all_summary.columns.tolist()
        reg_dyptes = [str(dtyp) for dtyp in reg_dyptes]
        for indx, col in enumerate(reg_columns):
            self.assertEqual(reg_dyptes[indx], STRUCTURE_DTYPES["summary"][col])

        all_interpolated = datapath.structured_data
        cycles_interpolated_dyptes = all_interpolated.dtypes.tolist()
        cycles_interpolated_columns = all_interpolated.columns.tolist()
        cycles_interpolated_dyptes = [str(dtyp) for dtyp in cycles_interpolated_dyptes]
        for indx, col in enumerate(cycles_interpolated_columns):
            self.assertEqual(
                cycles_interpolated_dyptes[indx],
                STRUCTURE_DTYPES["cycles_interpolated"][col],
            )

    # based on PCRT.test_cycles_to_reach_set_capacities
    def test_capacities_to_cycles(self):
        datapath = self.BEEPDatapathChildTest.from_json_file(self.cycle_run_file)
        cycles = datapath.capacities_to_cycles()
        self.assertGreaterEqual(cycles.iloc[0, 0], 100)

    # based on PCRT.test_capacities_at_set_cycles
    def test_cycles_to_capacities(self):
        datapath = self.BEEPDatapathChildTest.from_json_file(self.cycle_run_file)
        capacities = datapath.cycles_to_capacities()
        self.assertLessEqual(capacities.iloc[0, 0], 1.1)

    def test_semiunique_id(self):
        dp = copy.deepcopy(self.datapath_small_params)

        # Ensure repeated calls produce the same id
        self.assertEqual(dp.semiunique_id, dp.semiunique_id)

        if dp.is_structured:
            dp.unstructure()
        h_unstructured = dp.semiunique_id

        dp.structure()

        # Ensure repeated calls return the same hash (deterministic somewhat)
        h_structured = dp.semiunique_id
        self.assertNotEqual(h_unstructured, h_structured)

        # Changing paths must change the semiunique id
        # As differently datapaths with identical raw but different structuring
        # data saved under different filenames must appear as different
        with ScratchDir("."):
            dp.to_json_file("dp_test_hash.json", omit_raw=True)
            dp = self.BEEPDatapathChildTest.from_json_file("dp_test_hash.json")

            self.assertNotEqual(dp.semiunique_id, h_structured)


class TestBaseEIS(unittest.TestCase):
    class EISChildGood(EIS):
        def from_file(self):
            print("success EISChildGood")

    def setUp(self) -> None:
        self.raw_data = pd.DataFrame({"a": [1,2,3]})
        self.metadata = pd.DataFrame({"example": ["metadata"]})

    def test_BEEPDatapathWithEIS(self):

        # Must implement load_eis and from_file
        class BEEPDatapathWithEISChildTestGood(BEEPDatapathWithEIS):
            def from_file(self, path):
                print(f"{path}")

            def load_eis(self, *arg, **kwargs):
                print("success BEEPDatapathWithEISChildTestGood")

        class BEEPDatapathWithEISChildTestBad(BEEPDatapathWithEIS):
            def extra_method(self):
                pass

        example_metadata = {"example": "metadata"}

        BEEPDatapathWithEISChildTestGood(raw_data=self.raw_data, metadata=example_metadata)

        with self.assertRaises(TypeError):
            BEEPDatapathWithEISChildTestBad(raw_data=self.raw_data, metadata=example_metadata)

    def test_EIS(self):
        # must implement a from_file method
        class EISChildBad(EIS):
            def extra_method(self):
                pass

        eis = self.EISChildGood(data=self.raw_data, metadata=self.metadata)
        self.assertTrue(hasattr(eis, "from_dict"))
        self.assertTrue(hasattr(eis, "as_dict"))

        with self.assertRaises(TypeError):
            EISChildBad()

    def test_EIS_serialization(self):
        eis = self.EISChildGood(data=self.raw_data, metadata=self.metadata)
        d = eis.as_dict()

        eis_test = self.EISChildGood.from_dict(d)
        self.assertTrue(eis.data.equals(eis_test.data))
        self.assertTrue(eis.metadata.equals(eis_test.metadata))


if __name__ == "__main__":
    unittest.main()
