import os
import unittest

import pandas as pd
from monty.serialization import loadfn, dumpfn
from monty.tempfile import ScratchDir

from beep.structure.diagnostic import DiagnosticConfig
from beep.tests.constants import TEST_FILE_DIR


class TestDiagnosticConfig(unittest.TestCase):
    def test_DiagnosticConfig(self):
        # empty test case
        with self.assertRaises(ValueError):
            DiagnosticConfig({})

        # only one type of rpt cycle
        rpt_ix = {1, 2, 3}
        dc = DiagnosticConfig(
            diagnostic_config={
                "rpt": rpt_ix
            }
        )
        self.assertSetEqual(dc.rpt_ix, rpt_ix)
        self.assertSetEqual(dc.all_ix, rpt_ix)
        self.assertSetEqual(dc.hppc_ix, set())
        self.assertSetEqual(dc.reset_ix, set())

        # only one type of hppc_cycle
        hppc_ix = {12, 14, 1}
        dc = DiagnosticConfig(
            diagnostic_config={
                "hppc": hppc_ix
            }
        )
        self.assertSetEqual(dc.hppc_ix, hppc_ix)
        self.assertSetEqual(dc.all_ix, hppc_ix)
        self.assertSetEqual(dc.reset_ix, set())
        self.assertSetEqual(dc.rpt_ix, set())

        # test multiple labels for many cycle types
        rpt1_ix = set(range(1, 1002, 200))
        rpt2_ix = set(range(2, 1003, 200))
        hppc1_ix = {12, 512}
        hppc2_ix = {115, 718, 910}
        reset_ix = {0, 1000}
        abnormal_ix = set(range(5, 520, 50))

        dc = DiagnosticConfig(
            {
                "rpt1": rpt1_ix,
                "rpt2": rpt2_ix,
                "hppc1": hppc1_ix,
                "hppc2": hppc2_ix,
                "reset_": reset_ix,
                "abnormal": abnormal_ix
            },
        )
        # Test access via user-provided cycle type names
        self.assertSetEqual(dc.cycles["rpt1"], rpt1_ix)
        self.assertSetEqual(dc.cycles["rpt2"], rpt2_ix)
        self.assertSetEqual(dc.cycles["hppc1"], hppc1_ix)
        self.assertSetEqual(dc.cycles["hppc2"], hppc2_ix)
        self.assertSetEqual(dc.cycles["reset_"], reset_ix)
        self.assertSetEqual(dc.cycles["abnormal"], abnormal_ix)

        # Test access via std. type names
        all_rpt_ix = rpt1_ix.union(rpt2_ix)
        all_hppc_ix = hppc1_ix.union(hppc2_ix)
        all_diag_ix = set().union(all_rpt_ix, all_hppc_ix, reset_ix,
                                  abnormal_ix)
        self.assertSetEqual(dc.rpt_ix, all_rpt_ix)
        self.assertSetEqual(dc.hppc_ix, all_hppc_ix)
        self.assertSetEqual(dc.all_ix, all_diag_ix)
        self.assertSetEqual(dc.reset_ix, reset_ix)

        # Test access by cycle index
        self.assertEqual(dc.type_by_ix[2], "rpt2")
        self.assertEqual(dc.type_by_ix[1000], "reset_")
        self.assertEqual(dc.type_by_ix[355], "abnormal")

        # test error of overlapping cycle types on a single cycle index
        rpt_ix = {1, 2, 3}
        hppc_ix = {0, 101, 3, 1999}

        with self.assertRaises(ValueError):
            DiagnosticConfig(
                {
                    "rpt": rpt_ix,
                    "hppc": hppc_ix
                }
            )

    def test_kwargs(self):
        dc = DiagnosticConfig(
            {
                "rpt": (1, 2, 12)
            },
            kw1=True,
            kw2=43,
            kw3="something"
        )

        self.assertTrue(dc.params["kw1"])
        self.assertEqual(dc.params["kw2"], 43)
        self.assertEqual(dc.params["kw3"], "something")

        with self.assertRaises(TypeError):
            DiagnosticConfig(
                {
                    "rpt": (1, 2, 12)
                },
                kw1=True,
                kw2=[1, 15, 92]
            )

    def test_serialization(self):
        rpt_ix = {1, 2, 3}
        hppc_ix = {0, 101, 1999}

        dc = DiagnosticConfig(
            {
                "rpt": rpt_ix,
                "hppc": hppc_ix
            },
            parameter_set="SomeMadeUp_Paramset",
            extra_var=123
        )

        d = dc.as_dict()
        dc2 = DiagnosticConfig.from_dict(d)
        self.assertSetEqual(dc2.rpt_ix, dc.rpt_ix)
        self.assertSetEqual(dc2.hppc_ix, dc.hppc_ix)
        self.assertSetEqual(dc2.all_ix, dc.all_ix)
        self.assertEqual(dc2.params["parameter_set"], "SomeMadeUp_Paramset")
        self.assertEqual(dc2.params["extra_var"], 123)

        with ScratchDir("."):
            fname = "test_serialization_DiagnosticConfig.json"
            dumpfn(dc, fname)

            dc3 = loadfn(fname)
            self.assertSetEqual(dc3.rpt_ix, dc.rpt_ix)
            self.assertSetEqual(dc3.hppc_ix, dc.hppc_ix)
            self.assertSetEqual(dc3.all_ix, dc.all_ix)
            self.assertEqual(dc3.params["parameter_set"], "SomeMadeUp_Paramset")
            self.assertEqual(dc3.params["extra_var"], 123)

    def test_from_step_numbers(self):
        # read data written directly to CSV to avoid any future conflicts with
        # ingestion or changes in .from_file in BEEPDatapath
        fpath = os.path.join(TEST_FILE_DIR, "Nova_Regular_115_df_raw.csv")
        df_memsaved = pd.read_csv(fpath, index_col=0)

        dc = DiagnosticConfig.from_step_numbers(
            df_memsaved,
            matching_criteria={
                "hppc": ("contains", [(1, 2, 4, 6, 8)]),
                "rpt_lowrate": ("exact", [(12, 13)]),
                "rpt_highrate": ("exact", [(15, 16)])
            }
        )
        self.assertSetEqual(dc.cycle_type_to_cycle_ix["rpt_lowrate"],
                            {1, 25, 128})
        self.assertSetEqual(dc.cycle_type_to_cycle_ix["rpt_highrate"], {2, 26})
        self.assertSetEqual(dc.hppc_ix, {0, 8, 24, 127})
        self.assertSetEqual(dc.rpt_ix, {1, 2, 25, 26, 128})
