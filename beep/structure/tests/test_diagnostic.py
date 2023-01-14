import os
import unittest

from beep.structure.diagnostic import DiagnosticConfig
from beep.tests.constants import TEST_FILE_DIR


class TestDiagnosticConfig(unittest.TestCase):
    def test_DiagnosticConfig(self):
        # empty test case
        with self.assertRaises(ValueError):
            DiagnosticConfig({})

        # only one type of rpt cycle
        rpt_ix = {1, 2, 3}
        dc  = DiagnosticConfig(
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
        all_diag_ix = set().union(all_rpt_ix, all_hppc_ix, reset_ix, abnormal_ix)
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

    def test_serialization(self):
        rpt_ix = {1, 2, 3}
        hppc_ix = {0, 101, 1999}

        dc = DiagnosticConfig(
            {
                "rpt": rpt_ix,
                "hppc": hppc_ix
            }
        )

        d = dc.as_dict()
        dc2 = DiagnosticConfig.from_dict(d)
        self.assertSetEqual(dc2.rpt_ix, dc.rpt_ix)
        self.assertSetEqual(dc2.hppc_ix, dc.hppc_ix)
        self.assertSetEqual(dc2.all_ix, dc.all_ix)

