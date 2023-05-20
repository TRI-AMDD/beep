"""
Tests for classes representing steps of a battery cycle.
"""

import os
import unittest

import pandas as pd

from beep.structure.core.step import Step, MultiStep
from beep.structure.core.tests import DIR_TESTS_STRUCTURE_CORE


class TestStep(unittest.TestCase):
    """
    Tests for the Step class, representing one step of a battery cycle.
    """
    def setUp(self):
        kw = {"index_col": 0}
        self.df_srr_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_regular_charge.csv"), **kw)
        self.df_srr_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_regular_discharge.csv"), **kw)
        self.df_srr_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_regular_unknown.csv"), **kw)
        self.df_srd_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_diagnostic_charge.csv"), **kw)
        self.df_srd_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_diagnostic_discharge.csv"), **kw)
        self.df_srd_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_raw_diagnostic_unknown.csv"), **kw)
        self.df_ss_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_structured_charge.csv"), **kw)
        self.df_ss_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_structured_discharge.csv"), **kw)
        self.df_ss_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "step_structured_unknown.csv"), **kw)

        self.all_step_dfs = [
            self.df_srr_chg,
            self.df_srr_dchg,
            self.df_srr_rest,
            self.df_srd_chg,
            self.df_srd_dchg,
            self.df_srd_rest,
            self.df_ss_chg,
            self.df_ss_dchg,
            self.df_ss_rest
        ]

        self.df_mr_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_raw_charge.csv"), **kw)
        self.df_mr_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_raw_discharge.csv"), **kw)
        self.df_mr_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_raw_unknown.csv"), **kw)
        self.df_ms_chg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_structured_charge.csv"), **kw)
        self.df_ms_dchg = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_structured_discharge.csv"), **kw)
        self.df_ms_rest = pd.read_csv(os.path.join(DIR_TESTS_STRUCTURE_CORE, "multistep_structured_unknown.csv"), **kw)

        self.all_multistep_dfs = [
            self.df_mr_chg,
            self.df_mr_dchg,
            self.df_mr_rest,
            self.df_ms_chg,
            self.df_ms_dchg,
            self.df_ms_rest
        ]

        self.minimal_df = pd.DataFrame({
            "step_counter": [1]*10,
            "step_counter_absolute": [6]*10,
            "step_code": [3]*10,
            "step_label": ["chg"]*10,
            "cycle_index": [1]*10,
            "cycle_label": ["reg"]*10
        })

        self.test_class = Step

    def test_instantiation(self):
        for df in self.all_step_dfs:
            s = Step(df)
            # nans cannot be compared via numpy

            # Test data is not maligned by Step
            df.fillna(0, inplace=True)
            s.data.fillna(0, inplace=True)
            self.assertTrue(df.eq(s.data).all().all())

            # Ensure that even though existing columns are
            # equal, there are none added or removed
            true_cols = tuple(df.columns.tolist())
            test_cols = tuple(s.data.columns.tolist())
            self.assertTupleEqual(true_cols, test_cols)

        # Test minimal dataframe
        s = Step(self.minimal_df)
        self.assertTrue(self.minimal_df.eq(s.data).all().all())

        # Test that minimal dataframe is actually minimal
        # Ie., it should fail anything less than it
        for col in self.minimal_df.columns:
            df2 = self.minimal_df.copy()
            df2.drop(columns=[col], inplace=True)
            with self.assertRaises(KeyError):
                Step(df2)

    def test_getattr(self):
        # To ensure the "unique" attributes based on
        # data actually match the data.
        for df in self.all_step_dfs + [self.minimal_df]:
            s = Step(df)
            for attr in Step.uniques:
                a = getattr(s, attr)
                self.assertTrue(s.data[attr].eq(a).all())

            # Now update the data held by the step
            # And see if the attribute is also updated
            s.data["step_counter"] = 100

            for attr in Step.uniques:
                a = getattr(s, attr)
                self.assertTrue(s.data[attr].eq(a).all())

    def test_uniqueness(self):
        for multistep_df in self.all_multistep_dfs:
            with self.assertRaises(ValueError):
                Step(multistep_df)

        # And a manual test for each column
        for col in Step.uniques:
            df2 = self.minimal_df.copy()
            df2[col] = df2[col].iloc[:-1].tolist() + [100]
            with self.assertRaises(ValueError):
                Step(df2)

    def test_config(self):
        # Ensure there is no weird behavior going on
        # due to getattr/setattr
        s = self.test_class(self.minimal_df)
        self.assertIsInstance(s.config, dict)

        s.config = {"some": "option"}
        self.assertIsInstance(s.config, dict)

    def test_serialization(self):
        for df in self.all_step_dfs + [self.minimal_df]:
            s = self.test_class(df.copy())
            s.config = {"some": "option", "other": 2}
            d = s.as_dict()
            self.assertIsInstance(d, dict)
            self.assertIn("data", d)
            self.assertIn("config", d)

            s2 = self.test_class.from_dict(d)
            self.assertIsInstance(s2, self.test_class)

            # Make sure the data is the same
            # sans the index
            for step in (s, s2):
                step.data.fillna(0, inplace=True)

            s.data.reset_index(inplace=True, drop=True)
            s2.data.reset_index(inplace=True, drop=True)
            self.assertTrue(s.data.eq(s2.data).all().all())
            self.assertDictEqual(s.config, s2.config)


class TestMultiStep(TestStep):
    """
    Steps for the MultiStep class.

    For representing multiple steps as one within a single cycle
    (typically used for convenience with structuring).

    Note this class inherits TestStep, so some tests from
    TestStep are run automatically based on self.test_class.
    """
    def setUp(self):
        super().setUp()
        self.minimal_multistep_df = pd.DataFrame({
            "step_counter": [1]*5 + [2]*5,
            "step_counter_absolute": [6]*5 + [7]*5,
            "step_code": [3]*3 + [4]*7,
            "step_label": ["chg"]*10,
            "cycle_index": [1]*10,
            "cycle_label": ["reg"]*10
        })
        self.all_dfs = self.all_step_dfs + \
            self.all_multistep_dfs + \
            [self.minimal_df, self.minimal_multistep_df]

        self.test_class = MultiStep

    def test_instantiation(self):
        # Should work for all dataframes
        for df in self.all_dfs:
            ms = MultiStep(df)
            # nans cannot be compared via numpy
            # Test data is not maligned by MultiStep
            df.fillna(0, inplace=True)
            ms.data.fillna(0, inplace=True)
            self.assertTrue(df.eq(ms.data).all().all())

            # Ensure that even though existing columns are
            # equal, there are none added or removed
            true_cols = tuple(df.columns.tolist())
            test_cols = tuple(ms.data.columns.tolist())
            self.assertTupleEqual(true_cols, test_cols)

    def test_uniqueness(self):
        # Should only fail on mandatory uniques
        # And a manual test for each column
        for col in MultiStep.mandatory_uniques:
            df2 = self.minimal_multistep_df.copy()
            df2[col] = df2[col].iloc[:-1].tolist() + [100]
            with self.assertRaises(ValueError):
                Step(df2)

    def test_getattr(self):
        # To ensure the "unique" attributes based on
        # data actually match the data.
        for df in self.all_dfs:
            ms = MultiStep(df)
            for attr in MultiStep.mandatory_uniques:
                a = getattr(ms, attr)
                self.assertTrue(ms.data[attr].eq(a).all())

        # Try a manual example
        ms = MultiStep(self.minimal_multistep_df)
        self.assertListEqual(ms.step_counter, [1, 2])
        self.assertListEqual(ms.step_counter_absolute, [6, 7])
        self.assertEqual(ms.step_label, "chg")
        self.assertEqual(ms.cycle_label, "reg")

    # test_serialization and test_config are done
    # by parent class by setting test_class=MultiStep in setUp
