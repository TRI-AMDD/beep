import os
import unittest

import pandas as pd

from beep.structure.cli import auto_load_processed
from beep.features.base import BEEPFeaturizer, BEEPFeatureMatrix
from beep.tests.constants import TEST_FILE_DIR


class TestBEEPFeaturizer(unittest.TestCase):
    class ExampleBEEPFeaturizer(BEEPFeaturizer):

        DEFAULT_HYPERPARAMETERS = {
            "hyperparam_A": 12,
            "hyperparam_B": 13
        }

        def validate(self):
            if self.datapath.diagnostic_summary is not None:
                return True, None
            else:
                return False, "no diagnostic info"

        def create_features(self) -> None:
            self.features = pd.DataFrame(
                {
                    "f1": [1 * self.hyperparameters["hyperparam_A"]],
                    "f2": [2 * self.hyperparameters["hyperparam_B"]],
                    "f3": [3]
                }
            )

    class ExampleBEEPFeaturizerBad(BEEPFeaturizer):
        pass

    @classmethod
    def setUpClass(cls) -> None:
        cls.dp = auto_load_processed(
            os.path.join(TEST_FILE_DIR,
                         "PredictionDiagnostics_000132_00004C_structure.json")
        )

    def test_core_ops(self):

        # with hyperparameters not specified
        f = self.ExampleBEEPFeaturizer(self.dp, hyperparameters=None)
        val, msg = f.validate()
        self.assertTrue(val)
        self.assertIsNone(msg)

        f.create_features()
        X = f.features
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTupleEqual(X.shape, (1, 3))
        self.assertEqual(X["f1"].iloc[0], 12)
        self.assertEqual(X["f2"].iloc[0], 26)
        self.assertEqual(X["f3"].iloc[0], 3)


        # with hyperparameters specified
        hps = {
            "hyperparam_A": 20,
            "hyperparam_B": 3

        }
        f = self.ExampleBEEPFeaturizer(self.dp, hyperparameters=hps)
        val, msg = f.validate()
        self.assertTrue(val)
        self.assertIsNone(msg)

        f.create_features()
        X = f.features
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTupleEqual(X.shape, (1, 3))
        self.assertEqual(X["f1"].iloc[0], 20)
        self.assertEqual(X["f2"].iloc[0], 6)
        self.assertEqual(X["f3"].iloc[0], 3)


    def test_serialization(self):

        f = self.ExampleBEEPFeaturizer(self.dp)
        d = f.as_dict()

        import pprint
        pprint.pprint(d)
