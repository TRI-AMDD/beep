import os
import unittest

import pandas as pd

from beep.structure.cli import auto_load_processed
from beep.features.base import BEEPFeaturizer, BEEPFeatureMatrix, BEEPFeaturizationError, BEEPFeatureMatrixError
from beep.tests.constants import TEST_FILE_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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


        # with incomplete hyperparams
        hps = {
            "hyperparam_A": 11
        }
        with self.assertRaises(BEEPFeaturizationError):
            self.ExampleBEEPFeaturizer(self.dp, hyperparameters=hps)


    def test_serialization(self):
        f = self.ExampleBEEPFeaturizer(self.dp)
        with self.assertRaises(BEEPFeaturizationError):
            f.as_dict()

        f.create_features()
        d = f.as_dict()
        self.assertEqual(d["features"]["f1"][0], 12)
        self.assertEqual(d["hyperparameters"]["hyperparam_A"], 12)
        self.assertEqual(d["metadata"]["barcode"], "00004C")
        self.assertTrue("paths" in d)

        output_path = os.path.join(THIS_DIR, "BEEPFeaturizer_test_serialization.json.gz")
        f.to_json_file(output_path)
        f_reloaded = f.from_json_file(output_path)
        self.assertIsInstance(f_reloaded.features, pd.DataFrame)
        self.assertEqual(f_reloaded.hyperparameters["hyperparam_B"], 13)
        self.assertEqual(f_reloaded.metadata["channel_id"], 33)

