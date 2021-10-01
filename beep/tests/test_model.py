import os
import unittest

from monty.serialization import loadfn
from sklearn.linear_model import Lasso

from beep.model import BEEPLinearModelExperiment
from beep.tests.constants import TEST_FILE_DIR

TEST_FILES = os.path.join(TEST_FILE_DIR, "modelling_test_files")


class TestBEEPLinearModelExperiment(unittest.TestCase):

    def setUp(self) -> None:
        features_path = os.path.join(TEST_FILES, "features.json.gz")
        targets_path = os.path.join(TEST_FILES, "targets.json.gz")

        self.features = loadfn(features_path)
        self.targets = loadfn(targets_path)

    def test_core(self):
        """Test all core training, validation, and prediction properties of BLME
        """
        target = "capacity_0.92::TrajectoryFastCharge"

        # Test all linear model types with default settings
        for model_type in BEEPLinearModelExperiment.ALLOWED_MODELS:
            blme = BEEPLinearModelExperiment(
                self.features,
                self.targets,
                [target],
                model_type,
                homogenize_features=True,
                kfold=2,
                max_iter=100
            )
            model, training_errors = blme.train()
            model, training_errors, test_errors = blme.train_and_score(
                train_and_val_frac=0.8)

        # Test multiple targets
        targets = ["capacity_0.92::TrajectoryFastCharge",
                   "capacity_0.8::TrajectoryFastCharge"]

        blme = BEEPLinearModelExperiment(
            self.features,
            self.targets,
            targets,
            "elasticnet",
            homogenize_features=True,
            kfold=2,
            max_iter=100
        )

        model, training_errors = blme.train()
        model, training_errors, test_errors = blme.train_and_score(
            train_and_val_frac=0.8)

        for error_dict in (training_errors, test_errors):
            for metric in BEEPLinearModelExperiment.ERROR_METRICS.keys():
                self.assertIn(metric, error_dict)
                for target in targets:
                    self.assertIn(target, error_dict[metric])

        # Just predict on the training features for now
        y_pred, dropped = blme.predict(self.features, homogenize_features=True)
        self.assertTupleEqual(y_pred.shape, (4, 2))
        self.assertFalse(dropped)

    def test_serialization(self):
        json_filename = os.path.join(TEST_FILES, "model.json.gz")
        target = "capacity_0.92::TrajectoryFastCharge"
        blme = BEEPLinearModelExperiment(
            self.features,
            self.targets,
            [target],
            "lasso",
            homogenize_features=True,
            kfold=2,
            max_iter=100
        )

        blme.train()

        blme.to_json_file(json_filename)

        blme_reloaded = loadfn(json_filename)
        self.assertIsInstance(blme_reloaded.model, Lasso)
        predictions, dropped = blme_reloaded.predict(self.features)
        self.assertEqual(predictions.shape, (4, 1))
        self.assertFalse(dropped)
