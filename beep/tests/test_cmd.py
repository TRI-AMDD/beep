"""
Tests for the beep CLI.
"""
import os
import unittest
import shutil

from monty.serialization import loadfn, dumpfn
from click.testing import CliRunner

from beep import PROTOCOL_PARAMETERS_DIR
from beep.cmd import cli
from beep.tests.constants import TEST_FILE_DIR, SKIP_MSG, BIG_FILE_TESTS


class TestCLIBase(unittest.TestCase):
    runner = CliRunner()

    def setUp(self) -> None:
        self.output_dir = None
        self.status_json_path = None
        self.input_paths = []
        self.runner = None

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir)


class TestCLI(TestCLIBase):
    def setUp(self) -> None:
        self.output_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLIBase")
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


class TestCLIStructure(TestCLIBase):

    def setUp(self) -> None:
        inputs = [
            "2017-12-04_4_65C-69per_6C_CH29.csv",
            "2017-05-09_test-TC-contact_CH33.csv",  # Fails for not meeting naming convention
            "2017-08-14_8C-5per_3_47C_CH44.csv",
        ]
        self.input_paths = [os.path.join(TEST_FILE_DIR, path) for path in inputs]
        self.output_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLIStructure")
        self.status_json_path = os.path.join(self.output_dir, "status-structure.json")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def test_defaults(self):
        """Test the default structuring configuration with the CLI.
        """

        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "structure",
                "--output-dir",
                self.output_dir,
                *self.input_paths
            ]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)

        status = loadfn(self.status_json_path)

        self.assertTrue(status["files"][self.input_paths[0]]["validated"])
        self.assertFalse(status["files"][self.input_paths[1]]["validated"])
        self.assertFalse(status["files"][self.input_paths[2]]["validated"])
        self.assertTrue(status["files"][self.input_paths[0]]["structured"])
        self.assertFalse(status["files"][self.input_paths[1]]["structured"])
        self.assertFalse(status["files"][self.input_paths[2]]["structured"])

        self.assertTrue(os.path.exists(status["files"][self.input_paths[0]]["output"]))

    def test_advanced(self):
        """Test the structuring CLI with some options specified"""
        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "structure",
                "--output-dir",
                self.output_dir,
                "--automatic",
                "--protocol-parameters-dir",
                PROTOCOL_PARAMETERS_DIR,
                "--no-raw",
                *self.input_paths
            ]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)
        status = loadfn(self.status_json_path)

        self.assertTrue(status["files"][self.input_paths[0]]["structured"])
        self.assertTrue(status["files"][self.input_paths[0]]["validated"])
        self.assertEqual(status["files"][self.input_paths[0]]["structuring_parameters"]["diagnostic_resolution"], 500)

        self.assertTrue(os.path.exists(status["files"][self.input_paths[0]]["output"]))

    # @unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
    def test_s3(self):
        """Test the structuring using files from S3"""
        s3_key = "big_file_tests/PreDiag_000287_000128.092"

        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "structure",
                "--output-dir",
                self.output_dir,
                "--automatic",
                "--protocol-parameters-dir",
                PROTOCOL_PARAMETERS_DIR,
                "--no-raw",
                "--s3-bucket",
                "beep-sync-test-stage",
                "--s3-use-cache",
                s3_key,
            ]
        )
        print(result.output)
        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)


class TestCLIFeaturize(TestCLIBase):

    def setUp(self) -> None:
        inputs = [
            "PreDiag_000440_0000FB_structure.json",
            "PredictionDiagnostics_000132_00004C_structure.json"
        ]
        self.input_paths = [os.path.join(TEST_FILE_DIR, path) for path in inputs]

        self.output_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLIFeaturize")
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.status_json_path = os.path.join(self.output_dir, "status-featurize.json")

    def test_defaults(self):
        """Test a very basic CLI featurization."""
        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "featurize",
                "--output-dir",
                self.output_dir,
                "--featurize-with",
                "all_features",
                *self.input_paths
            ]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)

        status = loadfn(self.status_json_path)

        self.assertTrue(status["feature_matrix"]["created"])
        self.assertTrue(os.path.exists(status["feature_matrix"]["output"]))

        for fname in self.input_paths:
            self.assertTrue(status["files"][fname]["featurizers"][-1]["featurized"])
            self.assertTrue(status["files"][fname]["featurizers"][-1]["valid"])
            self.assertIsNone(status["files"][fname]["featurizers"][-1]["output"])

    def test_advanced(self):
        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "featurize",
                "--output-dir",
                self.output_dir,
                "--featurize-with",
                "HPPCResistanceVoltageFeatures",
                "--featurize-with",
                "CycleSummaryStats",
                "--featurize-with-hyperparams",
                '{"CycleSummaryStats": {"cycle_comp_num": [11, 101]}}',
                "--save-intermediates",
                *self.input_paths
            ]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)

        status = loadfn(self.status_json_path)

        self.assertTrue(status["feature_matrix"]["created"])
        self.assertTrue(os.path.exists(status["feature_matrix"]["output"]))

        for f, data in status["files"].items():
            for fresult in data["featurizers"]:
                # check intermediate files output
                self.assertTrue(os.path.exists(fresult["output"]))

        self.assertEqual(
            status["files"][self.input_paths[0]]["featurizers"][-1]["hyperparameters"]["cycle_comp_num"][0],
            11
        )


class TestCLITrain(TestCLIBase):

    def setUp(self) -> None:
        input_paths = [
            "features.json.gz",
            "targets.json.gz"
        ]
        self.input_paths = [os.path.join(TEST_FILE_DIR, "modelling_test_files", p) for p in input_paths]
        self.features_file = self.input_paths[0]
        self.targets_file = self.input_paths[1]

        self.output_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLITrain")
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.output_filename = os.path.join(self.output_dir, "model.json.gz")
        self.status_json_path = os.path.join(self.output_dir, "status-train.json")

    def test_basic(self):
        # Training only
        result = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "train",
                "--output-filename",
                self.output_filename,
                "--feature-matrix-file",
                self.features_file,
                "--target-matrix-file",
                self.targets_file,
                "--targets",
                "capacity_0.92::TrajectoryFastCharge",
                "--model-name",
                "elasticnet",
                "--kfold",
                2,
                "--max-iter",
                200
            ]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIsNotNone(result.output)
        status = loadfn(self.status_json_path)
        self.assertTrue(status["trained_model"]["created"])
        self.assertTrue(os.path.exists(status["trained_model"]["output"]))

        # Training and model results from validation set
        result2 = self.runner.invoke(
            cli,
            [
                "--output-status-json",
                self.status_json_path,
                "train",
                "--output-filename",
                self.output_filename,
                "--feature-matrix-file",
                self.features_file,
                "--target-matrix-file",
                self.targets_file,
                "--targets",
                "capacity_0.92::TrajectoryFastCharge",
                "--model-name",
                "elasticnet",
                "--kfold",
                2,
                "--train-on-frac-and-score",
                0.8,
                "--max-iter",
                200
            ]
        )
        self.assertEqual(result2.exit_code, 0)
        self.assertIsNotNone(result2.output)

        status = loadfn(self.status_json_path)

        self.assertTrue(status["trained_model"]["created"])
        self.assertTrue(os.path.exists(status["trained_model"]["output"]))
        self.assertAlmostEqual(status["model_results"]["test_fraction"], 0.8, 2)
        self.assertIn("test_error", status["model_results"])


class TestCLIPredict(TestCLIBase):
    def setUp(self) -> None:
        self.model_file = os.path.join()

    def test_defaults(self):
        pass

    def test_advanced(self):
        pass


class TestCLIProtocol(TestCLIBase):
    def setUp(self) -> None:
        pass

    def test_defaults(self):
        pass

    def test_advanced(self):
        pass


class TestCLIEndtoEnd(TestCLIBase):
    def setUp(self) -> None:
        pass

    def test_end_to_end(self):
        pass