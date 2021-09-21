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

    def setUp(self) -> None:
        self.output_dir = None
        self.status_json_path = None
        self.input_paths = []
        self.runner = CliRunner()

    def tearDown(self) -> None:
        shutil.rmtree(self.outputs_dir)


class TestCLI(TestCLIBase):
    def setUp(self) -> None:
        self.outputs_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLIBase")
        if not os.path.exists(self.outputs_dir):
            os.mkdir(self.outputs_dir)

    def tearDown(self) -> None:
        pass


class TestCLIStructure(TestCLIBase):

    def setUp(self) -> None:
        inputs = [
            "2017-12-04_4_65C-69per_6C_CH29.csv",
            "2017-05-09_test-TC-contact_CH33.csv",  # Fails for not meeting naming convention
            "2017-08-14_8C-5per_3_47C_CH44.csv",
        ]
        self.input_paths = [os.path.join(TEST_FILE_DIR, path) for path in inputs]
        self.outputs_dir = os.path.join(TEST_FILE_DIR, "cmd_TestCLIStructure")
        self.status_json_path = os.path.join(self.outputs_dir, "status-structure.json")

        if not os.path.exists(self.outputs_dir):
            os.mkdir(self.outputs_dir)

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
                self.outputs_dir,
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
                self.outputs_dir,
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
                self.outputs_dir,
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


class TestCLIFeatures(TestCLIBase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_defaults(self):
        pass

    def test_advanced(self):
        pass



class TestCLITrain(TestCLIBase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_defaults(self):
        pass

    def test_advanced(self):
        pass


class TestCLIPredict(TestCLIBase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_defaults(self):
        pass

    def test_advanced(self):
        pass


class TestCLIProtocol(TestCLIBase):
    pass


class TestCLIEndtoEnd(TestCLIBase):
    pass