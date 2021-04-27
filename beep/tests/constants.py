import os

BIG_FILE_TESTS = os.environ.get("BIG_FILE_TESTS", None) == "True"
SKIP_MSG = "Tests requiring large files with diagnostic cycles are disabled, set BIG_FILE_TESTS=True to run full tests"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")