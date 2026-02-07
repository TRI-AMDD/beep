import os

BIG_FILE_TESTS = os.environ.get("BEEP_BIG_TESTS", False) == "True"
SKIP_MSG = "Tests requiring large files with diagnostic cycles are disabled, set BEEP_BIG_TESTS=True to run full tests"
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


def _aws_credentials_available():
    """Check if AWS credentials are configured and valid."""
    try:
        import boto3
        sts = boto3.client('sts')
        sts.get_caller_identity()
        return True
    except Exception:
        return False


AWS_AVAILABLE = _aws_credentials_available()
AWS_SKIP_MSG = "AWS credentials not configured or invalid"
