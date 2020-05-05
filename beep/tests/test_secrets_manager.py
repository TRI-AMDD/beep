# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Splicing files"""

import os
import unittest
from beep import ENVIRONMENT
from beep.config import config
from beep.utils.secrets_manager import secret_accessible, get_secret

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class SecretTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_secret_accessible(self):
        available = secret_accessible(ENVIRONMENT)
        if available:
            secret_name = config[ENVIRONMENT]['kinesis']['stream']
            get_secret(secret_name)
        else:
            self.assertFalse(available)
