# Copyright Toyota Research Institute. All rights reserved.
"""Unit tests related to S3 operations"""

import unittest
import os
from beep.utils.s3 import download_s3_object


class S3Test(unittest.TestCase):

    bucket = "beep-sync-test-stage"
    key = "test_util/test_file.txt"
    destination_path = "test_util_s3_file.txt"

    def test_download_s3_object(self):

        download_s3_object(bucket=self.bucket,
                           key=self.key,
                           destination_path=self.destination_path)

        os._exists(self.destination_path)

    def tearDown(self) -> None:
        os.remove(self.destination_path)
