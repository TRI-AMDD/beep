# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests related to S3 operations"""

import unittest
import os
from beep.utils.s3 import download_s3_object

BIG_FILE_TESTS = os.environ.get("BIG_FILE_TESTS", None) == "True"
SKIP_MSG = "Tests requiring S3 access are disabled, set BIG_FILE_TESTS=True to run full tests"


@unittest.skipUnless(BIG_FILE_TESTS, SKIP_MSG)
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
