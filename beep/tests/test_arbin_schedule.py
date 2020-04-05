# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Generating Arbin Schedule files"""

import os
import unittest
import difflib
import json
from beep import SCHEDULE_TEMPLATE_DIR
from beep.protocol_tools.arbin_schedule_file import ScheduleFile

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class ArbinScheduleTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_dict_to_file(self):
        sdu = ScheduleFile(version='0.1')
        filename = '20170630-3_6C_9per_5C.sdu'
        testname = 'test1.sdu'
        sdu_dict = sdu.to_dict(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        with open(os.path.join(TEST_FILE_DIR, 'schedule_test.json'), 'w') as file:
            json.dump(sdu_dict, file)
        sdu.dict_to_file(sdu_dict, os.path.join(TEST_FILE_DIR, testname))
        hash1 = sdu.hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        hash2 = sdu.hash_file(os.path.join(TEST_FILE_DIR, testname))
        print(hash1)
        print(hash2)
        if hash1 != hash2:
            original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').readlines()
            parsed = open(os.path.join(TEST_FILE_DIR, testname), encoding='latin-1').readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                print(line)
        os.remove(os.path.join(TEST_FILE_DIR, testname))

    def test_fastcharge(self):
        sdu = ScheduleFile(version='0.1')
        filename = '20170630-3_6C_9per_5C.sdu'
        test_file = 'test.sdu'
        sdu.fast_charge_file(1.1 * 3.6, 0.086, 1.1 * 5, filename, os.path.join(TEST_FILE_DIR, test_file))
        hash1 = sdu.hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        hash2 = sdu.hash_file(os.path.join(TEST_FILE_DIR, test_file))
        if hash1 != hash2:
            original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').readlines()
            parsed = open(os.path.join(TEST_FILE_DIR, test_file), encoding='latin-1').readlines()
            self.assertFalse(list(difflib.unified_diff(original, parsed)))
            for line in difflib.unified_diff(original, parsed):
                print(line)
        os.remove(os.path.join(TEST_FILE_DIR, test_file))


