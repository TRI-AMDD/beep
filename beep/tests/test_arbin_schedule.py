# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Generating Arbin Schedule files"""

import os
import unittest
import difflib
from monty.serialization import dumpfn
from monty.tempfile import ScratchDir
from beep import SCHEDULE_TEMPLATE_DIR
from beep.protocol_tools.arbin_schedule_file import Schedule
from beep.utils import hash_file

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class ArbinScheduleTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_dict_to_file(self):
        filename = '20170630-3_6C_9per_5C.sdu'
        schedule = Schedule.from_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        testname = 'test1.sdu'
        with ScratchDir('.'):
            dumpfn(schedule, "schedule_test.json")
            schedule.to_file(testname)
            hash1 = hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
            hash2 = hash_file(testname)
            if hash1 != hash2:
                original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').read()
                parsed = open(testname, encoding='latin-1').read()
                self.assertFalse(list(difflib.unified_diff(original, parsed)))
                for line in difflib.unified_diff(original, parsed):
                    print(line)

    def test_fastcharge(self):
        filename = '20170630-3_6C_9per_5C.sdu'
        test_file = 'test.sdu'
        sdu = Schedule.from_fast_charge(1.1 * 3.6, 0.086, 1.1 * 5, os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
        with ScratchDir('.'):
            sdu.to_file(test_file)
            hash1 = hash_file(os.path.join(SCHEDULE_TEMPLATE_DIR, filename))
            hash2 = hash_file(test_file)
            # TODO: these shouldn't be the same, right?
            if hash1 != hash2:
                original = open(os.path.join(SCHEDULE_TEMPLATE_DIR, filename), encoding='latin-1').readlines()
                parsed = open(test_file, encoding='latin-1').readlines()
                udiff = list(difflib.unified_diff(original, parsed))
                for line in udiff:
                    print(line)
                self.assertFalse(udiff)


