# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Unit tests related to Splicing files"""

import os
import unittest
import numpy as np
from beep.utils import MaccorSplice

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class SpliceTest(unittest.TestCase):
    def setUp(self):
        self.arbin_file = os.path.join(TEST_FILE_DIR, "FastCharge_000000_CH29.csv")
        self.filename_part_1 = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038.078")
        self.filename_part_2 = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038con.078")
        self.output = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038joined.078")
        self.test = os.path.join(TEST_FILE_DIR, "xTESLADIAG_000038test.078")

    def test_maccor_read_write(self):
        splicer = MaccorSplice(self.filename_part_1, self.filename_part_2, self.output)

        meta_1, data_1 = splicer.read_maccor_file(self.filename_part_1)
        splicer.write_maccor_file(meta_1, data_1, self.test)
        meta_test, data_test = splicer.read_maccor_file(self.test)

        assert meta_1 == meta_test
        assert np.allclose(data_1['Volts'].to_numpy(), data_test['Volts'].to_numpy())
        assert np.allclose(data_1['Amps'].to_numpy(), data_test['Amps'].to_numpy())
        assert np.allclose(data_1['Test (Sec)'].to_numpy(), data_test['Test (Sec)'].to_numpy())

    def test_column_increment(self):
        splicer = MaccorSplice(self.filename_part_1, self.filename_part_2, self.output)
        meta_1, data_1 = splicer.read_maccor_file(self.filename_part_1)
        meta_2, data_2 = splicer.read_maccor_file(self.filename_part_2)
        data_1, data_2 = splicer.column_increment(data_1, data_2)

        assert data_1['Rec#'].max() < data_2['Rec#'].min()
