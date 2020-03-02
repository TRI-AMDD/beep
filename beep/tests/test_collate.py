#  Copyright (c) 2019 Toyota Research Institute

import unittest
import os

from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from pathlib import Path
from beep.collate import get_parameters_fastcharge, get_parameters_oed, process_files_json

TEST_DIR = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(TEST_DIR, "test_files")


class CollateTest(unittest.TestCase):
    def test_get_parameters_fastcharge(self):
        # Force a weird filename to be parsed
        source_directory = os.path.join(TEST_FILE_DIR)
        date, chno, strname, protocol = get_parameters_fastcharge('2019-03-06-per_CH1234', source_directory)
        self.assertEqual(date, "2019-03-06")
        self.assertEqual(chno, "CH1234")
        self.assertEqual(protocol, None)

        # Proper filename
        date, chno, strname, protocol = get_parameters_fastcharge("2017-12-04_4_65C-69per_6C_CH29.csv", source_directory)
        self.assertEqual(date, "2017-12-04")
        self.assertEqual(chno, "CH29")
        self.assertEqual(protocol, "4.65C(69%)-6C")

    def test_get_parameters_oed(self):
        # Proper filename
        source_directory = os.path.join(TEST_FILE_DIR)
        date, chno, strname, protocol = get_parameters_oed("2018-08-28_oed_0_CH1.csv", source_directory)
        print(date, chno, strname, protocol)
        self.assertEqual(date, "2018-08-28")
        self.assertEqual(chno, "CH1")
        self.assertEqual(protocol, '{"cc1": "5.6", "cc2": "6", "cc3": "4.8", "cc4": "3.574"}')

    @unittest.skipUnless(True, "toggle this test")
    def test_all_filenames(self):
        """Test to see if renaming works on all filenames"""
        files = loadfn(os.path.join(TEST_FILE_DIR, 'test_filenames.json'))
        source_directory = os.path.join(TEST_FILE_DIR)
        for filename in files:
            params = get_parameters_fastcharge(filename, source_directory)

        with ScratchDir('.'):
            os.environ["BEEP_ROOT"] = os.getcwd()
            os.mkdir("data-share")
            os.mkdir(os.path.join("data-share", "raw_cycler_files"))
            os.mkdir(os.path.join("data-share", "renamed_cycler_files"))
            #filter for only files with protocol in name
            files = [file for file in files if "batch8" not in file]
            for filename in files:
                Path(os.path.join("data-share", "raw_cycler_files", filename)).touch()
            process_files_json()
        pass  # to exit scratch dir context
