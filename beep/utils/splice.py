#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""Script for joining together two Maccor files that correspond to the same run
but data has been split between two files. The column increment function makes sure
that certain columns are monotonically increasing across the two files. The metadata
line from the first file is used in the final output file

Usage:
    splice.py [options]
    splice.py (-h | --help)

Options:
    -h --help                Show this screen
    --version                Show version

"""
import pandas as pd
from beep import StringIO
from beep import LOG_DIR
import os


class MaccorSplice:
    def __init__(self, input_1, input_2, output):
        """
        Args:
            input_1 (str): Filename corresponding to first file.
            input_2 (str): Filename corresponding to second file.
            output (str): Filename of output file.
        """
        self.input_1 = input_1
        self.input_2 = input_2
        self.output = output

    def read_maccor_file(self, filename):
        """
        Loads Maccor file and returns meta data line and data frame containing
        file data.

        Args:
            filename (str): path to file.

        Returns:
            str: first line of maccor file containing meta data.
            pandas.DataFrame: data frame with file data.

        """
        with open(filename) as f:
            lines = f.readlines()
        metadata_line = lines[0]
        data_lines = lines[1:]

        # Parse data
        data_text = '\n'.join(data_lines)
        tsv_part = pd.read_csv(StringIO(data_text), delimiter="\t")
        return metadata_line, tsv_part

    def write_maccor_file(self, metadata_line, dataframe, output):
        """
        Writes data and meta data into a Maccor file.

        Args:
            metadata_line (str): line containing meta data.
            dataframe (pandas.DataFrame): content data.
            output (str): output file name.
            output (str): output file name.
        """
        with open(output, 'w') as write_tsv:
            write_tsv.writelines(metadata_line)
            write_tsv.write(dataframe.to_csv(sep='\t', index=False))

    def column_increment(self, data_1, data_2):
        """
        Special increment logic.

        Args:
            data_1 (pandas.DataFrame):
            data_2 (pandas.DataFrame):

        Returns:
            pandas.DataFrame: data_1 transformed (incremented)
            pandas.DataFrame: data_2 transformed (incremented)
        """
        columns_to_update = ['Rec#', 'Cyc#', 'Test (Sec)', 'Loop1', 'Loop2', 'Loop3', 'Loop4']
        for column in columns_to_update:
            if data_2[column].iloc[0] < data_1[column].iloc[-1]:
                data_2[column] = data_2[column] + data_1[column].iloc[-1]

        return data_1, data_2

    def splice_operation(self, data_1, data_2):
        """
        Concatenates two data frames.

        Args:
            data_1 (pandas.DataFrame):
            data_2 (pandas.DataFrame):

        Returns:
            pandas.DataFrame: concatenated data frame.

        """
        data_final = pd.concat([data_1, data_2])
        return data_final

    def run_splice(self):
        """
        Reads two input maccor files. Concatenates the respective data frames.
        Writes to a new Maccor file.
        """

        metadata_line_1, data_1 = self.read_maccor_file(self.input_1)
        metadata_line_2, data_2 = self.read_maccor_file(self.input_2)
        data_1, data_2 = self.column_increment(data_1, data_2)
        data_final = self.splice_operation(data_1, data_2)
        self.write_maccor_file(metadata_line_1, data_final, self.output)
