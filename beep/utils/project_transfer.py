#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""This script is a small utility to rename files and effectively move them from one project to another.

Usage:
    project_transfer.py [options]
    project_transfer.py (-h | --help)

Options:
    -h --help                Show this screen
    --version                Show version

"""

from botocore.exceptions import ClientError
import os
import shutil
import boto3
from monty.tempfile import ScratchDir


class ProjectTransfer:
    def __init__(self,
                 input_project,
                 output_project,
                 bucket,
                 prefix,
                 dry_run=True):
        """
        Args:
            input_project (str): Name corresponding to the project to get the files from.
            output_project (str): Name of the project to transfer the files to.
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix of the input and output project in the S3 bucket.
            mode (str): mode to run in, if 'test' the output bucket is the 'beep-sync-test' bucket.
        """
        self.input_project = input_project
        self.output_project = output_project
        self.bucket = bucket
        self.prefix = prefix
        self.dry_run = dry_run

    def get_list_files(self, excluded_string):
        all_objects = self.get_all_objects(self.bucket, self.prefix)
        object_names = [obj['Key'] for obj in all_objects
                        if os.path.join(self.prefix, self.input_project) in obj['Key']]
        for string in excluded_string:
            object_names = [obj for obj in object_names if string not in obj]
        return object_names

    def get_all_objects(self, bucket, prefix):
        """
        Helper function to get common "subfolders" of folders in S3.

        Args:
            bucket (str): bucket name.
            prefix (str): prefix for which to list common prefixes.
        """
        object_names = []
        if not prefix.endswith('/'):
            prefix += "/"
        client = boto3.client('s3')
        paginator = client.get_paginator('list_objects_v2')
        result = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix)
        for response in result:
            object_names = object_names + response['Contents']
        return object_names

    def download_and_modify(self, object_names):
        """
        Downloads all S3 objects that are in the object_names list, runs a find and replace on the
        metadata in the first line of the file to change the project name from the input project name
        to the output project name.

        Args:
            object_names (list): List of object names in S3 that should be transferred to the output project.

        """
        with ScratchDir('.') as scratch_dir:
            for object_name in object_names:
                directory, name = os.path.split(object_name)
                s3 = boto3.client("s3")
                s3.download_file(self.bucket, object_name, os.path.join(scratch_dir, name))
                from_file = open(os.path.join(scratch_dir, name))
                line = from_file.readline()
                line = line.replace(self.input_project, self.output_project)
                to_file_name = os.path.join(scratch_dir, name.replace(self.input_project, self.output_project))
                to_file = open(to_file_name, "w")
                to_file.write(line)
                shutil.copyfileobj(from_file, to_file)
                if self.dry_run:
                    output_bucket = 'beep-sync-test'
                else:
                    output_bucket = self.bucket
                try:
                    new_object_name = os.path.join(directory, name.replace(self.input_project, self.output_project))
                    response = s3.upload_file(to_file_name, output_bucket, new_object_name)
                except ClientError as e:
                    print(e)


if __name__ == "__main__":
    transfer = ProjectTransfer("PredictionDiagnostics",
                               "PreDiag",
                               "beep-input-data",
                               "d3Batt/raw/maccor/STANFORD LOANER #2")
    # List of strings here can be used to filer out specific file names so that not all files are transferred over
    # '_000052' is a file generated using an even earlier version of the protocol and probably should not be transferred
    # over
    excluded_strings = ['_000052', 'Logs']
    names = transfer.get_list_files(excluded_strings)
    transfer.download_and_modify(names)
    print(names)
