#  Copyright (c) 2019 Toyota Research Institute

"""
Scripts for profiling memory use for the end to end pipeline.

Usage:
    python memprof.py

Options:
    -h --help        Show this screen
    --version        Show version
"""

import shutil
import os

import boto3

from memory_profiler import profile
from monty.tempfile import ScratchDir
from beep.structure import RawCyclerRun, ProcessedCyclerRun
from beep.run_model import DegradationModel
from beep.featurize import DegradationPredictor
from beep import S3_CACHE, tqdm

MEMORY_PROFILE_S3_OBJS = ["D3Batt_Data_publication/2017-05-12_5_4C-60per_3_6C_CH23.csv",
                          "D3Batt_Data_publication/2017-05-12_5_4C-60per_3_6C_CH23_Metadata.csv"]


@profile
def memory_profile(s3_objs=None):
    """
    Function for memory profiling pipelines with s3_objs.

    Args:
        s3_objs ([str]): list of s3_objs in the kitware d3batt
            publication s3 bucket.

    """
    s3_objs = s3_objs or MEMORY_PROFILE_S3_OBJS

    # Cache s3 objects
    cache_s3_objs(s3_objs, filter_existing_files=True)
    starting_dir = os.getcwd()

    with ScratchDir('.'):
        # Copy all pre-defined s3 objects
        for obj_name in s3_objs:
            shutil.copy(os.path.join(S3_CACHE, obj_name), '.')

        # Snip data path prefix
        data_paths = [os.path.basename(obj) for obj in s3_objs
                      if 'Metadata' not in obj]

        # Validation
        # validator = ValidatorBeep()
        # validator.validate_from_paths(data_paths)

        # Data structuring
        raw_cycler_runs = [RawCyclerRun.from_file(data_path)
               for data_path in data_paths]
        processed_cycler_runs = [ProcessedCyclerRun.from_raw_cycler_run(raw_cycler_run)
                                 for raw_cycler_run in raw_cycler_runs]

        # Featurization
        predictors = [
            DegradationPredictor.init_full_model(processed_cycler_run)
            for processed_cycler_run in processed_cycler_runs
        ]

        # Prediction
        predictions = [
            DegradationModel.init_full_model().predict(predictor)
            for predictor in predictors
        ]


def cache_s3_objs(obj_names, bucket_name='kitware',
                  filter_existing_files=True):
    """
    Quick function to download relevant s3 files to cache.

    Args:
        obj_names ([str]): s3 object keys.
        bucket_name (str): bucket name.
        filter_existing_files (bool): whether or not to filter existing files.

    Returns:
        None

    """

    # make cache dir
    if not os.path.isdir(S3_CACHE):
        os.mkdir(S3_CACHE)

    # Initialize s3 resource
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    # Put more s3 objects here if desired
    if filter_existing_files:
        obj_names = filter(lambda x: not os.path.isfile(os.path.join(S3_CACHE, x)), obj_names)
        obj_names = list(obj_names)

    for obj_name in tqdm(obj_names):
        path, filename = os.path.split(obj_name)
        # Make directory if it doesn't exist
        if not os.path.isdir(os.path.join(S3_CACHE, path)):
            os.makedirs(os.path.join(S3_CACHE, path))

        file_object = s3.Object(bucket_name, obj_name)
        file_size = file_object.content_length
        with tqdm(total=file_size, unit_scale=True, desc=filename) as t:
            bucket.download_file(
                obj_name, os.path.join(S3_CACHE, obj_name), Callback=hook(t))


def cache_all_kitware_data():
    """Quick function to cache all of the kitware data."""
    s3 = boto3.client("s3")
    all_objects = s3.list_objects(Bucket="kitware")
    kitware_objects = [obj['Key'] for obj in all_objects['Contents']
                       if obj['Key'].startswith("D3Batt_Data_publication/")]
    kitware_objects = [obj for obj in kitware_objects
                       if not 'struct' in obj]
    kitware_objects.remove("D3Batt_Data_publication/")
    cache_s3_objs(kitware_objects)


def hook(t):
    """tqdm hook for processing s3 downloads."""
    def inner(bytes_amount):
        t.update(bytes_amount)
    return inner


if __name__ == "__main__":
    memory_profile()
