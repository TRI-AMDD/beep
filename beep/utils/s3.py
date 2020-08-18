"""
Module for S3 operations
"""
import boto3


def download_s3_object(bucket, key, destination_path):
    """
    Download an object from a bucket/key location on S3 to local disk.

    Args:
        bucket (str):                   S3 bucket
        key (str):                      S3 key
        destination_path (str):         local path at which file object is to be stored
    """
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, destination_path)
    return
