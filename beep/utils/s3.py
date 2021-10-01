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


def list_s3_objects(bucket):
    """
    List all s3 objects available to user on S3 in a specific bucket.

    Args:
        bucket (str): S3 bucket

    Returns:
        ([boto3.ObjectSummary]): List of s3 objects.
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    return list(bucket.objects.all())
