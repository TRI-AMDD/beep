"""
Module to provide connection objects to various postgres environments.

Current environments:
- local     Local development environment. Non-secret credentials.
- stage     Beep stage environment. Credentials to database instance are
            obtained through AWS SecretsManager.

"""

import boto3
import base64
import warnings
from botocore.exceptions import ClientError
import json
from beep import ENVIRONMENT
from beep.config import config


def secret_accessible(environment):
    event_config = config[environment]["kinesis"]
    if "stream" in event_config:
        secret_name = event_config["stream"]
        try:
            _ = get_secret(secret_name)
        except Exception as e:
            print(e)
            return False
        else:
            return True
    else:
        return True


def get_secret(secret_name):
    """
    Returns the secret for the beep database and respective environment.

    Args:
        secret_name:    str representing the location in secrets manager

    Returns:
        secret          dict object containing database credentials

    """
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary,
        # one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            return json.loads(secret)
        else:
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response["SecretBinary"]
            )
            return json.loads(decoded_binary_secret)


def event_setup():
    # Setup events for testing
    if not secret_accessible(ENVIRONMENT):
        events_mode = "events_off"
    else:
        try:
            kinesis = boto3.client("kinesis")
            response = kinesis.list_streams()
            assert response is not None
            events_mode = "test"
        except Exception as e:
            warnings.warn("Cloud resources not configured, error: {}".format(e))
            events_mode = "events_off"
    return events_mode
