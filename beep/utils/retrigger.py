#!/usr/bin/env python3
#  Copyright (c) 2019 Toyota Research Institute

"""Script for re-triggering all of the files at a particular S3 location for processing.
The script produces events that mimic the events from the S3Syncer lambda function,
as a result the same actions should be taken on these files as if they were newly uploaded

Usage:
    retrigger.py [options]
    retrigger.py (-h | --help)

Options:
    -h --help                Show this screen
    --version                Show version
    --mode <mode>            Mode to run in [default: 'test']
    --s3_prefix <s3_prefix>  Prefix to use [default: 'd3Batt/raw/arbin']
    --s3_output <s3_output>  Output prefix to use [default: 'd3Batt/structure']

"""

import time
import datetime
import pytz
import boto3
import collections
import ast
from docopt import docopt
from beep.utils import KinesisEvents

S3_BUCKET_IN = "beep-input-data-stage"
S3_BUCKET_OUT = "beep-output-data-stage"


class DotDict(collections.OrderedDict):
    """Ordered dictionary that can reference keys with "dict.key" notation"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def eval_args(args):
    """Evaluate docopt arguments and get correctly typed objects

    Args:
        args (dict): Dictionary of arguments parsed by docopt.
    """

    def _fix_boolean(x):
        if x == "True":
            return True
        elif x == "False":
            return False
        else:
            return x

    def _parse_args(x):
        x = _fix_boolean(x)
        try:
            return ast.literal_eval(x)
        except ValueError:
            return x
        except SyntaxError:
            return x

    return DotDict(
        dict(zip(map(lambda x: x[2:], args.keys()), map(_parse_args, args.values())))
    )


def get_structure_name(object):
    file_name = object["Key"].split("/")[-1]
    structure_name = file_name.split(".")[0] + "_structure.json"
    return structure_name


def scan(config):
    print("scanning")
    s3 = boto3.client("s3")
    all_objects = s3.list_objects_v2(Bucket=S3_BUCKET_IN, Prefix=config.s3_prefix)

    objects = [obj for obj in all_objects["Contents"] if obj["Size"] > 1000]

    objects = [
        obj
        for obj in objects
        if "PredictionDiagnostics" in obj["Key"]
        and "x" not in obj["Key"]
        and "Complete" not in obj["Key"]
        # and obj['LastModified'] < datetime.datetime(2020, 3, 24, 5, 35, 43, tzinfo=tzutc())
        # and "_000175_" in obj['Key']
    ]

    old_objects = []
    old = datetime.datetime.now(pytz.utc) - datetime.timedelta(hours=6)
    for obj in objects:
        name = config.s3_output + "/" + get_structure_name(obj)
        structure_objects = s3.list_objects_v2(Bucket=S3_BUCKET_OUT, Prefix=name)
        # print(structure_objects)
        if (
            "Contents" in structure_objects.keys()
            and len(structure_objects["Contents"]) == 1
        ):
            if structure_objects["Contents"][0]["LastModified"] < old:
                old_objects.append(obj)
        else:
            old_objects.append(obj)

    objects = old_objects
    print(len(objects))

    events = KinesisEvents(service="S3Syncer", mode=config.mode)
    objects.reverse()
    for obj in objects:
        retrigger_data = {
            "filename": obj["Key"],
            "bucket": S3_BUCKET_IN,
            "size": obj["Size"],
            "hash": obj["ETag"].strip('"'),
        }
        events.put_upload_retrigger_event("complete", retrigger_data)
        print(retrigger_data)
        time.sleep(0.1)


if __name__ == "__main__":
    docopt_args = docopt(__doc__, version="BEEP Re-Trigger")
    parsed_args = eval_args(docopt_args)
    scan(parsed_args)
