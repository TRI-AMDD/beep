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

"""

import time
import datetime
from dateutil.tz import tzutc
import boto3
import collections
import ast
from docopt import docopt
from beep.utils import KinesisEvents

S3_BUCKET = "beep-input-data"


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
        if x == 'True':
            return True
        elif x == 'False':
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

    return DotDict(dict(zip(map(lambda x: x[2:], args.keys()), map(_parse_args, args.values()))))


def scan(config):
    print("scanning")
    s3 = boto3.client("s3")
    all_objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=config.s3_prefix)

    objects = [obj for obj in all_objects['Contents']
               if obj['Size'] > 1000]

    # db_objects = dim_run['file_path_data'].tolist()
    # print(db_objects)
    # print(len([obj for obj in objects if obj['Key'] not in db_objects]))
    # objects = [obj for obj in objects if obj['Key'] not in db_objects]
    objects = [obj for obj in objects if "PredictionDiagnostics" in obj['Key']
               and "x" not in obj['Key']
               and "Complete" not in obj['Key']
               # and obj['LastModified'] < datetime.datetime(2020, 3, 24, 5, 35, 43, tzinfo=tzutc())
               and "_000159_" in obj['Key']
               ]
    print(len(objects))

    events = KinesisEvents(service='S3Syncer', mode=config.mode)
    objects.reverse()
    for obj in objects:
        retrigger_data = {
            "filename": obj['Key'],
            "bucket": S3_BUCKET,
            "size": obj['Size'],
            "hash": obj["ETag"].strip('\"')
        }
        events.put_upload_retrigger_event('complete', retrigger_data)
        print(retrigger_data)
        time.sleep(1)


if __name__ == "__main__":
    docopt_args = docopt(__doc__, version='BEEP Re-Trigger')
    parsed_args = eval_args(docopt_args)
    scan(parsed_args)
