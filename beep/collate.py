# Copyright 2019 Toyota Research Institute. All rights reserved.
"""
Module and script for renaming cycler files.

Usage:
    collate

Options:
    -h --help       Show this screen
    --version       Show version

The `collate` script takes no input, and operates by assuming the BEEP_ROOT (default `/`)
has subdirectories `/data-share/raw_cycler_files` and `data-share/renamed_cycler_files/FastCharge`.

The script moves files from the `/data-share/raw_cycler_files` directory, parses the metadata,
and renames them according to a combination of protocol, channel number, and date, placing them in
`/data-share/renamed_cycler_files`.

The script output is a json string that contains the following fields:

* `fid` - The file id used internally for renaming
* `filename` - full paths for raw cycler filenames
* `strname` - the string name associated with the file (i. e. scrubbed of `csv`)
* `file_list` - full paths for the new, renamed, cycler files
* `protocol` - the cycling protocol corresponding to each file
* `channel_no` - the channel number corresponding to each file
* `date` - the date corresponding to each file

Example:
>>> collate
... {"fid": [0, 1, 2], "strname": ["2017-05-09_test-TC-contact", "2017-08-14_8C-5per_3_47C", "2017-12-04_4_65C-69per_6C"],
...  "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv",
...                "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv",
...                "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"],
...  "protocol": [null, "8C(5%)-3.47C", "4.65C(69%)-6C"], "date": ["2017-05-09", "2017-08-14", "2017-12-04"],
...  "channel_no": ["CH33", "CH44", "CH29"],
...  "filename": ["/data-share/raw_cycler_files/2017-05-09_test-TC-contact_CH33.csv",
...               "/data-share/raw_cycler_files/2017-08-14_8C-5per_3_47C_CH44.csv",
...               "/data-share/raw_cycler_files/2017-12-04_4_65C-69per_6C_CH29.csv"]}
"""


import os
import re
import shutil
import json
import warnings

import pandas as pd
from monty.serialization import dumpfn
from beep import tqdm
from docopt import docopt


SRC_DIR = os.path.join("data-share", "raw_cycler_files")
DEST_DIR = os.path.join("data-share", "renamed_cycler_files")
PROJECT_NAME = 'FastCharge'
METADATA_COLUMN_NAMES = ['fid', 'protocol', 'channel_no', 'date',
                         'strname', 'filename', 'file_list']


def get_parameters_fastcharge(filename, source_directory):
    """
    Parse the filename to get parameters out, with fallback to metadata file

    Args:
        filename (str): this is the name of the file to be renamed.
        source_directory (str): this is the path to the file to be renamed

    Returns:
        Returns the date, the channel number, the name of the file, and the protocol (CC1, Q1, CC2) as str.

    """
    filename = filename.rsplit('.', 1)[0]
    strname = filename.rsplit('_CH', 1)[0]

    # Get date
    date = re.match(r'(\d+)-(\d+)-(\d+)', strname)
    if date is None:
        warnings.warn("Date could not be parsed from {}".format(filename))
    else:
        date = date.group()

    # Get channel number
    chno = re.findall(r'(CH+\d*)', filename)
    if chno:
        chno = chno[0]
    else:
        chno = None
        warnings.warn("Channel number could not be parsed from {}".format(filename))

    try:
        param_str = strname.rsplit(date + '_', 1)[1]
        param = param_str.replace('_', '.')

        if param.find('.') < 0:
            param = find_meta(filename, source_directory)

        q1 = re.findall(r'(\d+per)', param)[0].split('per')[0]
        cc1 = re.findall(r'(\d*\.?\d+C)', param.split(q1 + 'per.')[0])[0]
        cc2 = re.findall(r'(\d*\.?\d+C)', param.split(q1 + 'per.')[1])[0]
        protocol = cc1+'('+q1+'%)-'+cc2
    except Exception as e:
        warnings.warn("Failed to parse protocol for {}: {}".format(filename, e))
        protocol = None

    return date, chno, strname, protocol


def get_parameters_oed(filename, source_directory):
    """
    Parse the filename to get parameters out, with fallback to the metadata file

    Args:
        filename (str): this is the name of the file to be renamed.
        source_directory (str): this is the path to the file to be renamed.

    Returns:
        str: Returns the date, the channel number, the name of the file, and the protocol (CC1, Q1, CC2) as str.

    """
    filename = filename.rsplit('.', 1)[0]
    strname = filename.rsplit('_CH', 1)[0]

    # Get date
    date = re.match(r'(\d+)-(\d+)-(\d+)', strname)
    if date is None:
        warnings.warn("Date could not be parsed from {}".format(filename))
    else:
        date = date.group()

    # Get channel number
    chno = re.findall(r'(CH+\d*)', filename)
    if chno:
        chno = chno[0]
    else:
        chno = None
        warnings.warn("Channel number could not be parsed from {}".format(filename))

    try:
        param_str = strname.rsplit(date + '_', 1)[1]
        param = param_str.replace('_', '.')

        if param.find('.') < 0 or param.find('oed') >= 0:
            param = find_meta(filename, source_directory)

        cc1 = param.split('.')[0].lower().replace('pt',  '.')
        cc2 = param.split('.')[1].lower().replace('pt', '.')
        cc3 = param.split('.')[2].lower().replace('pt', '.')
        cc4 = param.split('.')[3].lower().replace('pt', '.')
        protocol = {
            "cc1": cc1,
            "cc2": cc2,
            "cc3": cc3,
            "cc4": cc4
                  }
        protocol = json.dumps(protocol)
    except Exception as e:
        warnings.warn("Failed to parse protocol for {}: {}".format(filename, e))
        protocol = None

    return date, chno, strname, protocol


def find_meta(filename, source_directory):
    """
    Find the metadata associated with the filename, and return the parameter string.
    Useful for file names such as batch#.

    Args:
        filename (str): this is the name of the file to be renamed.
        source_directory (str): this is the path to the file to be renamed.

    Returns:
        str: string containing parameters, to be parsed in GetParameters.

    """
    metafile = os.path.join(source_directory, filename + '_Metadata.csv')
    metadf = pd.read_csv(metafile)
    metadf = metadf.rename(str.lower, axis='columns')

    schfile = metadf['schedule_file_name'][0].split('\\')[-1].split('.sdu')[0].split('-')[1]
    param = schfile.replace('_', '.')

    return param


def init_map(project_name, destination_directory):
    """
    Create the initial mapping file if the project is new; import the existing mapping file
    if the project has already been renamed.

    Args:
        project_name (str): name of the project.
        destination_directory (str): directory to save the renamed files to.

    Returns:
        str: unique file identifier (file_id).
        pandas.DataFrame : the dataframe of the mapping file.

    """
    project_path = os.path.join(destination_directory, project_name)
    map_filename = os.path.join(project_path, project_name + "map.csv")
    if not os.path.exists(project_path):
        os.makedirs(os.path.join(destination_directory, project_name))
        file_id = 0
        mapdf = pd.DataFrame(columns=METADATA_COLUMN_NAMES)
        open(map_filename, 'a').close()
    elif len(os.listdir(project_path)) == 1:
        file_id = 0
        mapdf = pd.DataFrame(columns=METADATA_COLUMN_NAMES)
    else:
        mapdf = pd.read_csv(map_filename)
        mapdf.columns = METADATA_COLUMN_NAMES
        file_id = mapdf['fid'].max() + 1
    return file_id, mapdf


def process_files_json():
    """
    Inspects the BEEP_ROOT directory and renames
    files according to the prescribed system of protocol/date/run ID
    associated with the file metadata.  Since this script operates
    only on filesystem assumptions, no input is required.

    Returns:
        (str): json string corresponding to the locations of the renamed files.
    """
    # chdir into beep root
    pwd = os.getcwd()
    os.chdir(os.environ.get("BEEP_ROOT", "/"))

    meta_list = list(filter(lambda x: '_Metadata.csv' in x, os.listdir(SRC_DIR)))
    file_list = list(filter(lambda x: '.csv' in x if x not in meta_list else None, os.listdir(SRC_DIR)))
    all_list = list(filter(lambda x: '.csv' in x, os.listdir(SRC_DIR)))

    all_list = sorted(all_list)
    dumpfn(all_list, "all_files.json")

    [file_id, mapdf] = init_map(PROJECT_NAME, DEST_DIR)

    new_file_index = file_id

    for filename in tqdm(sorted(file_list)):
        # If the file has already been renamed another entry should not be made
        if mapdf['filename'].str.contains(filename).sum() > 0:
            continue
        old_file = os.path.join(SRC_DIR, filename)
        new_path = os.path.join(DEST_DIR, PROJECT_NAME)
        shutil.copy(old_file, new_path)  # copy main data file
        shutil.copy(old_file.replace(".csv", '_Metadata.csv'), new_path)  # copy meta data file

        if PROJECT_NAME == 'FastCharge':
            [date, channel_no, strname, protocol] = get_parameters_fastcharge(filename, SRC_DIR)
        elif PROJECT_NAME == 'ClosedLoopOED':
            [date, channel_no, strname, protocol] = get_parameters_oed(filename, SRC_DIR)
        else:
            raise ValueError("Unsupported PROJECT_NAME: {}".format(PROJECT_NAME))

        df_dup = mapdf.set_index(['protocol', 'date'])
        if (protocol, date) in df_dup.index:
            row = mapdf[(mapdf['protocol'] == protocol) & (mapdf['date'] == date)]
            file_id = row['fid'].iloc[0]
            protocol = row['protocol'].iloc[0]
            date = row['date'].iloc[0]
            strname = row['strname'].iloc[0]
        else:
            file_id = new_file_index
            new_file_index = new_file_index + 1

        new_name = "{}_{}_{}".format(PROJECT_NAME, f'{file_id:06}', channel_no)
        new_file = os.path.join(DEST_DIR, PROJECT_NAME, "{}.csv".format(new_name))

        new_row = pd.DataFrame([[file_id, protocol, channel_no, date, strname,
                                os.path.abspath(old_file),
                                os.path.abspath(new_file)]],
                               columns=METADATA_COLUMN_NAMES)
        mapdf = mapdf.append(new_row)

        os.rename(os.path.join(DEST_DIR, PROJECT_NAME, filename), new_file)
        os.rename(os.path.join(DEST_DIR, PROJECT_NAME, filename).replace(".csv", "_Metadata.csv"),
                  new_file.replace(".csv", "_Metadata.csv"))

        mapdf.to_csv(os.path.join(DEST_DIR, PROJECT_NAME, PROJECT_NAME + "map.csv"), index=False)
    mapdf = mapdf.reset_index(drop=True)
    os.chdir(pwd)
    return json.dumps(mapdf.to_dict("list"))


def main():
    """
    Main function used in script, primarily used as a handle
    to get the output into stdout.
    """
    # There are no args, but parse them just so help works
    args = docopt(__doc__)
    print(process_files_json(), end="")
    return None


def add_suffix_to_filename(filename, suffix):
    """
    Utility for modify filename strings with _suffix modifier, e. g.

    >>>print(add_suffix_to_filename("this_file.json", "_processed"))
    >>>"this_file_processed.json"

    Args:
        filename (str): filename to be modified.
        suffix (str): suffix to be added before final '.'.

    Returns:
        str: modified string with suffix.

    """
    name, ext = os.path.splitext(filename)
    return ''.join([name, suffix, ext])


def scrub_underscore_suffix(filename):
    """
    Creates a new string from old filename with suffix e. g. "_processed"
    in "this_file_processed.json".

    Args:
        filename (str): filename with some suffix to be scrubbed.

    Returns:
        str: filename with suffix scrubbed.

    """
    scrubbed = re.sub(r"_[^_]+\.", ".", filename)
    return scrubbed


if __name__ == '__main__':
    main()
