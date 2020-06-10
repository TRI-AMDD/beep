# Copyright 2019 Toyota Research Institute. All rights reserved.
"""
Script for generating protocol files from
input parameters and procedure templates

Usage:
    generate_protocol [INPUT_JSON]

Options:
    -h --help        Show this screen
    --version        Show version


The `generate_protocol` script will generate a protocol file from input
parameters defined in the rows of a CSV-formatted input file.
It stores its outputs in `/data-share/protocols/`|
                                                |-`/procedures/`
                                                |-`/schedules/`
                                                |-`/names/`
For Maccor procedures the output procedures will be stored in `/data-share/protocols/procedures/`
For Arbin schedules the schedules will be stored in `/data-share/protocols/schedules/`
Additionally a file containing the names for the test files generated during the function call
will be stored in `/data-share/protocols/names/` with a date and time in the name to differentiate it
This file is to facilitate the process of starting tests on the cycling machines, by making it easier to
enter data into the appropriate fields.

The input json must contain the following fields
* `file_list` - filenames/paths corresponding to the input csv files

The output json will contain the following fields
* `file_list` - list of protocol files corresponding to the input files

Example:
$ generate_protocol '{"file_list": ["/data-share/raw/parameters/procedure_params_000112e3.csv"]}'
{
    "file_list": ["/data-share/protocols/procedures/name_1.000", "/data-share/protocols/procedures/name_2.000"]
}
"""

import os
import warnings
import json
import datetime
import csv

import pandas as pd
from docopt import docopt
from monty.serialization import loadfn
from beep import logger, __version__
from beep.protocol import PROCEDURE_TEMPLATE_DIR
from beep.protocol.maccor import Procedure
from beep.utils import KinesisEvents
s = {'service': 'ProtocolGenerator'}


def generate_protocol_files_from_csv(csv_filename, output_directory):
    """
    Generates a set of protocol files from csv filename input by
    reading protocol file input corresponding to each line of
    the csv file. Writes a csv file that.

    Args:
        csv_filename (str): CSV containing protocol file parameters.
        output_directory (str): directory in which to place the output files
    """
    # Read csv file
    protocol_params_df = pd.read_csv(csv_filename)

    new_files = []
    names = []
    result = ''
    message = {'comment': '',
               'error': ''}
    for index, protocol_params in protocol_params_df.iterrows():
        template = protocol_params['template']

        # Switch for template invocation
        if template == "EXP.000":
            procedure = Procedure.from_exp(
                **protocol_params[["cutoff_voltage", "charge_rate", "discharge_rate"]]
            )
        elif template == 'diagnosticV2.000':
            diag_params_df = pd.read_csv(os.path.join(PROCEDURE_TEMPLATE_DIR,
                                                      "PreDiag_parameters - DP.csv"))
            diagnostic_params = diag_params_df[diag_params_df['diagnostic_parameter_set'] ==
                                               protocol_params['diagnostic_parameter_set']].squeeze()

            # TODO: should these be separated?
            procedure = Procedure.from_regcyclev2(
                protocol_params
            )
            procedure.add_procedure_diagcyclev2(
                protocol_params["capacity_nominal"], diagnostic_params
            )

        # TODO: how are these different?
        elif template in ['diagnosticV3.000', 'diagnosticV4.000']:
            diag_params_df = pd.read_csv(os.path.join(PROCEDURE_TEMPLATE_DIR,
                                                      "PreDiag_parameters - DP.csv"))
            diagnostic_params = diag_params_df[diag_params_df['diagnostic_parameter_set'] ==
                                               protocol_params['diagnostic_parameter_set']].squeeze()

            procedure = Procedure.generate_procedure_regcyclev3(index, protocol_params)
            procedure.generate_procedure_diagcyclev3(
                    protocol_params["capacity_nominal"], diagnostic_params
            )
        else:
            warnings.warn("Unsupported file template {}, skipping.".format(template))
            result = "error"
            message = {'comment': 'Unable to find template: ' + template,
                       'error': 'Not Found'}
            continue

        filename_prefix = '_'.join(
            [protocol_params["project_name"], '{:06d}'.format(protocol_params["seq_num"])])
        filename = "{}.000".format(filename_prefix)
        filename = os.path.join(output_directory, 'procedures', filename)
        logger.info(filename, extra=s)
        if not os.path.isfile(filename):
            procedure.to_file(filename)
            new_files.append(filename)
            names.append(filename_prefix + '_')

        elif '.sdu' in template:
            logger.warning('Schedule file generation not yet implemented', extra=s)
            result = "error"
            message = {'comment': 'Schedule file generation is not yet implemented',
                       'error': 'Not Implemented'}

    # This block of code produces the file containing all of the run file
    # names produced in this function call. This is to make starting tests easier
    _, namefile = os.path.split(csv_filename)
    namefile = namefile.split('_')[0] + '_names_'
    namefile = namefile + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv'
    with open(os.path.join(output_directory, "names", namefile), 'w', newline='') as outputfile:
        wr = csv.writer(outputfile)
        for name in names:
            wr.writerow([name])
    outputfile.close()

    if not result:
        result = "success"
        message = {'comment': 'Generated {} protocols'.format(str(len(new_files))),
                   'error': ''}

    return new_files, result, message


def process_csv_file_list_from_json(
        file_list_json,
        processed_dir='data-share/protocols/'):
    """

    Args:
        file_list_json (str):
        processed_dir (str):

    Returns:
        str:
    """
    # Get file list and validity from json, if ends with .json,
    # assume it's a file, if not assume it's a json string
    if file_list_json.endswith(".json"):
        file_list_data = loadfn(file_list_json)
    else:
        file_list_data = json.loads(file_list_json)

    # Setup Events
    events = KinesisEvents(service='protocolGenerator', mode=file_list_data['mode'])

    file_list = file_list_data['file_list']
    all_output_files = []
    protocol_dir = os.path.join(os.environ.get("BEEP_ROOT", "/"),
                              processed_dir)
    for filename in file_list:
        output_files, result, message = generate_protocol_files_from_csv(
            filename, output_directory=protocol_dir)
        all_output_files.extend(output_files)

    output_data = {"file_list": all_output_files,
                   "result": result,
                   "message": message
                   }

    events.put_generate_event(output_data, "complete")

    return json.dumps(output_data)


def main():
    """Main function for the script"""
    logger.info('starting', extra=s)
    logger.info('Running version=%s', __version__, extra=s)
    try:
        args = docopt(__doc__)
        input_json = args['INPUT_JSON']
        print(process_csv_file_list_from_json(input_json), end="")
    except Exception as e:
        logger.error(str(e), extra=s)
        raise e
    logger.info('finish', extra=s)
    return None


if __name__ == "__main__":
    main()
