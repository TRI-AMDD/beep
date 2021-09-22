# Copyright [2020] [Toyota Research Institute]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import datetime
import csv
import pandas as pd
from monty.os import makedirs_p
import xmltodict

from beep import logger
from beep.protocol import PROCEDURE_TEMPLATE_DIR, BIOLOGIC_TEMPLATE_DIR
from beep.protocol.maccor import Procedure, insert_driving_parametersv1, insert_charging_parametersv1
from beep.protocol.biologic import Settings


class ProtocolException(BaseException):
    """Errors related to generating protocol"""
    pass


def template_detection(filename, encoding="UTF-8"):
    """
    Reads a template and gathers metadata for use in decision logic

    Args:
        filename (str): Full path to template
        encoding (str): encoding used to read input file
    """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            text = f.read().decode(encoding)
        template_data = xmltodict.parse(text, process_namespaces=False, strip_whitespace=True)
        template_length = len(template_data["MaccorTestProcedure"]["ProcSteps"]["TestStep"])
    else:
        template_length = None
    return template_length


def generate_protocol_files_from_csv(csv_filename, output_directory=None):

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

    successfully_generated_files = []
    file_generation_failures = []
    names = []
    result = ""
    message = {"comment": "", "error": ""}
    if output_directory is None:
        output_directory = PROCEDURE_TEMPLATE_DIR

    # Create required directories if not already present
    for subdir in ["settings", "procedures", "names"]:
        subdir_path = os.path.abspath(os.path.join(output_directory, subdir))
        if not os.path.exists(subdir_path):
            makedirs_p(subdir_path)

    for index, protocol_params in protocol_params_df.iterrows():
        template = protocol_params["template"]
        protocol = None
        # Filename for the output
        filename_prefix = "_".join(
            [
                protocol_params["project_name"],
                "{:06d}".format(protocol_params["seq_num"]),
            ]
        )
        if ".000" in template:  # Extension for maccor procedure files
            template_fullpath = os.path.join(PROCEDURE_TEMPLATE_DIR, template)
            template_length = template_detection(template_fullpath)
            if "diagnostic_parameter_set" in protocol_params:  # For parameters include diagnostics load those values
                diag_params_df = pd.read_csv(
                    os.path.join(PROCEDURE_TEMPLATE_DIR, "PreDiag_parameters - DP.csv")
                )
                diagnostic_params = diag_params_df[
                    diag_params_df["diagnostic_parameter_set"]
                    == protocol_params["diagnostic_parameter_set"]
                    ].squeeze()

            if template_length == 23 and template == "EXP.000":  # length and name for initial procedure files
                protocol = Procedure.from_exp(
                    **protocol_params[["cutoff_voltage", "charge_rate", "discharge_rate"]]
                )
            elif template_length == 72:  # length for V1 and V1 diagnostic templates without ending diagnostics
                protocol = Procedure.from_regcyclev2(protocol_params)
                protocol.add_procedure_diagcyclev2(
                    protocol_params["capacity_nominal"], diagnostic_params
                )
            elif template_length == 96:  # template length for diagnostic type cycling
                mwf_dir = os.path.join(output_directory, "mwf_files")
                if protocol_params["project_name"] == "RapidC":  # Project with charging waveform
                    waveform_name = insert_charging_parametersv1(protocol_params,
                                                                 waveform_directory=mwf_dir)
                    protocol = Procedure.generate_procedure_chargingv1(index,
                                                                       protocol_params,
                                                                       waveform_name,
                                                                       template=template_fullpath)
                elif protocol_params["project_name"] == "Drive":  # Project with discharging waveform
                    waveform_name = insert_driving_parametersv1(protocol_params,
                                                                waveform_directory=mwf_dir)
                    protocol = Procedure.generate_procedure_drivingv1(index,
                                                                      protocol_params,
                                                                      waveform_name,
                                                                      template=template_fullpath)
                else:  # Use the default parameterization for PreDiag/Prediction Diagnostic projects
                    protocol = Procedure.generate_procedure_regcyclev3(index,
                                                                       protocol_params,
                                                                       template=template_fullpath)
                protocol.generate_procedure_diagcyclev3(
                    protocol_params["capacity_nominal"], diagnostic_params
                )
            else:  # Case where its not possible to match the procedure template
                failure = {
                    "comment": "Unable to find template: " + template,
                    "error": "Not Found",
                }
                file_generation_failures.append(failure)
                warnings.warn("Unsupported file template {}, skipping.".format(template))
                result = "error"
                continue

            filename = "{}.000".format(filename_prefix)
            filename = os.path.join(output_directory, "procedures", filename)

        elif ".mps" in template and template == "formationV1.mps":  # biologic settings template and formation project
            protocol = Settings.from_file(os.path.join(BIOLOGIC_TEMPLATE_DIR, template))
            protocol = protocol.formation_protocol_bcs(protocol_params)
            filename = "{}.mps".format(filename_prefix)
            filename = os.path.join(output_directory, "settings", filename)
        elif ".sdu" in template:  # No schedule file templates implemented
            failure = {
                "comment": "Schedule file generation is not yet implemented",
                "error": "Not Implemented"
            }
            file_generation_failures.append(failure)
            logger.warning("Schedule file generation not yet implemented")
            result = "error"
            continue
        else:  # Unable to match to any known template format
            failure = {
                "comment": "Unable to find template: " + template,
                "error": "Not Found",
            }
            file_generation_failures.append(failure)
            warnings.warn("Unsupported file template {}, skipping.".format(template))
            result = "error"
            continue

        logger.info(filename)
        protocol.to_file(filename)
        successfully_generated_files.append(filename)
        names.append(filename_prefix + "_")

    # This block of code produces the file containing all of the run file
    # names produced in this function call. This is to make starting tests easier
    _, namefile = os.path.split(csv_filename)
    namefile = namefile.split("_")[0] + "_names_"
    namefile = namefile + datetime.datetime.now().strftime("%Y%m%d_%H%M") + ".csv"

    names_dir = os.path.join(output_directory, "names")
    os.makedirs(names_dir, exist_ok=True)

    with open(os.path.join(names_dir, namefile), "w", newline="") as outputfile:
        wr = csv.writer(outputfile)
        for name in names:
            wr.writerow([name])
    outputfile.close()

    num_generated_files = len(successfully_generated_files)
    num_generation_failures = len(file_generation_failures)
    num_files = num_generated_files + num_generation_failures

    message = {
        "comment": "Generated {} of {} protocols".format(num_generated_files, num_files),
        "error": ""
    }
    if not result:
        result = "success"
    else:
        message["error"] = "Failed to generate {} of {} protocols".format(num_generation_failures, num_files)
        logger.error(message["error"])

    return successfully_generated_files, file_generation_failures, result, message
