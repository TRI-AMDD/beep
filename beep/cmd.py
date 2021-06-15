import os
import sys
import glob
import pprint
import traceback

import click
from monty.json import MSONable

from beep import logger
from beep.structure.cli import auto_load
from beep.validate import BeepValidationError
from beep.structure import process_file_list_from_json

CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
CLICK_DIR = click.Path(file_okay=False, dir_okay=True, writable=True, readable=True)
BEEP_CMDS = ["structure", "featurize", "run_model"]
FILE_DELIMITER = ","
FILE_DELIMITER_ESC_SEQ = '\,'
FILE_DELIMITER_UNICODE_PLACEHOLDER = "ʤ"



STRUCTURE_CONFIG = {"service": "DataStructurer"}



class ContextPersister:
    """
    Class to hold persisting objects for downstream
    BEEP tasks.
    """
    def __init__(self, cwd=None):
        self.cwd = cwd

@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    """
    Base command for all BEEP subcommands. Sets CWD and persistent
    context.
    """
    ctx.ensure_object(ContextPersister)
    ctx.obj.cwd = os.path.abspath(os.getcwd())


@cli.command(
    help="Structure and/or validate one or more files. Argument "
         "is a space-separated list of files or globs."
)
@click.argument(
    'files',
    nargs=-1,
    type=CLICK_FILE,
)
@click.option(
    '--output-status-json',
    '-s',
    type=CLICK_FILE,
    multiple=False,
    help="File to output with JSON info about the states of "
         "structured and unstrutured entries. Useful for "
         "high-throughput structuring or storage in NoSQL"
         "databases."
)
@click.option(
    '--output-filenames',
    '-o',
    type=click.Path(),
    help="Filenames to write each input filename to. "
         "If not specified, auto-names each file by appending"
         "`-structured` before the file extension inside "
         "the current working dir.",
    multiple=True
)
@click.option(
    '--output-dir',
    '-d',
    type=CLICK_DIR,
    help="Directory to dump auto-named files to. Only works if"
         "--output-filenames is not specified."
)
@click.option(
    '--error-handling',
    '-e',
    type=click.STRING,
    help="Set to `halt` to throw stacktrace if critical structuring"
         "errors are encountered on any file. Set to `log` to pass"
         "critical errors to the status json.",
    default="log"
)
@click.option(
    '--protocol-parameters-file',
    '-p',
    type=CLICK_FILE,
    help="File path of a protocol parameters file to use for "
         "auto-structuring. If not specified, BEEP cannot auto-"
         "structure. Use with --automatic. Can alternatively"
         "be set via environment variable BEEP_PARAMETERS_PATH."
)
@click.option(
    '--v-range',
    '-v',
    type=(click.FLOAT, click.FLOAT),
    help="Lower, upper bounds for voltage range for structuring."
         "Overridden by auto-structuring if --automatic."
)
@click.option(
    '--resolution',
    '-r',
    type=click.INT,
    default=1000,
    help="Resolution for interpolation for structuring. Overridden"
         "by auto-structuring if --automatic."
)
@click.option(
    '--nominal-capacity',
    '-n',
    type=click.FLOAT,
    default=1.1,
    help="Nominal capacity to use for structuring. Overridden by"
         "auto-structuring if --automatic."
)
@click.option(
    '--full-fast-charge',
    '-f',
    type=click.FLOAT,
    default=0.8,
    help="Full fast charge threshold to use for structuring."
         "Overridden by auto-structuring if --automatic."
)
@click.option(
    '--charge-axis',
    '-c',
    type=click.STRING,
    default='charge_axis',
    help="Axis to use for charge step interpolation. Must be found"
         "inside the loaded dataframe."
)
@click.option(
    '--discharge-axis',
    '-x',
    type=click.STRING,
    default='voltage',
    help="Axis to use for discharge step interpolation. Must be "
         "found inside the loaded dataframe."
)
@click.option(
    '--automatic',
    is_flag=True,
    default=False,
    help="If --protocol-parameters-path or the BEEP_PARAMETERS_"
         "PATH environment variable is specified, will automatically "
         "determine structuring parameters. Will override all"
         "manually set structuring parameters."
)
@click.option(
    '--validation-only',
    is_flag=True,
    default=False,
    help='Skips structuring, only validates files.'
)
@click.option(
    '--s3',
    is_flag=True,
    default=False,
    help="Expands file paths to include those in s3 buckets. "
         "s3 must be preconfigured on system."
)
@click.pass_context
def structure(
        ctx,
        files,
        output_status_json,
        output_filenames,
        output_dir,
        error_handling,
        protocol_parameters_file,
        v_range,
        resolution,
        nominal_capacity,
        full_fast_charge,
        charge_axis,
        discharge_axis,
        automatic,
        validation_only,
        s3
):
    files = [os.path.abspath(f) for f in files]
    output_files = [os.path.abspath(f) for f in output_filenames]



    logger.info(f"Input files: {pprint.pformat(files)}", extra=STRUCTURE_CONFIG)
    logger.info(f"Output files: {pprint.pformat(output_files)}", extra=STRUCTURE_CONFIG)



    # if protocol_parameters_file
    #
    # params = {
    #     "v_range": v_range,
    #     "resolution": resolution,
    #     "nominal_capacity": nominal_capacity,
    #     "full_fast_charge": full_fast_charge,
    #     "charge_axis": charge_axis,
    #     "discharge_axis": discharge_axis
    # }
    #
    #
    # n_files = len(files)
    #
    #
    #
    # for i, f in enumerate(files):
    #
    #     op_result = {
    #         "validated": False,
    #         "structured": False,
    #         "input": f,
    #         "output": None,
    #         "traceback": None,
    #         "time_elapsed": None,
    #     }
    #     try:
    #         dp = auto_load(f)
    #
    #         logger.info(f"Validating file {i} of {n_files}: {f}")
    #         dp.validate()
    #         op_result["validated"] = True
    #         logger.info(f"Validated file {i} of {n_files}: {f}")
    #
    #         if not validation_only:
    #             logger.info(f"Structuring file {i} of {n_files}: {f}")
    #             if automatic:
    #                 dp.autostructure()
    #             else:
    #                 dp.structure(**params)
    #             op_result["structured"] = True
    #             logger.info(f"Structured file {i} of {n_files}: {f}")
    #
    #     except BeepValidationError:
    #         exc_type, exc_value, exc_traceback = sys.exc_info()
    #
    #
    #
    #     logger.info()


    # cwd = ctx.obj.cwd

    # click.echo(pprint.pformat(files))



    # click.echo(pprint.pformat(bad_globs), err=True)
    # res = process_file_list_from_json()
    # click.echo()