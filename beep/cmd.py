import os
import sys
import time
import pprint
import traceback

import click
from monty.serialization import dumpfn

from beep import logger, ENV_PARAMETERS_DIR
from beep.structure.cli import auto_load

CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
CLICK_DIR = click.Path(file_okay=False, dir_okay=True, writable=True, readable=True)
BEEP_CMDS = ["structure", "featurize", "run_model"]
STRUCTURED_SUFFIX = "-structured"
FEATURIZED_SUFFIX = "-featurized"

STRUCTURE_CONFIG = {"service": "DataStructurer"}



class ContextPersister:
    """
    Class to hold persisting objects for downstream
    BEEP tasks.
    """
    def __init__(self, cwd=None):
        self.cwd = cwd


def add_suffix(full_path, output_dir, suffix, modified_ext=None):
    basename = os.path.basename(full_path)
    stripped_basename, ext = os.path.splitext(basename)
    if modified_ext:
        ext = modified_ext
    new_basename = stripped_basename + suffix + ext
    return os.path.join(
        output_dir,
        new_basename
    )

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
         "databases. If not set, status json is not written."
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
    '--protocol-parameters-dir',
    '-p',
    type=CLICK_FILE,
    help="Directory of a protocol parameters files to use for "
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
         "inside the loaded dataframe. Can be used with --automatic."
)
@click.option(
    '--discharge-axis',
    '-x',
    type=click.STRING,
    default='voltage',
    help="Axis to use for discharge step interpolation. Must be "
         "found inside the loaded dataframe. Can be used with"
         "--automatic."
)
@click.option(
    '--halt-on-error',
    is_flag=True,
    help="Set to halt BEEP if critical structuring"
         "errors are encountered on any file. Otherwise, logs "
         "critical errors to the status json.",
    default="log"
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
    '--no-raw',
    is_flag=True,
    default=False,
    help="Does not save raw cycler data to disk. Saves disk space, but"
         "prevents files from being partially restructued."
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
        protocol_parameters_dir,
        v_range,
        resolution,
        nominal_capacity,
        full_fast_charge,
        charge_axis,
        discharge_axis,
        halt_on_error,
        automatic,
        validation_only,
        no_raw,
        s3
):
    files = [os.path.abspath(f) for f in files]
    n_files = len(files)

    # Output dir overrules output filenames
    if output_dir:
        # Use auto-naming in the output dir
        output_dir = os.path.abspath(output_dir)
        output_files = [
            add_suffix(f, output_dir, STRUCTURED_SUFFIX, modified_ext=".json.gz")
            for f in files
        ]

        if output_filenames:
            logger.warning(
                "Both --output-filenames and --output-dir were specified; "
                "defaulting to --output-dir with auto-naming."
            )
    else:
        if output_filenames:
            output_files = [os.path.abspath(f) for f in output_filenames]
            n_outputs = len(output_files)
            if n_files != n_outputs:
                raise ValueError(
                    f"Number of input files ({n_files}) does not match number "
                    f"of output filenames ({n_outputs})!"
                )
        else:
            # Use auto-naming in the cwd
            output_files = [
                add_suffix(f, ctx.obj.cwd, STRUCTURED_SUFFIX, modified_ext=".json.gz")
                for f in files
            ]

    logger.info(f"Input files: {pprint.pformat(files)}", extra=STRUCTURE_CONFIG)
    logger.info(f"Output files: {pprint.pformat(output_files)}", extra=STRUCTURE_CONFIG)


    if protocol_parameters_dir and ENV_PARAMETERS_DIR:
        logger.warning(
            "Both --protocol-parameters-dir and $BEEP_PARAMETERS_PATH were specified. "
            "Defaulting to path from --protocol-parameters-dir."
        )
        params_dir = protocol_parameters_dir
    elif protocol_parameters_dir and not ENV_PARAMETERS_DIR:
        params_dir = protocol_parameters_dir
    elif not protocol_parameters_dir and ENV_PARAMETERS_DIR:
        params_dir = ENV_PARAMETERS_DIR
    else:
        # neither are defined
        params_dir = None

    if automatic and not params_dir:
        logger.warning(
            "--automatic was passed but no protocol parameters "
            "directory was specified! Either set BEEP_PARAMETERS_DIR "
            "or pass --protocol-parameters-dir to use autostructuring."
        )

    params = {
        "v_range": v_range,
        "resolution": resolution,
        "nominal_capacity": nominal_capacity,
        "full_fast_charge": full_fast_charge,
        "charge_axis": charge_axis,
        "discharge_axis": discharge_axis
    }

    status_json = {}
    for i, f in enumerate(files):
        op_result = {
            "validated": False,
            "structured": False,
            "output": None,
            "traceback": None,
            "walltime": None,
        }

        t0 = time.time()
        try:
            dp = auto_load(f)

            logger.info(f"Validating file {i} of {n_files}: {f}")
            dp.validate()
            op_result["validated"] = True
            logger.info(f"Validated file {i} of {n_files}: {f}")

            if not validation_only:
                logger.info(f"Structuring file {i} of {n_files}: Read from {f}")
                if automatic:
                    dp.autostructure(
                        charge_axis=charge_axis,
                        discharge_axis=discharge_axis,
                        parameters_path=params_dir
                    )
                else:
                    dp.structure(**params)

                output_fname = output_files[i]
                dp.to_json_file(output_fname, omit_raw=no_raw)
                op_result["structured"] = True
                logger.info(f"Structured file {i} of {n_files}: Written to {output_fname}")

        except BaseException:
            tbinfo = sys.exc_info()
            tbfmt = traceback.format_tb(*tbinfo)
            logger.error("\n".join(tbfmt))
            op_result["traceback"] = tbfmt

            if halt_on_error:
                raise

        status_json[f] = op_result
        t1 = time.time()
        op_result["walltime"] = t1 - t0

        if output_status_json:
            dumpfn(status_json, output_status_json)
            logger.info(f"Wrote status json file to {output_status_json}")
