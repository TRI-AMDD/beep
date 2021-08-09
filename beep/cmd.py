import datetime
import os
import sys
import time
import pprint
import fnmatch
import hashlib
import logging
import functools
import traceback
import importlib

import click
from monty.serialization import dumpfn

from beep import logger, BEEP_PARAMETERS_DIR, S3_CACHE, formatter_jsonl, __version__
from beep.structure.cli import auto_load
from beep.featurize import \
    HPPCResistanceVoltageFeatures, \
    DiagnosticSummaryStats, \
    DiagnosticProperties, \
    TrajectoryFastCharge, \
    DeltaQFastCharge, \
    BeepFeatures
from beep.utils.s3 import list_s3_objects, download_s3_object
from beep.validate import BeepValidationError

CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
CLICK_DIR = click.Path(file_okay=False, dir_okay=True, writable=True, readable=True)
BEEP_CMDS = ["structure", "featurize", "run_model"]
STRUCTURED_SUFFIX = "-structured"
FEATURIZED_SUFFIX = "-featurized"


class ContextPersister:
    """
    Class to hold persisting objects for downstream
    BEEP tasks.
    """
    def __init__(self, cwd=None, run_id=None, tags=None):
        self.cwd = cwd
        self.run_id = run_id
        self.tags = tags


def add_suffix(full_path, output_dir, suffix, modified_ext=None):
    """
    Add structured filename suffixes.

    Args:
        full_path:
        output_dir:
        suffix:
        modified_ext:

    Returns:

    """
    basename = os.path.basename(full_path)
    stripped_basename, ext = os.path.splitext(basename)
    if modified_ext:
        ext = modified_ext
    new_basename = stripped_basename + suffix + ext
    return os.path.join(
        output_dir,
        new_basename
    )


def add_metadata_to_status_json(status_dict, run_id, tags):
    """Add some basic metadata to the status json.

    Args:
        status_dict (dict): Dictionary which will be written to status hson.
        run_id (int): Run id of this operation.
        tags ([str]): List of short string tags tagging an operation.

    Returns:
        (dict): Dictionary including BEEP metadata
    """
    metadata = {
        "beep_verison": __version__,
        "op_datetime_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "tags": tags
    }
    status_dict["metadata"] = metadata
    return status_dict


def md5sum(filename):
    """
    Get md5 sum hash of a file.

    Args:
        filename (str): Name of the file.

    Returns:
        (str) Hash digest h.
    """
    with open(filename, "rb") as f:
        d = f.read()
        h = hashlib.md5(d).hexdigest()
    return h


@click.group(invoke_without_command=False)
@click.option(
    "--log-file",
    "-l",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, readable=True),
    multiple=False,
    help="File to log formatted json to. Log will still be output in human "
         "readable form to stdout, but if --log-file is specified, it will "
         "be additionally logged to a jsonl (json-lines) formatted file.",
)
@click.option(
    "--run-id",
    "-r",
    type=click.INT,
    multiple=False,
    help="An integer run_id which can be optionally assigned to this run. "
         "It will be output in the metadata status json for any subcommand "
         "if the status json is enabled."
)
@click.option(
    "--tags",
    "-t",
    type=click.STRING,
    multiple=True,
    help="Add optional tags to the status json metadata. Can be later used for"
         "large-scale queries on database data about sets of BEEP runs. Example:"
         "'experiments_for_kristin'."
)
@click.pass_context
def cli(ctx, log_file, run_id, tags):
    """
    Base command for all BEEP subcommands. Sets CWD and persistent
    context.
    """
    ctx.ensure_object(ContextPersister)
    cwd = os.path.abspath(os.getcwd())
    ctx.obj.cwd = cwd
    ctx.obj.tags = tags
    ctx.obj.run_id = run_id

    if log_file:
        hdlr = logging.FileHandler(log_file, "a")
        hdlr.setFormatter(formatter_jsonl)
        logger.addHandler(hdlr)

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
    type=CLICK_DIR,
    help="Directory of a protocol parameters files to use for "
         "auto-structuring. If not specified, BEEP cannot auto-"
         "structure. Use with --automatic. Can alternatively "
         "be set via environment variable BEEP_PARAMETERS_PATH."
)
@click.option(
    '--v-range',
    '-v',
    type=(click.FLOAT, click.FLOAT),
    help="Lower, upper bounds for voltage range for structuring. "
         "Overridden by auto-structuring if --automatic."
)
@click.option(
    '--resolution',
    '-r',
    type=click.INT,
    default=1000,
    help="Resolution for interpolation for structuring. Overridden "
         "by auto-structuring if --automatic."
)
@click.option(
    '--nominal-capacity',
    '-n',
    type=click.FLOAT,
    default=1.1,
    help="Nominal capacity to use for structuring. Overridden by "
         "auto-structuring if --automatic."
)
@click.option(
    '--full-fast-charge',
    '-f',
    type=click.FLOAT,
    default=0.8,
    help="Full fast charge threshold to use for structuring. "
         "Overridden by auto-structuring if --automatic."
)
@click.option(
    '--charge-axis',
    '-c',
    type=click.STRING,
    default='charge_capacity',
    help="Axis to use for charge step interpolation. Must be found "
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
    '--s3-bucket',
    '-b',
    default=None,
    type=click.STRING,
    help="Expands file paths to include those in the s3 bucket specified. "
         "File paths specify s3 keys. Keys can be globbed/wildcarded. Paths "
         "matching local files will be prioritized over files with identical "
         "paths/globs in s3. Files will be downloaded to CWD."
)
@click.option(
    '--halt-on-error',
    is_flag=True,
    default=False,
    help="Set to halt BEEP if critical structuring "
         "errors are encountered on any file. Otherwise, logs "
         "critical errors to the status json.",
)
@click.option(
    '--automatic',
    is_flag=True,
    default=False,
    help="If --protocol-parameters-path or the BEEP_PARAMETERS_"
         "PATH environment variable is specified, will automatically "
         "determine structuring parameters. Will override all "
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
    help="Does not save raw cycler data to disk. Saves disk space, but "
         "prevents files from being partially restructued."
)
@click.option(
    '--s3-use-cache',
    is_flag=True,
    default=False,
    help="Use s3 cache defined with environment variable BEEP_S3_CACHE "
         "instead of downloading files directly to the CWD."
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
        s3_bucket,
        halt_on_error,
        automatic,
        validation_only,
        no_raw,
        s3_use_cache
):

    # download from s3 first, if needed
    if s3_bucket:
        logger.info(f"Fetching file list from s3 bucket {s3_bucket}...")
        s3_objs = list_s3_objects(s3_bucket)
        logger.info(f"Including {len(s3_objs)} available s3 objects in file match.")
        s3_keys = [o.key for o in s3_objs]

        # local files matching globs are pre-expanded by Click
        s3_keys_matched = []
        local_files = []
        for maybe_glob in files:
            # add direct matches
            if "*" not in maybe_glob:
                if maybe_glob in s3_keys:
                    s3_keys_matched.append(maybe_glob)
                else:
                    local_files.append(maybe_glob)
            else:
                # its a glob, and real local globs will
                # be pre-expanded by click, so the only
                # valid globs will be on s3. All remaining
                # globs are invalid/bad paths
                matching_files = fnmatch.filter(s3_keys, maybe_glob)
                if matching_files:
                    s3_keys_matched.append(matching_files)
                else:
                    local_files.append(maybe_glob)

        logger.info(f"Found {len(s3_keys_matched)} matching files on s3")
        local_files_from_s3 = []
        for s3k in s3_keys_matched:
            s3k_basename = os.path.basename(s3k)
            pardir = S3_CACHE if s3_use_cache else ctx.obj.cwd
            s3k_local_fullname = os.path.join(pardir, s3k_basename)
            logger.info(f"Fetching {s3k} from {s3_bucket}")
            download_s3_object(s3_bucket, s3k, s3k_local_fullname)
            logger.info(f"Fetched s3 file {s3k_basename} to {s3k_local_fullname}")
            local_files_from_s3.append(s3k_local_fullname)
        files = local_files + local_files_from_s3

    files = [os.path.abspath(f) for f in files]

    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File '{file}' not found on filesystem!")
    n_files = len(files)

    logger.info(f"Structuring {n_files} files")

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

    if protocol_parameters_dir and BEEP_PARAMETERS_DIR:
        logger.warning(
            "Both --protocol-parameters-dir and $BEEP_PARAMETERS_PATH were specified. "
            "Defaulting to path from --protocol-parameters-dir."
        )
        params_dir = protocol_parameters_dir
    elif protocol_parameters_dir and not BEEP_PARAMETERS_DIR:
        params_dir = protocol_parameters_dir
    elif not protocol_parameters_dir and BEEP_PARAMETERS_DIR:
        params_dir = BEEP_PARAMETERS_DIR
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
            "validation_schema": None,
            "structured": False,
            "output": None,
            "traceback": None,
            "walltime": None,
            "raw_md5_chksum": None
        }

        t0 = time.time()
        try:
            log_prefix = f"File {i + 1} of {n_files}"
            logger.debug(f"Hashing file '{f}' to MD5")
            op_result["raw_md5_chksum"] = md5sum(f)

            logger.info(f"{log_prefix}: Reading raw file {f} from disk...")
            dp = auto_load(f)
            logger.info(f"{log_prefix}: Validating: {f} according to schema file '{dp.schema}'")
            op_result["validation_schema"] = dp.schema

            is_valid, validation_reason = dp.validate()
            op_result["validated"] = is_valid

            if not is_valid:
                raise BeepValidationError(validation_reason)

            logger.info(f"File {i + 1} of {n_files}: Validated: {f}")

            if not validation_only:
                logger.info(f"{log_prefix}: Structuring: Read from {f}")
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
                logger.info(f"{log_prefix}: Structured: Written to {output_fname}")

        except KeyboardInterrupt:
            logging.critical("Keyboard interrupt caught - exiting...")
            click.Context.exit(1)

        except BaseException:
            tbinfo = sys.exc_info()
            tbfmt = traceback.format_exception(*tbinfo)
            logger.error(f"{log_prefix}: Failed/invalid: ({tbinfo[0].__name__}): {f}")
            op_result["traceback"] = tbfmt

            if halt_on_error:
                raise

        t1 = time.time()
        op_result["walltime"] = t1 - t0
        status_json[f] = op_result

    # Generate the status report
    succeeded, failed, invalid = [], [], []

    for input_fname, op_result in status_json.items():
        if op_result["validated"] and op_result["structured"]:
            succeeded.append(input_fname)
        elif op_result["validated"] and not op_result["structured"]:
            failed.append(input_fname)
        else:
            invalid.append(input_fname)

    logger.info(f"{'Validation' if validation_only else 'Structuring'} report:")

    logger.info(f"\t{'Structured' if validation_only else 'Succeeded'}: {len(succeeded)}/{n_files}")
    logger.info(f"\tInvalid: {len(invalid)}/{n_files}")
    for inv in invalid:
        logger.info(f"\t\t- {inv}")

    logger.info(f"\t{'Validated, not structured' if validation_only else 'Failed'}: {len(failed)}/{n_files}")
    for fail in failed:
        logger.info(f"\t\t- {fail}")

    status_json = add_metadata_to_status_json(status_json, ctx.obj.run_id, ctx.obj.tags)

    if output_status_json:
        dumpfn(status_json, output_status_json)
        logger.info(f"Wrote status json file to {output_status_json}")


@cli.command(
    help="Featurize one or more files. Argument "
         "is a space-separated list of files or globs. The same "
         "features are applied to each file."
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
         "featurized and unfeaturized entries. Useful for "
         "high-throughput featurization or storage in NoSQL"
         "databases. If not set, status json is not written."
)
@click.option(
    '--output-filenames',
    '-o',
    type=click.Path(),
    help="Filenames to write each input filename to. "
         "If not specified, auto-names each file by appending"
         "`-featurized` before the file extension inside "
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
    '--featurizers',
    "-f",
    default=("all",),
    multiple=True,
    type=click.STRING,
    help="List featurizers to apply (space-separated). BEEP "
         "native featurizers include rptdqdv, hppcresistance, "
         "hppcrelaxation, diagsumstats, dqfastcharge, diagprops, and "
         "trajfastcharge. "
         "For all featurizers, use 'all'. For external featurizers"
         "that inherit the BeepFeatures class, specify the featurizer"
         "name with a fully specified importable module name and "
         "class name e.g., my_package.my_module.MyClass."
)
@click.option(
    '--halt-on-error',
    is_flag=True,
    default=False,
    help="Set to halt BEEP if critical featurization "
         "errors are encountered on any file with any featurizer. "
         "Otherwise, logs critical errors to the status json.",
)
@click.option(
    '--autocheck',
    is_flag=True,
    default=False,
    help="Automatically check the featurizers"
)
@click.pass_context
def featurize(
        ctx,
        files,
        output_status_json,
        output_filenames,
        output_dir,
        featurizers,
        halt_on_error,
        autocheck
):
    files = [os.path.abspath(f) for f in files]
    n_files = len(files)

    logger.info(f"Featurizing {n_files} files")

    allowed_featurizers = {
        "rptdqdv": RPTdQdVFeatures,
        "hppcresistance": HPPCResistanceVoltageFeatures,
        "hppcrelaxation": HPPCRelaxationFeatures,
        "diagsumstats": DiagnosticSummaryStats,
        "diagprops": DiagnosticProperties,
        "trajfastcharge": TrajectoryFastCharge,
        "dqfastcharge": DeltaQFastCharge
    }

    f_map = {}
    for f in featurizers:
        if f in allowed_featurizers:
            f_str = allowed_featurizers[f].__name__
            if f_str not in f_map:
                f_map[f_str] = allowed_featurizers[f]
        elif f == "all":
            for fcls in allowed_featurizers.values():
                f_map[fcls.__name__] = fcls
        else:
            # it is assumed it will be an external module
            if "." not in f:
                logging.critical(
                    f"'{f}' not recognized as BEEP native featurizer "
                    f"or importable module."
                )
                click.Context.exit(1)

            modname, _, clsname = f.rpartition('.')
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)

            if not issubclass(cls, BeepFeatures):
                logging.critical(f"Class {cls.__name__} is not a subclass of BeepFeatures.")
                click.Context.exit(1)

            f_map[cls.__name__] = cls

    logger.info(f"Applying {len(f_map)} featurizers: {list(f_map.keys())}")

    # ragged featurizers apply is ok

    for file in files:
        pass


    # add metadata to status json



@cli.command(
    help="Run a machine learning model using pre-featurized cell data as input"
         "and returning predictions as output."
)
@click.argument(
    'files',
    nargs=-1,
    type=CLICK_FILE,
)
@click.pass_context
def run_model(ctx, files):
    pass



if __name__ == "__main__":

    pass
