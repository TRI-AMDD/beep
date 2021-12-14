"""
The entire BEEP CLI.

Guiding principles of how the CLI works:
 - Errors are thrown (stderr) if the *command itself* is bad.
 - Errors on individual files are caught during processing and
   reported via logging or status json.

The main "beep" command with no subcommand can specify where
and how to log all output and results (i.e., reporting).

The subcommands themselves specify where and how to process
actual BEEP operations such as structuring, featurization,
protocol generation, and running models.

"""

import os
import ast
import sys
import time
import copy
import pprint
import fnmatch
import hashlib
import logging
import datetime
import traceback
import importlib

import click
import numpy as np
import pandas as pd
from monty.serialization import dumpfn, loadfn

from beep import (
    logger,
    S3_CACHE,
    formatter_jsonl,
    __version__
)
from beep.structure.base import BEEPDatapath
from beep.structure.cli import auto_load, auto_load_processed
from beep.structure.validate import BEEPValidationError
from beep.features.base import (
    BEEPFeaturizer,
    BEEPFeaturizationError,
    BEEPFeatureMatrix,
)
from beep.features.core import (
    HPPCResistanceVoltageFeatures,
    DeltaQFastCharge,
    TrajectoryFastCharge,
    CycleSummaryStats,
    DiagnosticProperties,
    DiagnosticSummaryStats
)
from beep.features.intracell_losses import (
    IntracellCycles,
    IntracellFeatures
)
from beep.model import BEEPLinearModelExperiment
from beep.utils.s3 import list_s3_objects, download_s3_object
from beep.protocol.generate_protocol import generate_protocol_files_from_csv
from beep.protocol.generate_protocol import ProtocolException


CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
CLICK_DIR = click.Path(file_okay=False, dir_okay=True, writable=True, readable=True)
STRUCTURED_SUFFIX = "_structured"
FEATURIZED_SUFFIX = "_featurized"


class ContextPersister:
    """
    Class to hold persisting objects for downstream BEEP tasks in the
    command line interface.

    Args:
        cwd (str): The current working directory.
        run_id (int): The current run id, if available, to associate with operation.
        tags ([str]): A list of tags to associate with an operation.
        output_status_json (str): Filename of status json to write.
        halt_on_error (bool): Whether to halt on error during an operation.
    """
    def __init__(
            self,
            cwd=None,
            run_id=None,
            tags=None,
            output_status_json=None,
            halt_on_error=None

    ):
        self.cwd = cwd
        self.run_id = run_id
        self.tags = tags
        self.output_status_json = output_status_json
        self.halt_on_error = halt_on_error


def add_suffix(full_path, output_dir, suffix, modified_ext=None):
    """Add structured filename suffixes and format new file paths.

    Args:
        full_path (str): Full path of the existing file.
        output_dir (str): Desired output directory of file.
        suffix (str): Suffix to append to filename before the extension.
        modified_ext (str): Extension to replace current file extension
            with. For example, ".json".

    Returns:
        (str): Renamed and reformatted file path.

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


@click.group(
    invoke_without_command=False,
    help="Base BEEP command."
)
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
@click.option(
    '--output-status-json',
    '-s',
    type=CLICK_FILE,
    multiple=False,
    help="File to output with JSON info about the states of "
         "files which have had any beep subcommand operation"
         "run on them (e.g., structuring). Contains comprehensive"
         "info about the success of the operation for all files."
         "1 status json = 1 operation."
)
@click.option(
    '--halt-on-error',
    is_flag=True,
    default=False,
    help="Set to halt BEEP if critical featurization "
         "errors are encountered on any file with any featurizer. "
         "Otherwise, logs critical errors to the status json.",
)
@click.pass_context
def cli(ctx, log_file, run_id, tags, output_status_json, halt_on_error):
    """
    Base command for all BEEP subcommands. Sets CWD and persistent
    context.


    Args:
        log_file (str): The name of the log file to append log
            output to.
        run_id (int): Run identifier associated with this operation.
        tags ([str]): List of string tags to associate with this
            operation.
        output_status_json (str): File to write the status json to.
        halt_on_error (bool): If True, will halt the entire CLI
            operation if any exception occurs, even in suboperations.

    Returns:
        None
    """
    ctx.ensure_object(ContextPersister)
    cwd = os.path.abspath(os.getcwd())
    ctx.obj.cwd = cwd
    ctx.obj.tags = tags
    ctx.obj.run_id = run_id
    ctx.obj.output_status_json = output_status_json
    ctx.obj.halt_on_error = halt_on_error

    if log_file:
        hdlr = logging.FileHandler(log_file, "a")
        hdlr.setFormatter(formatter_jsonl)
        logger.addHandler(hdlr)


@cli.command(
    help="Structure and/or validate one or more files. Argument "
         "is a space-separated list of files or globs.",
    short_help="interpolate, validate, and standardize battery files"
)
@click.argument(
    'files',
    nargs=-1,
    type=CLICK_FILE,
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
         "structure. Use with --automatic."
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
    '--automatic',
    is_flag=True,
    default=False,
    help="If --protocol-parameters-path is specified, will "
         "automatically determine structuring parameters. Will override "
         "all manually set structuring parameters."
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
         "prevents files from being partially restructured."
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
        automatic,
        validation_only,
        no_raw,
        s3_use_cache
):
    """
    Structure files using the beep cli.


    raw files ---> structured json files.



    Args:
        ctx (ContextPersister): A context-persisting object containing
            metadata and logging specifications for subcommands.
        files ([str]): Paths or globs for raw input files to structure.
        output_filenames ([str]): Corresponding output filenames for each
            input file to write structured output to.
        output_dir (str): Output directory to write auto-named structured files
            into. Overrides output_filenames.
        protocol_parameters_dir (str): Directory containing protocol parameters
            for automatically determining structuring parameters.
        v_range ((float, float)): Tuple of voltage floats to determine voltage range
            for interpolation during structuring. Overridden by automatic, if
            automatic specified.
        resolution (int): Number of points to interpolate during structuring. Overridden
            by automatic, if automatic specified.
        nominal_capacity (float): Nominal capacity of the cell to use during structuring.
            Overridden by automatic, if automatic specified.
        full_fast_charge (float): Fast charge threshold to use during structuring.
            Overridden by automatic, if automatic specified.
        charge_axis (str): Standardized name of the charge axis to interpolate with
            respect to. Overridden by automatic, if automatic specified.
        discharge_axis (str): Standardized name of the discharge axis to interpolate
            with respect to. Overridden by automatic, if automatic specified.
        s3_bucket (str): S3 bucket to check for files. Files matching keys in this
            bucket will automatically be downloaded to the CWD or S3 cache and then
            used for structuring.
        automatic (bool): If true, overrides other structuring parameters and uses
            the specified protocol_parameters_dir to automatically determine
            structuring parameters.
        validation_only (bool): If True, only validates files, does not structure them
            or produce structured outputs.
        no_raw (bool): If True, keeps only the structured data, does not retain
            the raw data in the structured output datapaths. WARNING: files structured
            with no_raw=True cannot be restructured without the raw file and operations
            are not reversible without access to the raw file.
        s3_use_cache (bool): If true, uses the beep S3 cache to store raw files rather
            than dumping them in the CWD.

    Returns:
        None

    """

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
                    s3_keys_matched += matching_files
                else:
                    local_files.append(maybe_glob)

        logger.info(f"Found {len(s3_keys_matched)} matching files on s3")
        local_files_from_s3 = []
        for s3k in s3_keys_matched:
            s3k_basename = os.path.basename(s3k)
            pardir = S3_CACHE if s3_use_cache else ctx.obj.cwd
            pardir = os.path.abspath(pardir)
            if not os.path.exists(pardir):
                os.makedirs(pardir)
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

    if automatic and not protocol_parameters_dir:
        logger.warning(
            "--automatic was passed but no protocol parameters "
            "directory was specified! Default will be used."
            "Pass --protocol-parameters-dir to use autostructuring."
        )

    params = {
        "v_range": v_range,
        "resolution": resolution,
        "nominal_capacity": nominal_capacity,
        "full_fast_charge": full_fast_charge,
        "charge_axis": charge_axis,
        "discharge_axis": discharge_axis
    }

    status_json = {
        "op_type": "structure",
        "files": {}
    }

    log_prefix = "No file"
    for i, f in enumerate(files):
        op_result = {
            "validated": False,
            "validation_schema": None,
            "structured": False,
            "output": None,
            "traceback": None,
            "walltime": None,
            "raw_md5_chksum": None,
            "structuring_parameters": None,
            "processed_md5_chksum": None
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
                raise BEEPValidationError(validation_reason)

            logger.info(f"File {i + 1} of {n_files}: Validated: {f}")

            if not validation_only:
                logger.info(f"{log_prefix}: Structuring: Read from {f}")
                if automatic:
                    dp.autostructure(
                        charge_axis=charge_axis,
                        discharge_axis=discharge_axis,
                        parameters_path=protocol_parameters_dir
                    )
                else:
                    dp.structure(**params)

                output_fname = output_files[i]
                dp.to_json_file(output_fname, omit_raw=no_raw)
                op_result["structured"] = True
                op_result["structuring_parameters"] = dp.structuring_parameters
                op_result["output"] = output_fname
                op_result["processed_md5_chksum"] = md5sum(output_fname)
                logger.info(f"{log_prefix}: Structured: Written to {output_fname}")

        except KeyboardInterrupt:
            logging.critical("Keyboard interrupt caught - exiting...")
            click.Context.exit(1)

        except BaseException:
            tbinfo = sys.exc_info()
            tbfmt = traceback.format_exception(*tbinfo)
            logger.error(f"{log_prefix}: Failed/invalid: ({tbinfo[0].__name__}): {f}")
            op_result["traceback"] = tbfmt

            if ctx.obj.halt_on_error:
                raise

        t1 = time.time()
        op_result["walltime"] = t1 - t0
        status_json["files"][f] = op_result

    # Generate the status report
    succeeded, failed, invalid = [], [], []

    for input_fname, op_result in status_json["files"].items():
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

    osj = ctx.obj.output_status_json
    if osj:
        dumpfn(status_json, osj)
        logger.info(f"Wrote status json file to {osj}")


@cli.command(
    help="Featurize one or more files. Argument "
         "is a space-separated list of files or globs. The same "
         "features are applied to each file. Naming of output "
         "files is done automatically, but the output directory "
         "can be specified.",
    short_help="create feature matrices from structured files"
)
@click.argument(
    'files',
    nargs=-1,
    type=CLICK_FILE,
)
@click.option(
    '--output-filename',
    '-o',
    type=CLICK_FILE,
    help="Filename to save entre feature matrix to. If not specified, output file"
         "will be named with FeatureMatrix-[timestamp].json.gz. If specified, "
         "overrides the output dir for saving the feature matrix to file."
)
@click.option(
    '--output-dir',
    '-d',
    type=CLICK_DIR,
    help="Directory to dump auto-named files to."
)
@click.option(
    '--featurize-with',
    "-f",
    default=["all_features"],
    multiple=True,
    type=click.STRING,
    help="Specify a featurizer to apply by class name, e.g. "
         "HPPCResistanceVoltageFeatures. To apply more than one "
         "featurizer, use multiple -f <FEATURIZER> commands. To apply sets of"
         "core BEEP featurizers, pass either 'all_features' for all features or"
         "'all_targets' for all targets (features which can be used as targets). "
         "Note if 'all_features' "
         "or 'all_targets' is passed, other -f featurizers will be ignored. All "
         "feautrizers are attempted to apply with default hyperparameters; "
         "to specify your own hyperparameters, use --featurize-with-hyperparams."
         "Classes from installed modules not in core BEEP can be "
         "specified with the class name in absolute import format, "
         "e.g., my_package.my_module.MyClass."
)
@click.option(
    "--featurize-with-hyperparams",
    "-h",
    multiple=True,
    help="Specify a featurizer to apply by class name with your own hyperparameters."
         "(such as parameter directories or specific values for hyperparameters"
         "for this featurizer), pass a dictionary in the format: "
         "'{\"FEATURIZER_NAME\": {\"HYPERPARAM1\": \"VALUE1\"...}}' including the "
         "single quotes around the outside and double quotes for internal strings."
         "Custom hyperparameters will be merged with default hyperparameters if the "
         "hyperparameter dictionary is underspecified.",
)
@click.option(
    "--save-intermediates",
    is_flag=True,
    help="Save the intermediate BEEPFeaturizers as json files. Filenames "
         "are autogenerated and saved in output-dir if specified; otherwise, "
         "intermediates are written to current working directory."
)
@click.pass_context
def featurize(
        ctx,
        files,
        output_filename,
        output_dir,
        featurize_with,
        featurize_with_hyperparams,
        save_intermediates,
):
    """
    Create feature matrices based on structured inputs for use with
    downstream models.

    Can also be used for creating various targets, such as expected
    cycles for given capacities, etc.


    structured json files ---> feature matrix json file

    AND

    structured json files ---> featurizer json files (optional)

    Args:
        ctx (ContextPersister): A context-persisting object containing
            metadata and logging specifications for subcommands.
        files ([str]): Filenames or globs of structured json/json.gz files to
            use for generating a new feature matrix.
        output_filename (str): Filename of the output feature matrix file.
        output_dir (str): Directory to dump auto-named file into. Will also
            dump intermediate files here if save_intermediates.
        featurize_with ([str]): Featurizer class names to use with default
            featurizer hyperparameters. Can include builtin "core" featurizers
            such as CycleSummaryStats or custom featurizers inheriting from
            BEEPFeaturizer by passing the fully qualified importable name
            of the class, e.g. "my_pkg.my_module.my_class".
        featurize_with_hyperparams (dict): More advanced hyperparameter specification
            for featurizers. Pass in one dictionary per featurizer with the
            featurizer class name as the sole key and a dictionary of parameters
            as the sole value. E.g., {"SomeFeaturizerClass": {"param1": 12}}.
            Underspecified hyperparameter dictionaries are merged with defaults.
        save_intermediates (bool): If True, will save the intermediate json files
            of BEEPFeaturizer objects used to construct the feature martix.

    Returns:
        None

    """
    files = [os.path.abspath(f) for f in files]

    n_files = len(files)
    output_dir = os.path.abspath(output_dir) if output_dir else ctx.obj.cwd

    logger.info(f"Featurizing {n_files} files")

    core_fclasses_feats = [
        HPPCResistanceVoltageFeatures,
        DeltaQFastCharge,
        DiagnosticSummaryStats,
        CycleSummaryStats,
    ]

    core_fclasses_targets = [
        TrajectoryFastCharge,
        DiagnosticProperties,
    ]
    native_fclasses = core_fclasses_feats + core_fclasses_targets + [IntracellCycles, IntracellFeatures]

    core_fclasses_feats_map = {fclass.__name__: fclass for fclass in core_fclasses_feats}
    core_fclasses_targets_map = {fclass.__name__: fclass for fclass in core_fclasses_targets}
    native_fclasses_map = {fclass.__name__: fclass for fclass in native_fclasses}

    # Create canonical featurizer list if "all" options are selected
    if "all_features" in featurize_with:
        featurize_with = list(core_fclasses_feats_map.keys())
    elif "all_targets" in featurize_with:
        featurize_with = list(core_fclasses_targets_map.keys())

    # Feature class names along with hyperparameters
    # These are all default
    fclass_names_w_params = [(fclass_name, None) for fclass_name in featurize_with]

    # Add featurizers with custom parameters to list of featurizers to apply
    for fstr in featurize_with_hyperparams:
        fdict = ast.literal_eval(fstr)
        if not isinstance(fdict, dict):
            raise TypeError(f"Could not parse input featurizer with parameters string {fdict}")
        if len(fdict) != 1:
            raise ValueError(f"Featurizer must be specified as sole root key of hyperparam dictionary: {fdict}")
        fclass_name_w_params = [(k, v) for k, v in fdict.items()][0]
        fclass_names_w_params.append(fclass_name_w_params)

    # Determine actual classes to apply by joining with external modules
    fclass_tuples = []
    for fclass_name, fclass_params in fclass_names_w_params:
        if fclass_name in native_fclasses_map:
            fclass = native_fclasses_map[fclass_name]
        else:
            # it is assumed it will be an external module
            if "." not in fclass_name:
                logging.critical(
                    f"'{fclass_name}' not recognized as BEEP native featurizer "
                    f"or importable module."
                )
                click.Context.exit(1)

            modname, _, clsname = fclass_name.rpartition('.')
            mod = importlib.import_module(modname)
            cls = getattr(mod, clsname)

            if not issubclass(cls, BEEPFeaturizer):
                logging.critical(f"Class {cls.__name__} is not a subclass of BEEPFeaturizer.")
                click.Context.exit(1)
            fclass = cls

        # check parameter arguments and update with full hyperparameter specifications
        hps = copy.deepcopy(fclass.DEFAULT_HYPERPARAMETERS)
        if fclass_params is not None:
            hps.update(fclass_params)
        fclass_tuples.append((fclass, hps))

    logger.info(f"Applying {len(fclass_tuples)} featurizers to each of {n_files} files")

    # ragged featurizers apply is ok

    status_json = {
        "op_type": "featurize",
        "feature_matrix": None,
        "files": {}
    }
    i = 0
    bfs = []
    for file in files:
        log_prefix = f"File {i + 1} of {n_files}"

        t0_file = time.time()
        logger.debug(f"Hashing file '{file}' to MD5")
        op_result = {
            "walltime": None,
            "processed_md5_chksum": md5sum(file),
            "featurizers": []
        }

        logger.debug(f"{log_prefix}: Loading processed run '{file}'.")
        structured_datapath = auto_load_processed(file)
        logger.debug(f"{log_prefix}: Loaded processed run '{file}' into memory.")

        for fclass, f_hyperparams in fclass_tuples:
            fclass_name = fclass.__name__
            op_subresult = {
                "featurizer_name": fclass_name,
                "hyperparameters": f_hyperparams,
                "output": None,
                "valid": False,
                "featurized": False,
                "walltime": None,
                "traceback": None,
                "subop_md5_chksum": None
            }

            t0 = time.time()
            try:

                f = fclass(
                    structured_datapath=structured_datapath,
                    hyperparameters=f_hyperparams
                )

                is_valid, reason = f.validate()

                if is_valid:
                    op_subresult["valid"] = True
                    logger.info(f"{log_prefix}: Featurizer {fclass_name} valid with "
                                f"params {f_hyperparams} for '{file}'")
                else:
                    raise BEEPFeaturizationError(reason)

                f.create_features()
                op_subresult["featurized"] = True
                logger.info(
                    f"{log_prefix}: Featurizer {fclass_name} applied with params {f_hyperparams} for '{file}'")
                bfs.append(f)

                if save_intermediates:
                    intermediate_filename = f"{fclass_name}-{os.path.basename(file)}"
                    output_path = os.path.join(output_dir, intermediate_filename)
                    dumpfn(f, output_path)
                    logger.info(
                        f"{log_prefix}: Featurizer {fclass_name} features for '{file}' written to '{output_path}'")
                    op_subresult["output"] = output_path
                    logger.debug(f"Hashing sub-operation output file '{output_path}' to MD5")
                    op_subresult["subop_md5_chksum"] = md5sum(output_path)

            except KeyboardInterrupt:
                logger.critical("Keyboard interrupt caught - exiting...")
                click.Context.exit(1)

            except BaseException:
                tbinfo = sys.exc_info()
                logger.error(
                    f"{log_prefix}: Failed/invalid: ({tbinfo[0].__name__}): {fclass.__name__}")
                op_subresult["traceback"] = traceback.format_exception(*tbinfo)

                if ctx.obj.halt_on_error:
                    raise

            t1 = time.time()
            op_subresult["walltime"] = t1 - t0

            op_result["featurizers"].append(op_subresult)

        t1_file = time.time()
        op_result["walltime"] = t1_file - t0_file
        status_json["files"][file] = op_result
        i += 1

    feature_matrix_status = {}
    try:
        feature_matrix = BEEPFeatureMatrix(bfs)
        default_filename = f"FeatureMatrix-{datetime.datetime.now().strftime('%Y-%d-%m_%H.%M.%S.%f')}.json.gz"

        # Output filename specification overrides output dir
        if output_filename:
            output_filename = os.path.abspath(output_filename)
        else:
            output_filename = os.path.join(ctx.obj.cwd, default_filename)

        dumpfn(feature_matrix, output_filename)
        logger.info(f"Feature matrix of size {feature_matrix.matrix.shape} s"
                    f"uccessfully created and saved to {output_filename}")
        feature_matrix_status["created"] = True
        feature_matrix_status["traceback"] = None
        feature_matrix_status["output"] = output_filename
    except BaseException:
        if ctx.obj.halt_on_error:
            raise

        tbinfo = sys.exc_info()
        logging.critical(f"Feature matrix could not be created: '{tbinfo[0].__name__}'!")
        feature_matrix_status["created"] = False
        feature_matrix_status["traceback"] = traceback.format_exception(*tbinfo)
        feature_matrix_status["output"] = None

    status_json["feature_matrix"] = feature_matrix_status

    # Generate a summary output

    logger.info("Featurization report:")

    all_succeeded, some_succeeded, none_succeeded = [], [], []
    for file, data in status_json["files"].items():
        feats_succeeded = []

        for fdata in data["featurizers"]:
            feats_succeeded.append(fdata["featurized"])

        n_success = sum(feats_succeeded)
        if n_success == len(fclass_tuples):
            all_succeeded.append((file, n_success))
        elif n_success == 0:
            none_succeeded.append((file, n_success))
        else:
            some_succeeded.append((file, n_success))

    logger.info(f"\tAll {len(fclass_tuples)} featurizers succeeded: {len(all_succeeded)}/{n_files}")
    if len(all_succeeded) > 0:
        for filename, _ in all_succeeded:
            logger.info(f"\t\t- {filename}")

    if len(fclass_tuples) > 1:
        logger.info(f"\tSome featurizers succeeded: {len(some_succeeded)}/{n_files}")
        if len(some_succeeded) > 0:
            for filename, n_success in some_succeeded:
                logger.info(f"\t\t- {filename}: {n_success}/{len(fclass_tuples)}")

    logger.info(f"\tNo featurizers succeeded or file failed: {len(none_succeeded)}/{n_files}")
    if len(none_succeeded) > 0:
        for filename, _ in none_succeeded:
            logger.info(f"\t\t- {filename}")

    logger.info(f"\tFeaturization matrix created: {status_json['feature_matrix']['created']}")

    status_json = add_metadata_to_status_json(status_json, ctx.obj.run_id, ctx.obj.tags)
    osj = ctx.obj.output_status_json
    if osj:
        dumpfn(status_json, osj)
        logger.info(f"Wrote status json file to {osj}")


@cli.command(
    help="Train a machine learning model using all available data "
         "and save it to file.",
    short_help="train a model with existing features"
)
@click.option(
    "--output-filename",
    "-o",
    required=False,
    type=CLICK_FILE,
    help="Filename (json) to write the BEEP linear model object to "
         "when training is finished."
)
@click.option(
    '--feature-matrix-file',
    '-fm',
    required=True,
    type=CLICK_FILE,
    help="Featurization matrix serialized to file, containing "
         "features (X) for learning. Featurization matrices can "
         "be generated by the beep featurize command."
)
@click.option(
    '--target-matrix-file',
    '-tm',
    required=True,
    type=CLICK_FILE,
    help="Featurization matrix serialized to file, containing "
         "targets (one y or more) for learning. Featurization "
         "matrices can be generated by the beep featurize command."
)
@click.option(
    '--targets',
    '-t',
    required=True,
    multiple=True,
    type=click.STRING,
    help="Target columns to as from target matrix file. Must all "
         "be present in the target matrix file. If more than 1 is "
         "specified (e.g., -t 'col1' -t 'col2'), multitask regression "
         "will be performed. Column names will be '<Feature Name>::"
         "<Featurizer Class Name>' if --homogenize-features is "
         "set. If not, column names include long parameter hashes "
         "which must be included in this argument option."
)
@click.option(
    "--model-name",
    "-m",
    required=True,
    type=click.STRING,
    help="Name of the regularized linear model to use. Current selection "
         f"includes {BEEPLinearModelExperiment.ALLOWED_MODELS}."
)
@click.option(
    "--train-on-frac-and-score",
    "-s",
    type=click.FLOAT,
    help="Do hyperparameter tuning on part (a training fraction) of the "
         "dataset and use that fitted model to predict on a testing "
         "fraction of the dataset. Specify the training fraction as "
         "a float 0-1."
)
@click.option(
    "--alpha-lower",
    "-al",
    type=click.FLOAT,
    help="Lower bound on the grid for the alpha hyperparameter "
         " which will be explored during hyperparameter tuning. "
         "Must be specified with --alpha-upper and --n-alphas."
)
@click.option(
    "--alpha-upper",
    "-au",
    type=click.FLOAT,
    help="Upper bound on the grid for the alpha hyperparameter "
         " which will be explored during hyperparameter tuning. "
         "Must be specified with --alpha-lower and --n-alphas."
)
@click.option(
    "--n-alphas",
    "-an",
    type=click.FLOAT,
    help="Number of linearly spaced alphas to explore during "
         "hyperparameter tuning. If not specified, sklearn defaults "
         "are used. Must be specified with --alpha-upper and "
         "--alpha-lower."
)
@click.option(
    "--train-feature-nan-thresh",
    type=click.FLOAT,
    help="Threshold to keep a feature in the training dataset, in "
         "fraction of samples which must not be nan from 0-1. "
         "0 = any feature having any nan is dropped, 1 = no "
         "features are dropped."
)
@click.option(
    "--train-sample-nan-thresh",
    type=click.FLOAT,
    help="Threshold to keep a sample from the training data, in "
         "fraction of features which must not be nan "
         "from 0-1. 0 = any sample having any "
         "nan feature is dropped, 1 = no samples are dropped."
)
@click.option(
    "--predict-sample-nan-thresh",
    type=click.FLOAT,
    help="Threshold to keep a sample from any prediction set, including "
         "those used internally, in fraction of features which must not"
         "be nan."
)
@click.option(
    "--drop-nan-training-targets",
    is_flag=True,
    default=True,
    help="Drop samples containing any nan targets. If False and "
         "the targets matrix has nan targets, the command will fail."
)
@click.option(
    "--impute-strategy",
    type=click.STRING,
    help="Type of imputation to use, 'median', 'mean', or 'none'."
)
@click.option(
    "--kfold",
    type=click.INT,
    help="Number of folds to use in k-fold hyperparameter tuning."
)
@click.option(
    "--max-iter",
    type=click.INT,
    help="Number of iterations during training to fit linear parameters."
)
@click.option(
    "--tol",
    type=click.FLOAT,
    help="Tolerance for optimization."
)
@click.option(
    "--l1-ratios",
    type=click.STRING,
    help="Comma separated l1 ratios to try in hyperparameter optimization."
         "For example, '0.1,0.5,0.7,0.9,0.95,1.0', and all values must be "
         "between 0-1."
)
@click.option(
    "--homogenize-features",
    type=click.BOOL,
    help="Shorten feature names to only include the featurizer name and "
         "(very short) feature name. For example, "
         "'capacity_0.8::TrajectoryFastCharge', where features normally "
         "have names including their (long) parameter hashes. To use the "
         "literal feature names, specify False."
)
@click.pass_context
def train(
        ctx,
        output_filename,
        feature_matrix_file,
        target_matrix_file,
        targets,
        model_name,
        train_on_frac_and_score,
        alpha_lower,
        alpha_upper,
        n_alphas,
        train_feature_nan_thresh,
        train_sample_nan_thresh,
        predict_sample_nan_thresh,
        drop_nan_training_targets,
        impute_strategy,
        kfold,
        max_iter,
        tol,
        l1_ratios,
        homogenize_features
):
    """
    Use a feature matrix and a target matrix to train a machine learning model.
    Automatically performs hyperparameter tuning of linear models and can
    evaluate test fractions automatically, returning error metrics in the
    status and logs.


    feature matrix json + target matrix json ---> trained model json


    Args:
        ctx (ContextPersister): A context-persisting object containing
            metadata and logging specifications for subcommands.
        output_filename (str): The output filename to save the trained model
            to. Must be json.
        feature_matrix_file (str): Path of the file to use as the learning
            features, X. Must be a serialized BEEPFeatureMatrix.
        target_matrix_file (str): Path of the file containing the target
            features, y or Y (if multiple). Must be a serialized
            BEEPFeatureMatrix.
        targets ([str]): One or more literal target columns in the target matrix to
            use as learning target or targets. Must be present in the target
            matrix. If homogenize_features is False, must be the full name
            of the feature, including the parameter hash.
        model_name (str): Model name to train. E.g., "elasticnet", OR "lasso",
            OR "ridge".
        train_on_frac_and_score (float): If set, this will train on a fraction of
            data (specified by the float) and then predict on the rest, returning
            error metrics in the test fraction in the status json and log.
        alpha_lower (float): Lower bound for alpha for linear model search.
        alpha_upper (float): Upper bounds for alpha for the linear model search.
        n_alphas (int): Number of alphas to search in the range [alpha_lower,
            alpha_upper].
        train_feature_nan_thresh (float): Threshold fraction of non-nan a feature
            column in the training set must possess in order to be retained, 0-1. 0=
            features are never dropped (may cause errors), 1= any feature containing
            any nan sample is dropped.
        train_sample_nan_thresh (float): Threshold fraction of non-nan features a sample
            row in the training set must possess in order to be retained, 0-1. 0=
            samples are never dropped (may cause errors), 1= any sample containing
            any nan feature is dropped.
        predict_sample_nan_thresh (float): Threshold fraction of non-nan features a sample
            row in the prediction set must possess in order to be retained, 0-1. 0=
            samples are never dropped (may cause errors). 1= any sample containing
            any nan feature is dropped.
        drop_nan_training_targets (bool): Drop any samples in the training set
            which do not have non-nan targets.
        impute_strategy (str): Select an impute strategy for samples in training
            and prediction which, even after dropping nan columns and samples
            according to thresholds, still have nans.
        kfold (int): Number of folds to perform cross-validation and hyperparameter
            optimization with.
        max_iter (int): Number of iterations to use for fitting linear models.
        tol (float): Tolerance for optimization.
        l1_ratios (str): Comma-separated l1 ratios to use in hyperparameter
            optimization for linear models.
        homogenize_features (bool): If True, shortens feature names to only
            include the featurizer and feature name. For example,
            'capacity_0.8::TrajectoryFastCharge'. Can be used for merging disparate
            features created with different hyperparameters, as long as the feature
            name and the featurizer name match. If False, keeps the parameter hash
            of the feature in the feature name. Set to False if using features
            generated by identical featurizer classes but different hyperparameters.
            If False, must specify the long feature name in the targets including
            the parameter hash.

    Returns:
        None

    """
    feature_matrix_file = os.path.abspath(feature_matrix_file)
    target_matrix_file = os.path.abspath(target_matrix_file)
    targets = list(targets)

    logger.info(
        f"Running training using files {feature_matrix_file} (features) "
        f"and {target_matrix_file} (targets)."
    )

    alpha_params = [alpha_lower, alpha_upper, n_alphas]
    if any(alpha_params) and not all(alpha_params):
        alphas = np.linspace(start=alpha_lower, stop=alpha_upper, num=n_alphas)
    else:
        alphas = None

    if l1_ratios is not None:
        l1_ratios = [float(l1r) for l1r in l1_ratios.strip().split(",")]

    additional_kwargs = {
        "train_feature_nan_thresh": train_feature_nan_thresh,
        "train_sample_nan_thresh": train_sample_nan_thresh,
        "predict_sample_nan_thresh": predict_sample_nan_thresh,
        "drop_nan_training_targets": drop_nan_training_targets,
        "impute_strategy": impute_strategy,
        "kfold": kfold,
        "max_iter": max_iter,
        "tol": tol,
        "l1_ratios": l1_ratios,
        "homogenize_features": homogenize_features,
        "alphas": alphas,
    }

    # only pass in arguments which will override the defaults of the lower class
    additional_kwargs = {k: v for k, v in additional_kwargs.items() if v is not None}

    logger.debug(f"Hashing file '{feature_matrix_file}' to MD5")
    logger.debug(f"Hashing file '{target_matrix_file}' to MD5")
    status_json = {
        "op_type": "train",
        "files": {
            "features": feature_matrix_file,
            "targets": target_matrix_file,
            "features_md5_chksum": md5sum(feature_matrix_file),
            "targets_md5_chksum": md5sum(target_matrix_file)
        },
        "model_results": {},
        "walltime": None,
        "trained_model": {
            "created": False,
            "output": None,
        },
        "traceback": None,
    }

    t0 = time.time()
    model, training_errors, test_errors = None, None, None
    try:
        bfm = BEEPFeatureMatrix.from_json_file(feature_matrix_file)
        btm = BEEPFeatureMatrix.from_json_file(target_matrix_file)

        blme = BEEPLinearModelExperiment(
            feature_matrix=bfm,
            target_matrix=btm,
            targets=targets,
            model_name=model_name,
            **additional_kwargs
        )

        if train_on_frac_and_score is not None:
            logger.info("Beginning training and scoring on test set.")
            model, training_errors, test_errors = blme.train_and_score(train_and_val_frac=train_on_frac_and_score)
            status_json["model_results"]["training_error"] = training_errors
            status_json["model_results"]["test_error"] = test_errors
            status_json["model_results"]["test_fraction"] = train_on_frac_and_score
            status_json["model_results"]["optimal_hyperparameters"] = blme.optimal_hyperparameters
        else:
            logger.info("Beginning training on all available data")
            model, training_errors = blme.train()
            test_errors = None
    except KeyboardInterrupt:
        logging.critical("Keyboard interrupt caught - exiting...")
        click.Context.exit(1)
    except BaseException:
        tbinfo = sys.exc_info()
        tbfmt = traceback.format_exception(*tbinfo)
        logger.error(f"Model training failed: ({tbinfo[0].__name__})")
        status_json["traceback"] = tbfmt

        if ctx.obj.halt_on_error:
            raise

    t1 = time.time()
    status_json["walltime"] = t1 - t0

    if model:
        logger.info(f"Model {model} trained, finding optimal hyperparamters {blme.optimal_hyperparameters}.")
        logger.info(f"Training error summary: {training_errors}")
        if test_errors:
            logger.info(f"Testing error summary: {test_errors}")

        default_filename = f"LinearModelExperiment-{datetime.datetime.now().strftime('%Y-%d-%m_%H.%M.%S.%f')}.json.gz"
        if output_filename:
            output_filename = os.path.abspath(output_filename)
        else:
            output_filename = os.path.join(ctx.obj.cwd, default_filename)
        dumpfn(blme, output_filename)
        logger.info(f"Wrote model {model} to path: {output_filename}")
        status_json["trained_model"]["created"] = True
        status_json["trained_model"]["output"] = output_filename

    status_json = add_metadata_to_status_json(status_json, ctx.obj.run_id, ctx.obj.tags)
    osj = ctx.obj.output_status_json
    if osj:
        dumpfn(status_json, osj)
        logger.info(f"Wrote status json file to {osj}")


@cli.command(
    help="Run a previously trained model to predict degradation targets."
         "The MODEL_FILE passed should be an output of 'beep train' or a"
         "serialized BEEPLinearModelExperiment object.",
    short_help="use a trained model to predict degradation"
)
@click.argument(
    "model_file",
    nargs=1,
    type=CLICK_FILE
)
@click.option(
    "--feature-matrix-file",
    "-fm",
    required=True,
    multiple=False,
    help="Feature matrix to use as input to the model. Predictions are based"
         "on these features."
)
@click.option(
    "--output-filename",
    "-o",
    required=False,
    type=CLICK_FILE,
    help="Filename (json) to write the final predicted dataframe to."
)
@click.option(
    "--predict-sample-nan-thresh",
    type=click.FLOAT,
    help="Threshold to keep a sample from any prediction set."
)
@click.pass_context
def predict(
        ctx,
        model_file,
        feature_matrix_file,
        output_filename,
        predict_sample_nan_thresh,

):
    """
    Predict degradation characteristics using a previously trained model.


    trained model json + new feature matrix json ---> predictions json


    Args:
        ctx (ContextPersister): A context-persisting object containing
            metadata and logging specifications for subcommands.
        model_file (str): The json or json.gz of a trained model file. Must
            be a BEEPLinearModelExperiment class serialized to disk.
        feature_matrix_file (str): Path to a feature matrix file to use
            as input features for inference. These features will be used to
            predict the degradation targets the previously trained model
            was fit on.
        output_filename (str): Path to output predictions to. Will be saved in
            dataframe format as a json.
        predict_sample_nan_thresh (float): Threshold fraction of features a sample
            row in the prediction set must possess in order to be dropped, 0-1. 0=
            samples are never dropped (may cause errors). 1= any sample containing
            any nan feature is dropped. Overrides any predict sample nan threshold
            set during model training.

    Returns:
        None

    """
    if output_filename and not output_filename.endswith(".json"):
        raise ValueError("--output-filename must end with '.json'.")

    default_filename = f"PredictedDegradationDF-" \
                       f"{datetime.datetime.now().strftime('%Y-%d-%m_%H.%M.%S.%f')}" \
                       f".json"
    if output_filename:
        output_filename = os.path.abspath(output_filename)
    else:
        output_filename = os.path.join(ctx.obj.cwd, default_filename)

    model_file = os.path.abspath(model_file)
    feature_matrix_file = os.path.abspath(feature_matrix_file)

    logger.debug(f"Hashing file '{model_file}' to MD5")
    logger.debug(f"Hashing file '{feature_matrix_file}' to MD5")
    status_json = {
        "op_type": "predict",
        "files": {
            "model": model_file,
            "predict_on_features": feature_matrix_file,
            "model_md5_chksum": md5sum(model_file),
            "predict_on_features_md5_chksum": md5sum(feature_matrix_file)
        },
        "walltime": None,
        "predictions": {
            "created": False,
            "output": None
        },
        "traceback": None,
    }

    t0 = time.time()
    predicted = None
    try:
        logger.debug(
            f"Loading linear model from file on disk at path: {model_file}")

        bfm = BEEPFeatureMatrix.from_json_file(feature_matrix_file)
        model = BEEPLinearModelExperiment.from_json_file(model_file)

        # Optionally override the previously set sample nan thresh
        # for prediction dataframes
        if predict_sample_nan_thresh:
            model.predict_sample_nan_thresh = predict_sample_nan_thresh

        logging.debug(
            f"Model instantiated ok, predicting on {bfm.matrix.shape[0]}"
            f" inference samples."
        )

        predicted, dropped_ix = model.predict(bfm)
        logging.info(
            f"Successfully predicted {predicted.shape[0]} samples of "
            f"{bfm.matrix.shape[0]} original samples."
        )

    except KeyboardInterrupt:
        logging.critical("Keyboard interrupt caught - exiting...")
        click.Context.exit(1)
    except BaseException:
        tbinfo = sys.exc_info()
        tbfmt = traceback.format_exception(*tbinfo)
        logger.error(f"Model inference failed: ({tbinfo[0].__name__})")
        status_json["traceback"] = tbfmt

        if ctx.obj.halt_on_error:
            raise

    t1 = time.time()
    status_json["walltime"] = t1 - t0

    if predicted is not None:
        predicted.to_json(output_filename)
        status_json["predictions"]["created"] = True
        status_json["predictions"]["output"] = output_filename
        logger.info(f"Successfully wrote predicted dataframe to {output_filename}.")

    status_json = add_metadata_to_status_json(status_json, ctx.obj.run_id, ctx.obj.tags)
    osj = ctx.obj.output_status_json
    if osj:
        dumpfn(status_json, osj)
        logger.info(f"Wrote status json file to {osj}")


@cli.command(
    help="Generate protocol for battery cyclers from a csv file input.",
    short_help="generate protocol for new cycler experiments"
)
@click.argument(
    "csv_file",
    nargs=1,
    type=CLICK_FILE
)
@click.option(
    "--output-dir",
    "-d",
    help="Directory to output files to. At least three subdirs will be created in this directory"
         "in order to organize the generated protocol files."
)
@click.pass_context
def protocol(
        ctx,
        csv_file,
        output_dir
):
    """
    Generate cycler protocol files using a templating csv and set of template files.


    WARNING: This CLI function is still under migration and may not work as intended.



    protocol generating csv ---> protocol files


    Args:
        ctx (ContextPersister): A context-persisting object containing
            metadata and logging specifications for subcommands.
        csv_file (str): Path to the protocol generating csv.
        output_dir (str): Path to the output directory where protocol files
            should be kept. Subdirectories are automatically created for
            various protocol file types.

    Returns:
        None

    """
    click.secho("Protocol generation has yet to be migrated entirely to the new CLI. May be unstable.", fg="red")

    cwd = ctx.obj.cwd

    csv_file = os.path.abspath(csv_file)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input file {csv_file} does not exist")

    output_dir = os.path.abspath(output_dir) if output_dir else cwd
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output dir {output_dir} does not exist.")

    logger.debug(f"Generating protocol based on '{csv_file}' into '{output_dir}'.")

    status_json = {
        "op_type": "protocol",
        "files": {
            "csv": csv_file,
            "csv_md5_chksum": md5sum(csv_file)
        },
        "walltime": None,
        "protocol": {},
        "traceback": None,
    }

    t0 = time.time()
    output_files, failures, result, message = [], [], "", {}
    try:
        results = generate_protocol_files_from_csv(csv_file, output_directory=output_dir)
        output_files, file_generation_failures, result, message = results

        if result == "error" or failures:
            msg = message["comment"]
            err = message["error"]
            raise ProtocolException(f"'{err}: {msg}'")
        else:
            logger.info(f"Created {len(output_files)} protocol files in `{output_dir}`.")
    except BaseException:
        tbinfo = sys.exc_info()
        tbfmt = traceback.format_exception(*tbinfo)
        logger.error(f"Protocol generation failed with : ({tbinfo[0].__name__})")
        status_json["traceback"] = tbfmt
    t1 = time.time()

    for of in output_files:
        status_json["protocol"][of] = {"generated": True}

    for f in failures:
        status_json["protocol"][f] = {"generated": False}

    status_json["walltime"] = t1 - t0
    status_json = add_metadata_to_status_json(status_json, ctx.obj.run_id, ctx.obj.tags)
    osj = ctx.obj.output_status_json
    if osj:
        dumpfn(status_json, osj)
        logger.info(f"Wrote status json file to {osj}")


@cli.command(
    help="View BEEP files for debugging and analysis.",
    short_help="viewer for beep files, help debug"
)
@click.argument(
    "file",
    nargs=1,
    type=CLICK_FILE
)
def inspect(file):
    """
    Inspect any BEEP object serialized to disk from its filename.

    Args:
        file (str): Path of the file to inspect. Should be json or json.gz.

    Returns:
        None

    """
    file = os.path.abspath(file)
    try:
        o = loadfn(file)

        # To weed out pesky datapaths from old serializations
        o.as_dict()
    except (AttributeError, ValueError, KeyError):

        # Maybe it is a structured file?
        if file.endswith(".json") or file.endswith(".json.gz"):
            try:
                o = auto_load_processed(file)
            except BaseException:
                raise ValueError(f"File format of json file {file} not "
                                 f"recognized with beep version {__version__}")

        # Or it could be a raw file?
        else:
            try:
                o = auto_load(file)
                logger.debug(f"Loaded potential raw file {file} as Datapath.")
            except BaseException:
                raise ValueError(f"File format of {file} not loadable with beep.")

    logger.info(f"Loaded {file} as type {type(o)}.")

    if isinstance(o, BEEPDatapath):
        print(f"\n\nBEEP Datapath: {file}\n\n")

        print(f"\nSemiunique id: '{o.semiunique_id}'")

        print("\nFile paths")
        pprint.pprint(o.paths)

        print("\nFile metadata:")
        pprint.pprint(o.metadata.raw)

        print(f"\nValidation schema: {o.schema}")

        print("\nStructuring parameters:")
        pprint.pprint(o.structuring_parameters)

        print("\nStructured attributes:")

        for attr in [
            "structured_summary",
            "structured_data",
            "diagnostic_data",
            "diagnostic_summary",
            "raw_data",
        ]:

            print(f"\n{attr}:")
            a = getattr(o, attr)
            if isinstance(a, pd.DataFrame):
                print(a)
                print(a.info())
            else:
                print("\tNo object.")

    elif isinstance(o, BEEPFeaturizer):
        print("\n\nBEEP Featurizer: {file}\n\n")

        print("\nFile paths:")
        pprint.pprint(o.paths)

        print(f"\nLinked datapath semiunique id: {o.linked_semiunique_id}")

        print("\nHyperparameters:")
        pprint.pprint(o.hyperparameters)

        print("\nMetadata:")
        pprint.pprint(o.metadata)

        print("\nFeatures:")
        print(o.features)

    elif isinstance(o, BEEPFeatureMatrix):
        print(f"\n\nBEEP Feature Matrix: {file}\n\n")

        print("\nFeaturizers:")
        for f in o.featurizers:
            d_display = f.as_dict()
            d_display.pop("features")
            print(f"\n\tFeaturizer {f.__class__.__module__} {f.__class__.__name__}")
            d_formatted = pprint.pformat(d_display)
            d_formatted = "\t" + d_formatted.replace("\n", "\n\t")
            print(d_formatted)

        print("\nMatrix:")
        print(o.matrix)
        print(o.matrix.info())

    elif isinstance(o, BEEPLinearModelExperiment):
        d = o.as_dict()

        print(f"\n\nBEEP Linear Model Experiment: {file}\n\n")

        print(f"\nTargets: {o.targets}")

        print(f"\nModel name: {o.model_name}")

        print(f"\nImpute strategy: {o.impute_strategy}")

        print(f"\nHomogenize features: {o.homogenize_features}")

        print("\nNaN Thresholds:")
        for k in [
            "train_feature_drop_nan_thresh",
            "train_sample_drop_nan_thresh",
            "predict_sample_nan_thresh"
        ]:
            print(f"\t-{k}: {d[k]}")

        print("\nModel parameters:")
        for p, v in d["model_sklearn"].items():
            print(f"\t- {p}: {v}")

        print("\nMatrices:")
        for attr in ["feature_matrix", "target_matrix"]:
            a = getattr(o, attr).matrix

            print(f"\n{attr}")
            print(a)
            print(a.info())

    else:
        pprint.pprint(o.as_dict())
        logger.warning(f"Could not identify object {file}! Dictionary representation printed.")
