import click


from beep.structure import process_file_list_from_json



CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
CLICK_DIR = click.Path(file_okay=False, dir_okay=True, writable=True, readable=True)
BEEP_CMDS = ["structure", "featurize", "run_model"]


def parse_files_glob(files_glob):
    files_sep = files_glob.split(files_glob)



@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@cli.command(
    help="Structure and/or validate one or more files. Argument "
         "is a comma-separated list of files or wildcards.")
@click.argument(
    'files',
    nargs=1,
    type=click.STRING,
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
    type=click.STRING,
    help="Filenames to write each input filename to. "
         "If not specified, auto-names each file by appending"
         "`-structured` before the file extension inside "
         "the current working dir."
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
         "PATH environment variable is specified, will automatically"
         "determine structuring parameters. Will override all"
         "manually set structuring parameters."
)
def structure(
        ctx,
        files_glob,
        status_json,
        files_output,
        dir_output,
        error_handling,
        parameters_file,
        v_range,
        resolution,
        nominal_capacity,
        full_fast_charge,
        chg_axis,
        dchg_axis,
        automatic,



):
    click.echo("Running structure ok.")

#
