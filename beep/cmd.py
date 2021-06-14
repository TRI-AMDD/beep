import click


from beep.structure import process_file_list_from_json



CLICK_FILE = click.Path(file_okay=True, dir_okay=False, writable=False, readable=True)
BEEP_CMDS = ["structure", "featurize", "run_model"]


def parse_files_glob(files_glob):
    files_sep = files_glob.split(files_glob)



@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand not in ["init", "example"]:
        pass

@cli.command(help="Structure and/or validate one or more files.")
@click.argument('files', nargs=1, type=CLICK_FILE, help="Comma-separated list of files to structure. Do not include spaces. Filenames can include wildcards (*).")
@click.argument('--output-status-json', '-s', type=CLICK_FILE, multiple=False, help="File to output with JSON info about the states of structured and unstrutured entries.")
def structure(files_glob, status_json, ):
    click.echo("Running structure ok.")

#
