# Overview

The beep base command specifies options for creating metadata and logging for all subcommands.

This page is a general overview of options that are common among *any* beep subcommand. You can expect 
options on this page to pertain to basically any beep CLI operation's inputs, outputs, and file formats.




## Basics


The BEEP CLI can be used like:

```bash
$: beep <options> <subcommand>
```

Options for the base `beep` command are specified before the subcommand. All beep subcommands take at least one file as input and return one or more files as output.


Beep has six subcommands:

- [`beep structure`](/Command%20Line%20Interface/2%20-%20structuring/): Parse, interpolate, clean, and standardize a wide range of battery cycler output files. 
- [`beep featurize`](/Command%20Line%20Interface/3%20-%20featurize/): Generate features for learning from structured files.
- [`beep train`](/Command%20Line%20Interface/4%20-%20train/): Train a machine learning model based on features.
- [`beep predict`](/Command%20Line%20Interface/5%20-%20predict/): Predict battery degradation based on learning features and a previously trained model. 
- [`beep protocol`](/Command%20Line%20Interface/6%20-%20protocol/): Generate cycler protocol from pre-made templates for a wide range of cyclers.
- [`beep inspect`](/Command%20Line%20Interface/7%20-%20inspect/): Visually inspect and debug beep files on disk.

For more info on any command or the base command, simply pass `--help` as an option.


The help dialog for `beep` base command looks like:

```bash 
$: beep --help

Usage: beep [OPTIONS] COMMAND [ARGS]...

  Base BEEP command.

Options:
  -l, --log-file FILE            File to log formatted json to. Log will still
                                 be output in human readable form to stdout,
                                 but if --log-file is specified, it will be
                                 additionally logged to a jsonl (json-lines)
                                 formatted file.
  -r, --run-id INTEGER           An integer run_id which can be optionally
                                 assigned to this run. It will be output in
                                 the metadata status json for any subcommand
                                 if the status json is enabled.
  -t, --tags TEXT                Add optional tags to the status json
                                 metadata. Can be later used forlarge-scale
                                 queries on database data about sets of BEEP
                                 runs. Example:'experiments_for_kristin'.
  -s, --output-status-json FILE  File to output with JSON info about the
                                 states of files which have had any beep
                                 subcommand operationrun on them (e.g.,
                                 structuring). Contains comprehensiveinfo
                                 about the success of the operation for all
                                 files.1 status json = 1 operation.
  --halt-on-error                Set to halt BEEP if critical featurization
                                 errors are encountered on any file with any
                                 featurizer. Otherwise, logs critical errors
                                 to the status json.
  --help                         Show this message and exit.

Commands:
  featurize  Featurize one or more files.
  predict    Run a previously trained model to predict degradation...
  protocol   Generate protocol for battery cyclers from a csv file input.
  structure  Structure and/or validate one or more files.
  train      Train a machine learning model using all available data and...

```



## Output streams

The beep base command options are used for specifying if and where to output the metadata and status of any CLI operation.

Human-readable output will always be logged to stdout, for example:

```
2021-09-21 16:14:43 INFO     Structuring 1 files
2021-09-21 16:14:43 DEBUG    Hashing file '/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv' to MD5
2021-09-21 16:14:43 INFO     File 1 of 1: Reading raw file /beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv from disk...
2021-09-21 16:14:44 INFO     File 1 of 1: Validating: /beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv according to schema file '/beep/beep/validation_schemas/schema-arbin-lfp.yaml'
2021-09-21 16:14:44 INFO     File 1 of 1: Validated: /beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv
2021-09-21 16:14:44 INFO     File 1 of 1: Structuring: Read from /beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv
2021-09-21 16:14:44 INFO     Beginning structuring along charge axis 'charge_capacity' and discharge axis 'voltage'.
2021-09-21 16:15:21 INFO     File 1 of 1: Structured: Written to /beep/beep/CLI_TEST_FILES_FEATURIZATION/tmp.json.gz
2021-09-21 16:15:21 INFO     Structuring report:
2021-09-21 16:15:21 INFO        Succeeded: 1/1
2021-09-21 16:15:21 INFO        Invalid: 0/1
2021-09-21 16:15:21 INFO        Failed: 0/1
```




But other output streams are also available:

### `--log-file`

Machine-readable json log file to write. If not specified, no log file will be created. Example:

```
{"time": "2021-09-21 16:13:48,938", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": "Structuring 1 files"}
{"time": "2021-09-21 16:13:48,939", "level": "DEBUG", "process": "67214", "module": "cmd", "func": "structure", "msg": "Hashing file '/beep/beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json' to MD5"}
{"time": "2021-09-21 16:13:49,228", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": "File 1 of 1: Reading raw file /beep/beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json from disk..."}
{"time": "2021-09-21 16:13:50,390", "level": "ERROR", "process": "67214", "module": "cmd", "func": "structure", "msg": "File 1 of 1: Failed/invalid: (EmptyDataError): /beep/beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json"}
{"time": "2021-09-21 16:13:50,391", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": "Structuring report:"}
{"time": "2021-09-21 16:13:50,391", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": " Succeeded: 0/1"}
{"time": "2021-09-21 16:13:50,391", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": " Invalid: 1/1"}
{"time": "2021-09-21 16:13:50,391", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": "         - /beep/beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json"}
{"time": "2021-09-21 16:13:50,391", "level": "INFO", "process": "67214", "module": "cmd", "func": "structure", "msg": " Failed: 0/1"}
{"time": "2021-09-21 16:14:43,291", "level": "INFO", "process": "67264", "module": "cmd", "func": "structure", "msg": "Structuring 1 files"}
{"time": "2021-09-21 16:14:43,291", "level": "DEBUG", "process": "67264", "module": "cmd", "func": "structure", "msg": "Hashing file '/beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv' to MD5"}
{"time": "2021-09-21 16:14:43,385", "level": "INFO", "process": "67264", "module": "cmd", "func": "structure", "msg": "File 1 of 1: Reading raw file /beep/beep/tests/test_files/2017-12-04_4_65C-69per_6C_CH29.csv from disk..
```

### `--output-status-json`
JSON file to write containing comprehensive structured metadata about any operation and all of its sub-operations. If not specified, no status json will be written. Example:

```
{
  "op_type": "featurize",
  "feature_matrix": {
    "created": true,
    "traceback": null,
    "output": "/beep/beep/CLI_TEST_FILES_FEATURIZATION/features.json.gz"
  },
  "files": {
    "/beep/beep/CLI_TEST_FILES_FEATURIZATION/PreDiag_000440_0000FB_structure.json": {
      "walltime": 8.546396970748901,
      "output": null,
      "processed_md5_chksum": "5848d8598584e45addfa8129bb078d95",
      "featurizers": {
        "HPPCResistanceVoltageFeatures": {
          "output": null,
          "valid": true,
          "featurized": true,
          "walltime": 1.2403650283813477,
          "traceback": null,
          "subop_md5_chksum": null
        },
        "DeltaQFastCharge": {
          "output": null,
          "valid": true,
          "featurized": true,
          "walltime": 0.05008506774902344,
          "traceback": null,
          "subop_md5_chksum": null
        },
        "DiagnosticSummaryStats": {
          "output": null,
          "valid": true,
          "featurized": true,
          "walltime": 0.19507122039794922,
          "traceback": null,
          "subop_md5_chksum": null
        },
        "CycleSummaryStats": {
          "output": null,
          "valid": true,
          "featurized": true,
          "walltime": 0.013413190841674805,
          "traceback": null,
          "subop_md5_chksum": null
        }
      }
    },
...
  "metadata": {
    "beep_verison": "2021.8.2.15",
    "op_datetime_utc": "2021-09-04 00:40:12",
    "run_id": null,
    "tags": []
  }
}

```

Any one beep command (e.g., `beep structure *`), regardless of how many files it intakes or generates, will always produce exactly one status json if `--output-status-json` is defined.


## Fault-tolerance

### `--halt-on-error`

By default, BEEP runs all operations in a fault-tolerant manner. This means that if the CLI command syntax is valid, but internally an operation or sub-operation fails, the process will
return successful. 

To disable this behavior, which will cause *any* error in any operation or sub-operation to fail the entire command use the `--halt-on-error` flag.


## Extra metadata and run-tracking with status json

Running many experiments can make it difficult to keep track of which input and output files correspond to which experiment. Data about input files and output files is kept in 
the status json, but for further tracking there are two arguments which can be specified:

### `--run-id`
An integer run_id to associate with this operation. The `run-id` is recorded in the `metadata` field of any operation in its status json.


### `--tags`
A list of string tags to associate with this operation. The `tags` are recorded in the `metadata` field of any operation in its status json.


An example of a status json containing a user run id and user tags:


```
# in status json output
...
  "metadata": {
    "beep_verison": "2021.8.2.15",
    "op_datetime_utc": "2021-09-04 00:40:12",
    "run_id": 234,
    "tags": ["my_tag_1", "TRI_experiments_2021", "debugging"]
  }
```


## Controlling compression and output file formats

Serialization in `beep` is done by the [`monty` library](https://guide.materialsvirtuallab.org/monty/); to use compression on any output
files, status files, or intermediate files in any beep subcommand, append `.gz` to the end of the output filename(s).

For example:


```shell
# For example, write our status json to a regular (uncompressed) json file
# And write our feature matrix output artifact to a gzipped json file

$: beep -s status.json featurize * outputFeatureMatrix.json.gz
```

Although they are not officially supported, other compression methods (such as `.bz2`) and file formats (`.yaml`) may be serialized to/from `beep` if they are supported
by the current version of `monty`. 
