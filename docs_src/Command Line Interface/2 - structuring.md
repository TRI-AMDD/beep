# Structure


The `beep structure` command takes in *N* raw battery cycler files (mostly text or csv) and produces *N* structured data json files. 

![cli_structuring](../static/op_graphic_structuring.png)

The structured json files can be loaded either with the BEEP python `BEEPDatapath` interface (see [Advanced Structuring](/tutorial2/)) or with subsequent CLI commands such as `beep featurize` (see [CLI - Featurize](/Command%20Line%20Interface/3%20-%20featurize/)).



## Structuring help dialog

```
Usage: beep structure [OPTIONS] [FILES]...

  Structure and/or validate one or more files. Argument is a space-separated
  list of files or globs.

Options:
  -o, --output-filenames PATH     Filenames to write each input filename to.
                                  If not specified, auto-names each file by
                                  appending`-structured` before the file
                                  extension inside the current working dir.
  -d, --output-dir DIRECTORY      Directory to dump auto-named files to. Only
                                  works if--output-filenames is not specified.
  -p, --protocol-parameters-dir DIRECTORY
                                  Directory of a protocol parameters files to
                                  use for auto-structuring. If not specified,
                                  BEEP cannot auto-structure. Use with
                                  --automatic.
  -v, --v-range <FLOAT FLOAT>...  Lower, upper bounds for voltage range for
                                  structuring. Overridden by auto-structuring
                                  if --automatic.
  -r, --resolution INTEGER        Resolution for interpolation for
                                  structuring. Overridden by auto-structuring
                                  if --automatic.
  -n, --nominal-capacity FLOAT    Nominal capacity to use for structuring.
                                  Overridden by auto-structuring if
                                  --automatic.
  -f, --full-fast-charge FLOAT    Full fast charge threshold to use for
                                  structuring. Overridden by auto-structuring
                                  if --automatic.
  -c, --charge-axis TEXT          Axis to use for charge step interpolation.
                                  Must be found inside the loaded dataframe.
                                  Can be used with --automatic.
  -x, --discharge-axis TEXT       Axis to use for discharge step
                                  interpolation. Must be found inside the
                                  loaded dataframe. Can be used with--
                                  automatic.
  -b, --s3-bucket TEXT            Expands file paths to include those in the
                                  s3 bucket specified. File paths specify s3
                                  keys. Keys can be globbed/wildcarded. Paths
                                  matching local files will be prioritized
                                  over files with identical paths/globs in s3.
                                  Files will be downloaded to CWD.
  --automatic                     If --protocol-parameters-path is specified,
                                  will automatically determine structuring
                                  parameters. Will override all manually set
                                  structuring parameters.
  --validation-only               Skips structuring, only validates files.
  --no-raw                        Does not save raw cycler data to disk. Saves
                                  disk space, but prevents files from being
                                  partially restructued.
  --s3-use-cache                  Use s3 cache defined with environment
                                  variable BEEP_S3_CACHE instead of
                                  downloading files directly to the CWD.
  --help                          Show this message and exit.

```