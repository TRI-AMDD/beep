[![Build Status](https://travis-ci.com/TRI-AMDD/beep.svg?branch=master)](https://travis-ci.com/TRI-AMDD/beep)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/beep/badge.svg?branch=master)](https://coveralls.io/github/TRI-AMDD/beep?branch=master)

# Summary

Beep is software designed to support Battery Estimation and Early Prediction
of cycle life corresponding to the research of the
[d3batt program](https://d3batt.mit.edu/) and the 
[Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).


Beep enables parsing and handing of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life.  Currently beep supports 
arbin and maccor cyclers.

## Installation
Use `pip install beep` to install.

If you want to develop beep, clone the repo via git and use 
pip (or `python setup.py develop`)  for an editable install:

```bash
git clone git@github.com:ToyotaResearchInstitute/BEEP.git
cd BEEP
pip install -e .
```

## Testing
You can use nose or pytests for running unittests (use `pip install nose` 
to install nose if not installed)

```bash
nosetests beep
```

Note that the editable install (as prescribed above), is necessary
for nosetests.

## Using scripts

The standard installation procedure above should install and link console scripts
with currently available BEEP functionality.  Each beep script takes a JSON string
as input in order to provide flexibility and more facile automation.  They are documented
below:

### collate
The `collate` script takes no input, and operates by assuming the BEEP_ROOT (default `/`)
has subdirectories `/data-share/raw_cycler_files` and `data-share/renamed_cycler_files/FastCharge`.

The script moves files from the `/data-share/raw_cycler_files` directory, parses the metadata,
and renames them according to a combination of protocol, channel number, and date, placing them in
`/data-share/renamed_cycler_files`.

The script output is a json string that contains the following fields:

* `fid` - The file id used internally for renaming
* `filename` - full paths for raw cycler filenames
* `strname` - the string name associated with the file (i. e. scrubbed of `csv`)
* `file_list` - full paths for the new, renamed, cycler files
* `protocol` - the cycling protocol corresponding to each file
* `channel_no` - the channel number corresponding to each file
* `date` - the date corresponding to each file

Example:
```bash
$ collate
```
```json
{
    "mode": "run",
    "fid": [0, 
            1, 
            2],
    "strname": ["2017-05-09_test-TC-contact", 
                "2017-08-14_8C-5per_3_47C", 
                "2017-12-04_4_65C-69per_6C"],
    "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"],
    "protocol": [null, 
               "8C(5%)-3.47C", 
               "4.65C(69%)-6C"],
    "date": ["2017-05-09", 
             "2017-08-14", 
             "2017-12-04"],
    "channel_no": ["CH33", 
                   "CH44", 
                   "CH29"],
    "filename": ["/data-share/raw_cycler_files/2017-05-09_test-TC-contact_CH33.csv", 
                 "/data-share/raw_cycler_files/2017-08-14_8C-5per_3_47C_CH44.csv", 
                 "/data-share/raw_cycler_files/2017-12-04_4_65C-69per_6C_CH29.csv"]
}
```

### validate
The validation script, `validate`, runs the validation procedure contained
in beep.validate on renamed files according to the output of `rename` above.
It also updates a general json validation record in `/data-share/validation/validation.json`.

The input json must contain the following fields

* `file_list` - the list of filenames to be validated
* `mode` - mode for events i.e. 'test' or 'run'
* `run_list` - list of run_ids for each of the files, used by the database for linking data

The output json will have the following fields:

* `validity` - a list of validation results, e. g. `["valid", "valid", "invalid"]`
* `file_list` - a list of full path filenames which have been processed

Example:
```bash
$ validate '{
    "mode": "run",
    "run_list": [1, 20, 34],
    "strname": ["2017-05-09_test-TC-contact", 
                "2017-08-14_8C-5per_3_47C", 
                "2017-12-04_4_65C-69per_6C"],
    "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"],
    "protocol": [null, 
               "8C(5%)-3.47C", 
               "4.65C(69%)-6C"],
    "date": ["2017-05-09", 
             "2017-08-14", 
             "2017-12-04"],
    "channel_no": ["CH33", 
                   "CH44", 
                   "CH29"],
    "filename": ["/data-share/raw_cycler_files/2017-05-09_test-TC-contact_CH33.csv", 
                 "/data-share/raw_cycler_files/2017-08-14_8C-5per_3_47C_CH44.csv", 
                 "/data-share/raw_cycler_files/2017-12-04_4_65C-69per_6C_CH29.csv"]
}'
```
```json
{"validity": ["invalid",
              "invalid",
              "valid"],
 "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
               "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv", 
               "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"],
}
```

### structure

The `structure` script will run the data structuring on specified filenames corresponding
to validated raw cycler files.  It places the structured datafiles in `/data-share/structure`.

The input json must contain the following fields:
* `file_list` - a list of full path filenames which have been processed
* `validity` - a list of boolean validation results, e. g. `[True, True, False]`
* `mode` - mode for events i.e. 'test' or 'run'
* `run_list` - list of run_ids for each of the files, used by the database for linking data

The output json contains the following fields:

* `invalid_file_list` - a list of invalid files according to the validity
* `file_list` - a list of files which have been structured into processed_cycler_runs

Example:
```bash
$ structure '{
    "mode": "run",
    "run_list": [1, 20, 34],
    "validity": ["invalid", "invalid", "valid"], 
    "file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv", 
                  "/data-share/renamed_cycler_files/FastCharge/FastCharge_2_CH29.csv"]}'
```
```json
{
  "invalid_file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
                       "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv"], 
  "file_list": ["/data-share/structure/FastCharge_2_CH29_structure.json"],
}
```

### featurize
The `featurize` script will generate features according to the methods
contained in beep.generate_features.  It places output files corresponding to 
features in `/data-share/features/`.

The input json must contain the following fields

* `file_list` - a list of processed cycler runs for which to generate features
* `mode` - mode for events i.e. 'test' or 'run'
* `run_list` - list of run_ids for each of the files, used by the database for linking data

The output json file will contain the following:

* `file_list` - a list of filenames corresponding to the locations of the features

Example:
```bash
$ featurize '{
    "mode": "run",
    "run_list": [1, 20, 34],
    "invalid_file_list": ["/data-share/renamed_cycler_files/FastCharge/FastCharge_0_CH33.csv", 
                          "/data-share/renamed_cycler_files/FastCharge/FastCharge_1_CH44.csv"], 
    "file_list": ["/data-share/structure/FastCharge_2_CH29_structure.json"]
}'
```
```json
{
  "file_list": ["/data-share/features/FastCharge_2_CH29_full_model_features.json"]}
```

### run_model
The `run_model` script will generate a model and create predictions
based on the features previously generated by the generate_features.
It stores its outputs in `/data-share/predictions/`

The input json must contain the following fields
* `file_list` - list of files corresponding to model features
* `mode` - mode for events i.e. 'test' or 'run'
* `run_list` - list of run_ids for each of the files, used by the database for linking data

The output json will contain the following fields
* `file_list` - list of files corresponding to model predictions

Example:
```bash
$ run_model '{
    "mode": "run",
    "run_list": [34],
    "file_list": ["/data-share/features/FastCharge_2_CH29_full_model_features.json"]
}'
```
```json
{
  "file_list": ["/data-share/predictions/FastCharge_2_CH29_full_model_predictions.json"],
}
```
