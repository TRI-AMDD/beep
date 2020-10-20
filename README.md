![Testing - main](https://github.com/TRI-AMDD/beep/workflows/Testing%20-%20main/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/TRI-AMDD/beep/badge.svg?branch=master)](https://coveralls.io/github/TRI-AMDD/beep?branch=master)

# Battery Evaluation and Early Prediction (BEEP)

BEEP is a set of tools designed to support Battery Evaluation and Early Prediction of cycle life corresponding to the research of the [d3batt program](https://d3batt.mit.edu/) and the [Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).


BEEP enables parsing and handing of electrochemical battery cycling data
via data objects reflecting cycling run data, experimental protocols,
featurization, and modeling of cycle life.  Currently beep supports 
arbin, maccor and biologic cyclers.

We are currently looking for experienced python developers to help us improve this package and implement new features.
Please contact any of the maintainers for more information.

# Table of Contents
1. [Installation](#installation)
2. [Environment](#environment)
3. [Testing](#testing)
4. [Using scripts](#using-scripts)
5. [Data requirements](#data-requirements)
6. [How to cite](#how-to-cite)

## Installation
Use `pip install beep` to install.

If you want to develop BEEP, clone the repo via git and use 
pip (or `python setup.py develop`)  for an editable install:

```bash
git clone git@github.com:ToyotaResearchInstitute/BEEP.git
cd BEEP
pip install -e .[tests]
```
## Environment
To configure the use of AWS resources its necessary to set the environment variable `BEEP_ENV`. For most users `'dev'`
is the appropriate choice since it assumes that no AWS resources are available. 
```.env
export BEEP_ENV='dev'
```
For processing file locally its necessary to configure the folder structure 
```.env
export BEEP_PROCESSING_DIR='/path/to/beep/data/'
```

## Testing
You can use pytest for running unittests. In order to run tests the environment variable
needs to be set (i.e. `export BEEP_ENV='dev'`)

```bash
pytest beep
```

## Using scripts

The standard installation procedure above should install and link console scripts
with currently available BEEP functionality.  Each BEEP script takes a JSON string
as input in order to provide flexibility and more facile automation.  They are documented
below:

### collate
The `collate` script takes no input, and operates by assuming the BEEP_PROCESSING_DIR (default `/`)
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
    "mode": "events_off",
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
in `beep.validate` on renamed files according to the output of `rename` above.
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
    "mode": "events_off",
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
    "mode": "events_off",
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
    "mode": "events_off",
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
    "mode": "events_off",
    "run_list": [34],
    "file_list": ["/data-share/features/FastCharge_2_CH29_full_model_features.json"]
}'
```
```json
{
  "file_list": ["/data-share/predictions/FastCharge_2_CH29_full_model_predictions.json"],
}
```

## Data requirements

BEEP automatically parses and structures data based on specific outputs from various 
battery cyclers. The following column headers marked "required" are required for downstream processing of each 
cycler. BEEP currently supports five brands of battery cyclers:

- [Arbin](#arbin)
- [Maccor](#maccor)
- [Indigo](#indigo)
- [BioLogic](#biologic)
- [Neware](#neware)


### Arbin

Arbin data files are of the form `name_of_file_CHXX.csv` with an associated metadata file `name_of_file_CHXX_Metadata.csv`


##### Cycler Data

| Column name (case insensitive) | Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `data_point` |   | index of this data point  |  |   `int32` |
| `test_time` |   |  time of data point relative to start | seconds  |   `float32` |
| `datetime` | ✓  |  time of data point relative to epoch time | seconds  |   `float32` |
| `step_time` |   |  elapsed time counted from the starting point of present active step |  seconds |   `float32` |
| `step_index` | ✓  | currently running step number in the active schedule  |   |   `int16` |
| `cycle_index` | ✓  | currently active test cycle number  |   |   `int32` |
| `current` | ✓  | measured value of present channel current  |  Amps |   `float32` |
| `voltage` | ✓  | measured value of present channel voltage  |  Volts |   `float32` |
| `charge_capacity` | ✓  | cumulative value of present channel charge capacity  | Amp-hr  |   `float64` |
| `discharge_capacity` | ✓  | cumulative value of present channel discharge capacity  | Amp-hr  |   `float64` |
| `charge_energy` | ✓  | cumulative value of present channel charge energy  |  Watt-hr  |   `float64` |
| `discharge_energy` | ✓  | cumulative value of present channel discharge energy   | Watt-hr  |   `float64` |
| `dv/dt` |   | the first-order change rate of voltage  |  Volts/seconds |   `float32` |
| `internal_resistance` |   | calculated internal resistance |  Ohms |   `float32` |
| `temperature` |   | cell temperature | °Celsius |   `float32` |


##### Metadata

| Field name | Required |
|------------|-------------|
| `test_id`  |  |
| `device_id`  |  |
| `iv_ch_id`  |  |
| `first_start_datetime`  |  |
| `schedule_file_name`  |  |
| `item_id`  |  |
| `resumed_times`  |  |
| `last_end_datetime`  |  |
| `databases`  |  |
| `grade_id`  |  |
| `has_aux`  |  |
| `has_special`  |  |
| `schedule_version`  |  |
| `log_aux_data_flag`  |  |
| `log_special_data_flag`  |  |
| `rowstate`  |  |
| `canconfig_filename`  |  |
| `m_ncanconfigmd5`  |  |
| `value`  |  |
| `value2`  |  |


### Maccor

Maccor files are single tabular text files matching the regex pattern `".*\\d{5,6}.*\\d{3}"`.

| Column name (case insensitive) | Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `rec#` | ✓  | data point number (index)  |   |   `int32` |
| `cyc#` | ✓  | cycle number  |   |   `int32` |
| `step` | ✓  | step number  |   |   `int16` |
| `test (sec)` | ✓  | total time elapsed  |  seconds |   `float32` |
| `step (sec)` | ✓  | time within this step   | seconds  |   `float32` |
| `amp-hr` | ✓  | charge capacity   | Amp-hr  |   `float64` |
| `watt-hr` | ✓  | charge energy  | Watt-hr  |   `float64` |
| `amps` | ✓  | channel current  |  Amps |   `float32` |
| `volts` | ✓  | channel voltage  | Volts  |   `float32` |
| `state` | ✓  | charging/discharging/etc. state of the battery  |   |   `category` |
| `es` | ✓  |   |   |   `category` |
| `dpt time` | ✓  | date and time of data point  | Date-Time  |   `str` |
| `acimp/ohms` | ✓  | AC impedance of circuit  |  Ohm |   `float32` |
| `dcir/ohms` | ✓  | DC internal resistance  |  Ohm |   `float32` |
| `wf chg cap` | ✓  |  charge capacity (based on waveform, if available)  | Amp-hh   |   `float32` |
| `wf dis cap` | ✓  |  discharge capacity (based on waveform, if available) | Amp-hr  |   `float32` |
| `wf chg e` | ✓  | charge energy (based on waveform, if available)  | Watt-hr  |   `float32` |
| `wf dis e` | ✓  | discharge energy (based on waveform, if available) | Watt-hr  |   `float32` |
| `range` | ✓  |   |   |   `uint8` |
| `var1` | ✓  |   |   |   `float16` |
| `var2` | ✓  |   |   |   `float16` |
| `var3` | ✓  |   |   |   `float16` |
| `var4` | ✓  |   |   |   `float16` |
| `var5` | ✓  |   |   |   `float16` |
| `var6` | ✓  |   |   |   `float16` |
| `var7` | ✓  |   |   |   `float16` |
| `var8` | ✓  |   |   |   `float16` |
| `var9` | ✓  |   |   |   `float16` |
| `var10` | ✓  |   |   |   `float16` |
| `var11` | ✓  |   |   |   `float16` |
| `var12` | ✓  |   |   |   `float16` |
| `var13` | ✓  |   |   |   `float16` |
| `var14` | ✓  |   |   |   `float16` |
| `var15` | ✓  |   |   |   `float16` |


### Indigo

Indigo files are single hierarchical data files (`*.h5`) with the mandatory group store field `"time_series_data"`.

| Column name (case insensitive)| Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `cell_coulomb_count_c` | ✓  |  instantaneous cell charge | Coulombs  |  |
| `cell_current_a` | ✓  |   | A  |  |
| `cell_energy_j` | ✓  |  cell energy  | Joules  |  |
| `cell_id` | ✓  | identifier of the cell  |   |  |
| `cell_power_w` |   |  instantaneous cell power  |  Watts |   |
| `cell_temperature_c` |    |   temperature of the cell |   °Celsius  |  |
| `cell_voltage_v` | ✓  | voltage of the cell  | Volts  |  |
| `cycle_count` | ✓  |  index of the cycle |   |  |
| `experiment_count` |   |  index of the experiment |   |  |
| `experiment_type` |   |   |   |  |
| `half_cycle_count` | ✓  |   |   |  |
| `system_time_us` | ✓  | test time of data point relative to epoch   |  microseconds |  |
| `time_s` |   | time elapsed since test beginning | seconds  |  |


### BioLogic

BioLogic files are ASCII text files of the form `*.mpt` with matching `*.mpl` log/metadata files.

*BioLogic cycler data is currently only supported for raw operations (e.g., ingestion via `RawCyclerRun` analysis) and is not supported for downstream processing.*

| Column name | Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `cycle number` | ✓  |  index of this cycle |   | `int` |
| `half cycle` | ✓  |   |   | `int` |
| `Ecell/V` | ✓  | cell potential  |  Volts | `float` |
| `I/mA` | ✓  | cell current  |  mAmps | `float` |
| `Q discharge/mA.h` | ✓  |  discharge capacity | mAmp-hr  | `float` |
| `Q charge/mA.h` | ✓  |  charge capacty | mAmp-hr  | `float` |
| `Energy charge/W.h` | ✓  | charge energy  | Watt-hr  | `float` |
| `Energy discharge/W.h` | ✓  | discharge energy  |  Watt-hr | `float` |

Various other fields in BioLogic data or metadata files are not required.


### Neware

Neware files are singular `*.csv` files.

*Note: Neware files use non-standard csv formatting; some fields may require further processing or structuring before input to `beep`.*


| Column name | Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `Record ID` | ✓  |  index of this data point |   | `int32` |
| `Realtime` | ✓  |  date-time format for this point  |   |  |
| `Time(h:min:s.ms)` | ✓  | recorded time for this point  | seconds  | `float32` |
| `Step ID` | ✓  | index of this step  |   | `int16` |
| `Cycle ID` | ✓  | index of this cycle  |   | `int32` |
| `Current(mA)` | ✓  | cell current  |  mAmps | `float32` |
| `Voltage(V)` | ✓  | cell voltage  |  Volts | `float32` |
| `Capacitance_Chg(mAh)` | ✓  |  charge capacity | mAmp-hr  | `float64` |
| `Capacitance_DChg(mAh)` | ✓  | discharge capacity  | mAmp-hr  | `float64` |
| `Engy_Chg(mWh)` | ✓  | charge energy  | mWatt-hr  | `float64` |
| `Engy_DChg(mWh)` | ✓  | discharge energy | mWatt-hr  | `float64` |
| `DCIR(O)` | ✓  |  DC internal resistance |   | `float32` |
| `Capacity(mAh)` | ✓  |   | mAmp-hr  |  |
| `Capacity Density(mAh/g)` | ✓  |   |  mAmp-hr/gram |  |
| `Energy(mWh)` | ✓  |   | mWatt-hr  |  |
| `CmpEng(mWh/g)` | ✓  |   | mWatt-hr/gram  |  |
| `Min-T(C)` | ✓  | mimumum cell temperature  | °Celsius  |  |
| `Max-T(C)` | ✓  | max cell temperature  | °Celsius  |  |
| `Avg-T(C)` | ✓  | average cell temperature  |  °Celsius |  |
| `Power(mW)` | ✓  | instantaneous power  |  mWatt |  |
| `dQ/dV(mAh/V)` | ✓  | differential capacity  | mAmp-hr/Volt  |  |
| `dQm/dV(mAh/V.g)` | ✓  | differential capacity density  | mAmp-hr/Volt-gram  |  |
| `Temperature(C)` | ✓  |  temperature (alternate sensor) |  °Celsius  | `float32` |



## How to cite
If you use BEEP, please cite this article:

> P. Herring, C. Balaji Gopal, M. Aykol, J.H. Montoya, A. Anapolsky, P.M. Attia, W. Gent, J.S. Hummelshøj, L. Hung, H.-K. Kwon, P. Moore, D. Schweigert, K.A. Severson, S. Suram, Z. Yang, R.D. Braatz, B.D. Storey, SoftwareX 11 (2020) 100506.
[https://doi.org/10.1016/j.softx.2020.100506](https://doi.org/10.1016/j.softx.2020.100506)

