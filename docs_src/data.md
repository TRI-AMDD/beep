# Cycler data requirements

BEEP automatically parses and structures data based on specific outputs from various 
battery cyclers. The following column headers marked "required" are required for downstream processing of each 
cycler. BEEP currently supports five brands of battery cyclers:

- [Arbin](#arbin)
- [Maccor](#maccor)
- [Indigo](#indigo)
- [BioLogic](#biologic)
- [Neware](#neware)


---
 
## Arbin

Arbin data files are of the form `name_of_file_CHXX.csv` with an associated metadata file `name_of_file_CHXX_Metadata.csv`


#### Cycler Data

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


#### Metadata

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


## Maccor

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


## Indigo

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


## BioLogic

BioLogic files are ASCII text files of the form `*.mpt` with matching `*.mpl` log/metadata files.

*BioLogic cycler data is currently only supported for structuring operations (e.g., ingestion via `BioLogicDatapath` analysis) and is not supported for downstream processing.*

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


## Neware

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


## Battery Archive

Battery Archive files are singular csvs matching the file pattern `*timeseries*.csv`.


| Column name (case insensitive) | Required |   Explanation  | Unit |  Data Type |
|-------------|----------|-------------|------|------------|
| `Cycle_Index ` | ✓  |  index of this cycle |   | `int` |
| `Current (A)` | ✓  | cell current  |  Amps | `float` |
| `Voltage (V)` | ✓  | cell potential  |  Volts | `float` |
| `Charge_Capacity (Ah)` | ✓  | charge capacity  |  amp-hr | `float` |
| `Discharge_Capacity (Ah)` | ✓  | discharge capacity  |  amp-hr | `float` |
| `Charge_Energy (Wh)` | ✓  | charge energy  |  watt-hr | `float` |
| `Discharge_Energy (Wh)` | ✓  | discharge energy  |  watt-hr | `float` |
| `Cell_Temperature (C)` | ✓  | temperature of the cell | °Celsius | `float` |
| `Environmental_Temperature (C)` |   | environmental temperature | °Celsius | `float` |
| `Test_Time (s)` | ✓ | test time | seconds | `float` |
| `Date_Time` | ✓ | datetime string, in `'%Y-%m-%d %H:%M:%S.%f'` format |  | `str` |


No metadata ingestion is supported for Battery Archive files at this time.
