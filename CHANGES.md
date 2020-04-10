

v2020.4.10
----------
* Change the paused column in the summary data to be int type
* Reduce actual peak memory usage and compute time - Thanks to Linda Hung

v2020.4.9
---------
* Reduce memory and reduce warnings - Thanks to Linda Hung

v2020.4.8
---------
* Add print statements for stdout only in the stage enviroment
* Add test for events using secrets manager
* Switch over the kinesis event stream name to be pulled from secrets manager


v2020.4.5
---------
* Add function to determine if there is a pause in the run
* Change method for interpolation of the HPPC cycle to be based on local voltage range

v2020.3.31
----------
* Add appveyor integration and fix path problems to allow windows compatibility - Thanks to Joseph Montoya
* Update the rest to increment by 12 min per protocol
* add cumulative capacity to summary stats
* Update the template file to change lower safety voltage to 2.0V
* add charge energy and discharge energy to the biologic parser
* add method for biologic file parsing

v2020.3.19
----------
* add cumulative capacity to summary stats
* add biologic parsing functionality (beta)
* add coverage, travis to CI
* updates to batch processing

v2020.3.8
---------
* Update the key for looking up diagnostic parameters to be `diagnostic_parameter_set`
* Fix for the diagnostic summary so that cycle_type is correctly assigned

v2020.3.5
---------
* add waveform step as placeholder
* interpolate `date_time_iso` for diagnostic cycles

v2020.3.2
---------
* Update tests to be standalone
* Added local logs directory

v2020.2.28
----------
* update README.md
* add administration doc
* add step index counter for diagnostic cycles

v2020.2.22a
----------
* update description, add testing back

v2020.2.22
----------
* change package name to beep from beep-ep (thank you Ben J. Lindsay!)

v2020.2.21
----------
* update invoke tasks
* update version