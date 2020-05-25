

v2020.5.24
----------

* Address Issue #45 - Failing tests with NoCredential Error
* Change to more descriptive name of the data directory and add defaults
* Changing the exception catch to be broad, exceptions change depending on environment
* Make creation of directories automatic
* Add validation criteria for maccor files to have minimum cycle number
* Changing the exception catch to be broad, exceptions change depending on environment
* Adding data types to raw arbin data
* Rough draft of features from diagnostic cycles - Thanks to Chirranjeevi Gopal
* include truncated PreDiag structure file for tests
* add lmfit to requirements + test fix
* updates to class variables

v2020.5.16
----------
* Adding test for time base interpolation
* Add check to ensure that charge cycle interpolation on charge capacity is uniform
* Change the interpolation axis to be the max range across the run
* generate maccor waveform file from power waveforms
* Change to single feature object for better generalization
* Use ABC abstract class for more explicit rules
* Refactor featurization into class objects


v2020.5.5
---------
* Remove code that was deleting the charge interpolation and interpolate on capacity
* Increasing the number of retries
* Adding retry wrapper to get secret for events
* power waveform generation from velocity profiles - Thanks to Chirru!

v2020.4.25
----------
* Add retry logic for logger when NoCredentialsError is encountered
* Change logging to stdout to cleaner method
* Add typing for reloading of the structured file
* Add .yaml file with the data types for each of the data frames
* Cast data types for all data frames in the structure

v2020.4.20
----------
* Add newline to put_record data
* Capitalize protocol generator service name
* Adding instructions to setup testing in README

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