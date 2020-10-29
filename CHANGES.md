

v2020.10.29.16
--------------
* Adding asserts to check for the interval information
* Move parameter functions and switch to those functions
* Delete initialization cycle and add diagnostic intervals
* Changed throughput calculations to only have regular cycles, with a separate column for diagnostic cycles
* Fix structuring code so that diagnostic cycles are excluded from the summary
* Uploading Iris_params csv file and adding test_from_csv function for Iris parameters
* change fname for windows
* docstring update
* add filters for waveform discharge and charge
* complete tutorial testing with teardown
* tutoral working tests

v2020.10.22.15
--------------
* Add diagnostic parameters for PHEV cells
* Update interpolation for time axis (TODO: make step specific) (Thanks Chirru Gopal)
* Add documentation and better tutorial (Thanks Alex Dunn)

v2020.10.19.20
--------------
* Change the method for ensuring monotonicity and add testing 

v2020.10.14.21
--------------
* update copyright notices
* Fix test file for biologic to maccor
* Use .coveragerc - Thanks Patrick Moore

v2020.10.13.11
--------------
* Fixes for featurization bugs - Patrick Herring
* Documentation and clean up for `from_file` methods in structuring - Thanks Alex Dunn!
* Biologic to maccor converter draft - Thanks Will Powelson!


v2020.9.29.19
-------------
* Update version number to include hour so that packages can be released more than once a day
* Add missing waveform files

v2020.9.29
----------

* Changed test assertions, function description - Thanks Bruis van Vlijmen!
* Biologic to Maccor converter - Thanks Will Powelson!
* Waveforms in production protocols.

v2020.9.16
----------
* Revert unintentional change during release

v2020.9.15
----------
* Improve workflow event coverage - Patrick Moore
* Output results files from generate_protocol
* Adding file write out to the test
* Bug fixes for featurizations - Chirranjeevi Gopal
* Download big files for testing. - Daniel Schweigert
* Biologic conversion methods - Will Powelson

v2020.8.21
----------
* Add data set class to facilitate model training - Thanks to Chirranjeevi Gopal
* Create CODE_OF_CONDUCT.md - Thanks to Patrick Moore
* allow multiple hyperparam sets for a feature class
* Add Neware parsing - Thanks to Patrick Herring
* Add linting and testing to GH actions - Thanks to Joseph Montoya
* Add test for procedure with waveform

v2020.7.29
----------
* Opening up the validation limits for the xIris project
* change nose to pytest in readme
* add CI infrastructure to contributions
* adding contribution guidelines - Thanks Chirru!

v2020.7.17
----------
* Add blurb about looking for python dev
* Refactored featurize module to split up DiagnosticCycleFeatures class into separate classes for 
each set of features from different diagnostic cycles.
* Added a yaml file to set default hyperparameters for feature generation. Alternately, a hyperparameter
dict can be passed as an argument (`params_dict`) to the `from_run()` method
* Metadata for features includes hyperparameter information 

v2020.7.8
---------
* Fix missing modules in package

v2020.7.8
---------
* HPPC feature update - Thanks to Xiao Cui
* Resolve warnings during tests - Thanks to Tino Sulzer
* Change logging options in the config for local to eliminate some errors
* Add additional test to expand coverage
* Add changes for validation based on project name with test
* Add project look up for validation schema
* make corresponding changes in the final get hppc feature function, soc is changed to 8, and added some one new diffusion feature.
* a new set of helper functions to calculate the diffusion coefficient for  the 40s rests in hppc
* changed the v_diff function a bit to make it more consistent with others
* the new function will give you 54 different values of resistances based on soc and time scale
* a new helper function for calculating resistance
* deleting something
* changing soc=7 to soc=8
* Include notebook used to demonstrate the use of beep with a fast charge file
* update comments and docstring
* add validation criteria for featurize
* Update docstring for to_file method
* Add docstring for to_file method
* try pytest-cov removal
* Clean up and commenting
* Finish writing initial tests for the parser
* Creating ordered dict for the steps
* Alter connection try to avoid tests when AWS not present
* Remove repeated method
* Fix assertions
* Expand coverage of tests
* Change to optional argument for protocol directory
* Removing reptitive method
* Remove spaces around step type
* Fix step headers
* Working tests, wrong step headers
* Working version of the initialization
* Intermediate fix to some import statements
* Remove import statement
* Adding protocol schema dir
* Adding procedure directory
* Fix imports
* Refactor of protocol generation by sub-classing DashOrderedDict - Thanks to Joseph Montoya

v2020.6.2
---------
* New feature classes based on diagnostic cycles.
* Tests run within scratchDir to eliminate files being generated by tests.
* Events mode set to off if AWS secrets not accessible.
* New feature class for prediction outcomes.
* Helper module for new featurization classes.
* Automate directory creation for running tests under different environments.
* Add datatypes to arbin cycling data.

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
