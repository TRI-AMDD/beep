## Conversion file schema for cyclers

This folder is intended to contain rough documentation of data columns and
metadata fields of different instruments, along with some data used for the
conversion of the columns/metadata fields to a common set in the invocation of
of the beep.structure.RawCyclerRun object.  Schema may be represented in json or yaml.

In each schema, the following may be specified:

* **file_pattern**: a regex corresponding to how the file name should
look, e. g. *.csv, or biologic.\d+.txt
* **metadata_fields**: a set of key-value pairs that define how the raw parsed
metadata should be remapped. The key is the raw metadata field and the
value is the processed metadata field
* **data_columns**: a set of key-value pairs that define how the raw parsed data
column names should be remapped, similar to the above metadata fields, except to
be used on a parsed dataframe

Example:

```yaml
file_pattern: "instr.prefix*"
metadata_fields:
    my_instrument's_crazy_test_id: test_id
    my_instrument's_crazy_device_id: device_id
    my_instrument's_nonstandard_metadata_field: _custom_field_1
data_columns:
    data_point: data_point

```

Currently supported formats:

* **Arbin**
* **MACCOR**

Unsupported **biologic** and **pec** formats might be supported in the future, 
but are outside of the scope of this project for now.