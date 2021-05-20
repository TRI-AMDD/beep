# Unsupported Cyclers - Writing your own `BEEPDatapath`


If you are using a cycler not supported by BEEP, you can still use BEEP to structure, featurize, and run models on your data! 
To do this, you simply inherit from the `BEEPDatapath` base class described [on the Advanced Tutorial](tutorial2.md).

`BEEPDatapath` handles all structuring of battery cycler files by taking them from raw cycler output files (usually csvs or text) and converting them
into consistent interfaces for structuring.


## The Simplest Case: Using `from_file`

To put your cycler data in a format BEEP can understand, inherit from the `BEEPDatapath` class and implement the `from_file` classmethod.


### Requirements

Your from file method will need to produce the following data to work correctly with BEEP.

#### 1. A dataframe of the battery cycler data, in a standard format

The dataframe should have at least the following columns, named exactly as described:

- `test_time`: Time of the test, in seconds
- `cycle_index`: Integer index of the cycle number
- `current`: Current drawn to/from battery, in amps
- `voltage`: Voltage, in volts
- `charge_capacity`: Charge capacity of the battery, in amp-hours
- `discharge_capacity`: Discharge capacity of the battery, in amp-hours
- `charge_energy`: Charge energy of the battery, in watt-hours
- `discharge_energy`: Discharge energy of the battery, in watt-hours
- `step_index`: Index integer of the charge-step, e.g., resting = 1, charging = 2, etc.
- `step_time`: amount of time spent in this charge-step, in seconds.

(Optional):

- `temperature`: Temperature of the cell itself
- `date_time`: Date time, as timestamp (ms from unix epoch)
- `date_time_iso`: Date time in UTC time zone, formatted using `.isoformat()`
- `internal_resistance`: Internal resistance of battery, in ohm

The dataframe may contain other data, if available from your cycler output.


#### 2. Metadata dictionary

All available metadata from the cycler run should be gathered by `from_file`. This can include things like:
- `barcode`
- `protocol`
- `channel_id`
- and other cycler-specific metadata.

The metadata should be a dictionary.



#### 3. Paths to raw input files

Finally, paths to all raw input files should be collected as a dictionary, mapping file type to the absolute path. For example, if each
run of your cycler requires a time series file and a metadata file, the paths dictionary would look like:


```python

paths = {
    "raw": "/path/to/raw/timeseries.csv",
    "metadata": "/path/to/metadata.json"
}
```

Note `raw` and `metadata` are special keys. While having these two exact paths is recommended, arbitrary other paths to supporting files
can be passed in the paths dictionary without any special naming convention. For example:


```python

paths = {
    "raw": "/path/to/raw/timeseries.csv",
    "metadata": None,
    "my_other_required_filetype_path": "/path/to/somefile.hd5"
}
```


### Column Mapping

To transparently keep consistent data types and column names, we recommend making the following class attributes in your `BEEPDatapath` child class:

- `COLUMN_MAPPING`: Maps raw column names to BEEP canonical names
- `COLUMNS_IGNORE`: Raw column names to ignore, if they are not needed (for example, `Environmental Temperature (C)`) 


### Code Example - putting it all together

Once your `from_file` method is able to extract the three requirements in the correct format, you should be able to pass those
objects to the `cls` constructor inside of `from_file`. For example:



```python
from beep.structure.base import BEEPDatapath


class MyCyclerDatapath(BEEPDatapath):
    
    
    @classmethod
    def from_file(cls, path, metadata_path=None):
        
        # some code to get the raw data in BEEP format
        


```