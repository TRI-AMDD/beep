# Advanced Structuring: Unsupported cyclers


If you are using a cycler not supported by BEEP, you can still use BEEP to structure, featurize, and run models on your data! 
To do this, you simply inherit from the `BEEPDatapath` base class described in the [Structuring Tutorial](/Command%20Line%20Interface/2%20-%20structuring/) to create your own Datapath.

`BEEPDatapath` handles all structuring of battery cycler files by taking them from raw cycler output files (usually csvs or text) and converting them
into consistent interfaces for structuring.

Your custom datapath will work with BEEP's capabilities similarly to all existing cyclers/datapaths. 


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
- `DATA_TYPES`: Mapping of BEEP canoncial column name to data type, in pandas-parsable format. For example, if your cycle index should be 32-pt integer, you can include the key-value `"cycle_index": "int32"` in your `DATA_TYPES` class attribute.


### Code Example - putting it all together

Once your `from_file` method is able to extract the three requirements in the correct format, you should be able to pass those
objects to the `cls` constructor inside of `from_file`. For example:



```python
import os
import json

import pytz
import pandas as pd

from beep.structure.base import BEEPDatapath


class MyCyclerDatapath(BEEPDatapath):

    COLUMN_MAPPING = {
        "test_time (s)": "test_time",
        "cycle_index": "cycle_index",
        "current (a)": "current",
        "voltage (v)": "voltage",
        "charge_capacity (ah)": "charge_capacity",
        "discharge_capacity (ah)": "discharge_capacity",
        "charge_energy (wh)": "charge_energy",
        "discharge_energy (wh)": "discharge_energy",
        "cell_temperature (c)": "temperature",
        "date_time": "date_time",
        "steptime": "step_time",
        "stepix": "step_index"
    }

    # Columns to ignore
    COLUMNS_IGNORE = ["environment_temperature (c)"]

    # Mapping of data types for BEEP columns
    DATA_TYPES = {
        "test_time": "float64",
        "cycle_index": "int32",
        "current": "float32",
        "voltage": "float32",
        "charge_capacity": "float64",
        "discharge_capacity": "float64",
        "charge_energy": "float64",
        "discharge_energy": "float64",
        "temperature": "float32",
        "date_time": "float32",
        "step_time": "float32",
        "step_index": "int32"
    }
    
    
    @classmethod
    def from_file(cls, path, metadata_path=None):
        
        # some code to get the raw data in BEEP format
        # assuming it does not need to be further augmented
        df = pd.read_csv(path)
        df = df.drop(cls.COLUMNS_IGNORE)
        df.rename(columns=cls.COLUMN_MAPPING, inplace=True)
        
        
        # For example, adding a date_time_iso column if not already present
        df["date_time_iso"] = df["date_time"].apply(
            lambda x: x.from_timestamp().replace(tzinfo=pytz.UTC).isoformat()
        )
        
        # Cast all data types to those specified as class attrs
        for column, dtype in cls.DATA_TYPES.items():
            if column in df:
                if not df[column].isnull().values.any():
                    df[column] = df[column].astype(dtype)
        
        # Read in metadata from a separate json file, for example
        if metadata_path:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # specify all paths absolutely
        paths = {
            "raw": os.path.abspath(path),
            "metadata": os.path.abspath(metadata_path)
        }

        # Return the 3 required objects to BEEPDatapath
        return cls(df, metadata, paths)

```




### After your BEEPDatapath is working

Once your BEEPDatapath is able to load raw files using `from_file`, all of BEEP's other modules and methods should work with it like they do with any
other Datapath/cycler.

For example, structuring your BEEPDatapath requires only calling the parent `BEEPDatapath`'s `.structure` method.

For more info on the capabilities of `BEEPDatapath`, see the [Structuring Tutorial](/Command%20Line%20Interface/2%20-%20structuring/).



## Advanced usage

Your cycler may possess capabilities for data or structuring outside of base `BEEPDatapath`'s capabilities. In this case,
it may be needed to implement additional methods or override `BEEPDatapath` methods beyond `from_file`. The specific implementation will depend 
on your cycler's capabilities; however, it is recommended not to override the following methods in particular:


- `BEEPDatapath.structure`
- `BEEPDatapath.autostructure`
- `BEEPDatapath.as_dict`
- `BEEPDatapath.from_dict`

If these methods are overridden in an incompatible way, it is likely they will break further downstream BEEP tasks, such as diagnostic
structuring or featurization.