# Python tutorial 2: Next steps

Here you'll find more info about creating and using beep to do your own custom cycler analysie.


- [`BEEPDatapath` - One object for ingestion, structuring, and validation](#beepdatapath)
- [Cycler not supported? Try making your own `BEEPDatapath`](#making-your-own-beepdatapath)
- [Batch functions for structuring](#batch-functions-for-structuring)
- [Featurization](#featurization)
- [Running and analyzing models](#running-and-analyzing-models)



## `BEEPDatapath` 
### One class for ingestion, structuring, and validation

`BEEPDatapath` is an abstract base class that can handle ingestion, structuring, and validation for many types of cyclers. A datapath
object represents a complete processing pipeline for battery cycler data.

Each cycler has it's own `BEEPDatapath` class:

- `ArbinDatapath`
- `MaccorDatapath`
- `NewareDatapath`
- `IndigoDatapath`
- `BiologicDatapath`

**All these datapaths implement the same core methods, properties, and attributes, listed below:**

### Methods for loading and serializing battery cycler data

#### `*Datapath.from_file(filename)`
Classmethod to load a raw cycler output file (e.g., a csv) into a datapath object. Once loaded, you can validate or structure the file.
  
```python
# Here we use ArbinDatapath as an example
from beep.structure import ArbinDatapath

datapath = ArbinDatapath.from_file("my_arbin_file.csv")

```

#### `*Datapath.to_json_file(filename)`
Dump the current state of a datapath to a file. Can be later loaded with `from_json_file`.
  
```python
from beep.structure import NewareDatapath

datapath = NewareDatapath.from_file("/path/to/my_raw_neware_file")

# do some operations
...

# Write the processed file to disk, which can then be loaded.
datapath.to_json_file("my_processed_neware_data.json")
```


#### `*Datapath.from_json_file(filename)`
Classmethod to load a processed cycler file (e.g., a previously structured Datapath) into a datapath object.  
  
```python
from beep.structure import MaccorDatapath

datapath = MaccorDatapath.from_json_file("my_previously_serialized_datapath.json")
```


#### `*Datapath(data, metadata, paths=None, **kwargs)`: Initialize any cycler from the raw data (given as a pandas dataframe) and metadata (given as a dictionary). Paths can be included to keep track of where various cycler files are located. **Note: This is not the recommended way to create a `BEEPDatapath`, as `data` and `metadata` must have specific formats to load and structure correctly.


### Validation and structuring with `BEEPDatapath`s

#### `*Datapath.validate()`
Validate your raw data. Will return true if the raw data is valid for your cycler (i.e., can be structured successfully).

```python
from beep.structure import IndigoDatapath


datapath = IndigoDatapath.from_file("/path/to/my_indigo_file")

is_valid = datapath.validate()

print(is_valid)

# Out:
# True or False
```

#### `*Datapath.structure(*args)`
Interpolate and structure your data using specified arguments. Once structured, your `BEEPDatapath` is able to access things like the diagnostic summary, interpolated cycles, cycle summary, diagnostic summary, cycle life, and more (see [Analysis and attributes of core attributes of `BEEPDatapath`](#analysis-and-core-attributes-of-beepdatapath))

```python

from beep.structure import ArbinDatapath

datapath = ArbinDatapath.from_file("my_arbin_file.csv")

# Structure your data by manually specifying parameters.
datapath.structure(v_range=[1.2, 3.5], nominal_capacity=1.2, full_fast_charge=0.85)

```

#### `*Datapath.autostructure()`
Run structuring using automatically determined parameters. BEEP can automatically detect the structuring parameters based on your raw data.


```python
from beep.structure import BiologicDatapath


datapath = BiologicDatapath.from_file("path/to/my/biologic_data_file")

# Automatically determines structuring parameters and structures data
datapath.autostructure()
```


### Analysis and core attributes of `BEEPDatapath`


#### `*Datapath.paths`

Access all paths of files related to this datapath. `paths` is a simple mapping of `{file_description: file_path}` which holds the paths of **all** files related to this datapath, including raw data, metadata, EIS files, and structured outputs.



#### `*Datapath.raw_data`

The raw data, loaded into a standardized dataframe format, of this datapath's battery cycler data.


#### `*Datapath.metadata`

An object holding all metadata for this datapath's cycler run.


#### `*Datapath.structured_data`

The structured (interpolated) data, as a dataframe. The format is similar to that of `.raw_data`. The datapath must be structured before this attribute is available.


#### `*Datapath.structured_summary`

A summary of the structured cycler data, as a dataframe. The datapath must be structured before this attribute is available.


#### `*Datapath.diagnostic_data`

The structured (interpolated) data for diagnostic cycles, as a dataframe. The format is similar to that of `.structured_data`. The datapath must be structured before this attribute is available.


#### `*Datapath.diagnostic_summary`

A summary of the structured diagnostic cycle data, as a dataframe. The datapath must be structured before this attribute is available.




---


## Making your own `BEEPDatapath`
#### If your cycler is not already supported by BEEP


## Batch functions for structuring



## Structuring legacy BEEP files



## Featurization

More documentation for featurization coming soon!



## Running and analyzing models

More documentation for running models coming soon!