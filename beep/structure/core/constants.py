
TQDM_STRUCTURED_SUFFIX = "(structured)"
TQDM_RAW_SUFFIX = "(raw)"


# the minimal set of standard columns that must be present
# before dataframes can be converted to objects
# designed to avoid downstream problems
MINIMAL_COLUMNS_INGESTION = [
    "test_time",
    "cycle_index",
    "step_index",
    "charge_capacity",
    "discharge_capacity",
    "current",
    "voltage",
]

# A minimum set of columns that must be present in raw dataframes
# Ensures the ingestion is working correctly
# todo: these should really be used more widely, e.g. in Step and Cycle,
# todo: not just CycleContainer
MINIMUM_COLUMNS_RAW = MINIMAL_COLUMNS_INGESTION + [
    "cycle_label",
    "step_label",
    "step_counter",
    "step_counter_absolute",
    "datum"
]



# CycleContainer level config ONLY
CONTAINER_CONFIG_DEFAULT = {
    "dtypes": {
        'test_time': 'float64',              # Total time of the test
        'date_time_iso': 'datetime64',       # ISO datetime
        'cycle_index': 'int32',              # Index of the cycle
        'cycle_label': 'category',           # Label of the cycle - default="regular"
        'current': 'float32',                # Current
        'voltage': 'float32',                # Voltage
        'temperature': 'float32',            # Temperature of the cell
        'internal_resistance': 'float32',    # Internal resistance of the cell
        'charge_capacity': 'float32',        # Charge capacity of the cell
        'discharge_capacity': 'float32',     # Discharge capacity of the cell
        'charge_energy': 'float32',          # Charge energy of the cell
        'discharge_energy': 'float32',       # Discharge energy of the cell
        'step_index': 'int16',               # Index of the step (i.e., type), according to the cycler output
        'step_counter': 'int32',             # Counter of the step within cycle, according to the cycler
        'step_counter_absolute': 'int32',    # BEEP-determined step counter across all cycles
        'step_label': 'category',            # Label of the step - default is automatically determined
        'datum': 'int32',                    # Data point, an index.
    },

    # todo: cycle-level retain config is only used during interpolation
    # todo: not for retaining columns during ingestion
    # If retain is None, all columns are kept, including nonstandard
    # columns. If retain is a list, only columns in the list are kept.
    "retain": None
}

# Cycle level config ONLY
# Config mode 1: Constant n point per step within a cycle.
# k steps within a cycle will result in n*k points per cycle.
# Config for mode 2: Constant n points per step label within a cycle,
# regardless of k steps in cycle.
# for $i \in S$ step labels, $n_i$ points per step label, will result in $\sum_i n_i$ points per cycle.
# Note: for temporally disparate steps with the same steps label, strange behavior can occur.
CYCLE_CONFIG_DEFAULT = {
    "preaggregate_steps_by_step_label": False,
}

# Step level config ONLY
# For a "columns" value of None, ALL columns will be interpolated except
# for those known to be constant (e.g., cycle label)
STEP_CONFIG_DEFAULT = {
    "field_name": "voltage",
    "field_range": None,
    "columns": None,
    "resolution": 1000,
    "exclude": False,
    "min_points": 2
}