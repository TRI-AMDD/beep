import json

import numpy as np
from monty.io import zopen
import os
from typing import Union
from beep import logger
from beep.structure.core.run import Run
from beep.structure.core.cycles_container import CyclesContainer
from beep.structure.diagnostic import DiagnosticConfig
from beep.structure.core.interpolate import CONTAINER_CONFIG_DEFAULT
from beep.structure.core.constants import TQDM_STRUCTURED_SUFFIX, \
    TQDM_RAW_SUFFIX
import pandas as pd


def load_json_safe(filename):
    """
    Safely load a json object which may be zipped or not, but
    do not uncompress it with monty.
    """
    with zopen(filename, "r") as f:
        d = json.load(f)
    return d


def load_run_from_ProcessedCyclerRun_file(
        filename: Union[str, os.PathLike]
) -> Run:
    """
    Load a Run from a legacy ProcessedCyclerRun file.
    Note that since processed cycler runs did not contain raw data,
    they cannot be restructured.

    Args:
        filename (str): Path to the ProcessedCyclerRun file.

    Returns:
        Run: The Run loaded from the file.
    """
    d = load_json_safe(filename)
    if d.get("@class", None) != "ProcessedCyclerRun":
        logger.error(
            "Mismatching ProcessedCyclerRun class! The file json is malformed.")
    metadata = {}
    for k in ("barcode", "protocol", "channel_id"):
        if k in d:
            metadata[k] = d[k]

    metadata["from_legacy"] = True

    dfi = pd.DataFrame(d["cycles_interpolated"])
    dfdi = pd.DataFrame(d["diagnostic_interpolated"])
    dfs = pd.DataFrame(d["summary"])
    dfds = pd.DataFrame(d["diagnostic_summary"])

    diagnostic = {}
    # typically, these files have cycle_type listed
    if "cycle_type" in dfds.columns:
        for i, df in dfds.groupby("cycle_index"):
            cycle_type = df["cycle_type"].unique()[0]
            if cycle_type in diagnostic:
                diagnostic[cycle_type].add(i)
            else:
                diagnostic[cycle_type] = {i}
    else:
        logger.warning("No cycle type column found in diagnostic summary; could not load diagnostic.")
    dc = DiagnosticConfig(diagnostic) if diagnostic else None

    df_all = pd.concat([dfi, dfdi])

    # Account for step_index being called step_code now
    if "step_index" in df_all.columns:
        df_all.rename(columns={"step_index": "step_code"}, inplace=True)

    # Most legacy PCRs have the step_type, but
    # only sometimes the step_code counter.
    # We can't use CycleContainer.from_dataframe for this reason
    if "step_type" in df_all.columns:
        counter = df_all["step_type"]
    else:
        counter = df_all["step_code"]
    df_all["step_counter_absolute"] = counter.ne(counter.shift()).cumsum()
    df_all["step_counter"] = np.nan

    for cix in df_all.cycle_index.unique():
        step_types = df_all.loc[df_all.cycle_index == cix, "step_type"]
        df_all.loc[df_all.cycle_index == cix, "step_counter"] = step_types.ne(
            step_types.shift()).cumsum()

    # Convert any other columns to Run format
    df_all.rename(
        columns={"cycle_type": "cycle_label", "step_type": "step_label"},
        inplace=True
    )
    df_all["cycle_label"] = df_all["cycle_label"].fillna("regular")

    dtypes = {k: v for k, v in CONTAINER_CONFIG_DEFAULT["dtypes"].items() if
              k in df_all.columns}
    df_all = df_all.astype(dtypes, errors="ignore")
    structured_cc = CyclesContainer.from_dataframe(df_all,
                                                   tqdm_desc_suffix=TQDM_STRUCTURED_SUFFIX)

    r = Run(
        raw_cycle_container=None,
        structured_cycle_container=structured_cc,
        metadata=metadata,
    )
    # Avoid diagnostic trying to set any value on raw data
    # when setting diagnostic with the Run method
    r._diagnostic = dc
    r.summary_regular = dfs
    r.summary_diagnostic = dfds

    logger.warning(
        "Run loaded from legacy ProcessedCyclerRun: Data views may be misordered when using .data calls.")
    logger.warning(
        "Run loaded from legacy ProcessedCyclerRun: Columns and dtypes not rigorously enforced; downstream processing may throw errors.")
    logger.warning(
        "Run loaded from legacy ProcessedCyclerRun: Step/Cycle labels may be incorrect, no new charge states will be assigned.")
    logger.warning(
        "Run loaded from legacy ProcessedCyclerRun: No raw data available. File cannot be restructured.")
    return r


def load_run_from_BEEPDatapath_file(
        filename: Union[str, os.PathLike]
) -> Run:
    """
    Load a Run from a legacy BEEPDatapath file.

    Args:
        filename (str, Pathlike): The name of the json or .json.gz file
            serialized to disk by BEEPDatapath.

    Returns:
        Run: The Run converted from a BEEPDatapath.

    """
    d = load_json_safe(filename)
    if "Datapath" not in d.get("@class", None) or "beep.structure" not in d.get(
            "@module", None):
        logger.error(
            "Mismatching ProcessedCyclerRun class! The file json is malformed.")

    metadata = d.get("metadata", {})
    for k in ('barcode', 'protocol', 'channel_id', 'schema_path',
              'structuring_parameters'):
        if k in d:
            metadata[k] = d[k]


    paths = d.get("paths", {})
    interpolated_regular = d.get("cycles_interpolated", None)
    interpolated_diagnostic = d.get("diagnostic_interpolated", None)

    interpolated_regular = pd.DataFrame(
        interpolated_regular) if interpolated_regular else None
    interpolated_diagnostic = pd.DataFrame(
        interpolated_diagnostic) if interpolated_diagnostic else None

    all_interpolated = pd.concat(
        [interpolated_regular, interpolated_diagnostic])

    summary = d.get("summary", None)
    summary_diagnostic = d.get("diagnostic_summary", None)
    summary = pd.DataFrame(summary) if summary else None
    summary_diagnostic = pd.DataFrame(
        summary_diagnostic) if summary_diagnostic else None

    diag_dict = d.get("diagnostic_config", None)
    diagnostic = DiagnosticConfig.from_dict(diag_dict) if diag_dict else None

    raw_data = d.get("raw_data", None)
    raw_data = pd.DataFrame(raw_data) if raw_data else None

    # Account for step_index being called step_code now
    for df in (raw_data, all_interpolated):
        if df is not None:
            if "step_index" in df.columns:
                df.rename(columns={"step_index": "step_code"}, inplace=True)

    raw_cc = CyclesContainer.from_dataframe(raw_data,
                                            tqdm_desc_suffix=TQDM_RAW_SUFFIX) if raw_data is not None else None
    structured_cc = CyclesContainer.from_dataframe(all_interpolated,
                                                   tqdm_desc_suffix=TQDM_STRUCTURED_SUFFIX) if interpolated_regular is not None else None

    return Run(
        raw_cycle_container=raw_cc,
        structured_cycle_container=structured_cc,
        diagnostic=diagnostic,
        metadata=metadata,
        paths=paths,
        summary_regular=summary,
        summary_diagnostic=summary_diagnostic
    )