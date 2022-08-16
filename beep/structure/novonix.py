import os
from datetime import datetime

import pandas as pd

from beep.structure.base import BEEPDatapath
from beep.conversion_schemas import NOVONIX_CONFIG
from beep import VALIDATION_SCHEMA_DIR, logger


class NovonixDatapath(BEEPDatapath):
    """
    A BEEPDatapath for ingesting and structuring Novonix data files.
    """
    conversion_config = NOVONIX_CONFIG
    external_summary = None

    @classmethod
    def from_file(cls, path, summary_path=None):
        """Create a NovonixDatapath from a raw Novonix cycler file.

        Args:
            path (str, Pathlike): file path for novonix file.
            summary_path (str, Pathlike): file path for the novonix summary file.
                This will be accessible under the NovonixDatapath.external_summary
                attr.

        Returns:
            (NonovixDatapath)
        """
        # format raw data
        metadata_conversion = cls.conversion_config["metadata_fields"]
        metadata = {k: None for k in metadata_conversion.keys()}
        metadata_conversion = {v: k for k, v in metadata_conversion.items()}

        with open(path, "rb") as f:
            i = 1
            search_lines = 500
            header_starts_line = None

            begin_summary = False
            end_summary = False

            while header_starts_line is None:
                line = f.readline()
                if b'Cycle Number' in line:
                    header_starts_line = i
                elif b"[Summary]" in line:
                    begin_summary = True
                elif b"[End Summary]" in line:
                    end_summary = True

                # Rip metadata from summary section
                if begin_summary and not end_summary:
                    l = str(line)
                    for conversion_phrase in metadata_conversion:
                        if conversion_phrase in l:
                            k = metadata_conversion[conversion_phrase]
                            metadata_value = \
                                l.split(":")[-1].strip(). \
                                replace("\\n", "").replace("'", "")
                            metadata[k] = metadata_value if metadata_value else None
                            break
                i += 1
                if i > search_lines:
                    raise LookupError("Unable to find the header line in first "
                                      "{} lines of file".format(search_lines))
        raw = pd.read_csv(path, sep='\t', header=None, encoding="utf-8")
        raw.dropna(axis=0, how='all', inplace=True)
        data = raw.iloc[header_starts_line - 1:]
        data = data[0].str.split(',', expand=True)
        headers = data.iloc[0]
        data = pd.DataFrame(data.values[1:], columns=headers, index=None)

        # format columns
        map = cls.conversion_config['data_columns']
        type_map = {j: map[j]['data_type'] for j in map}
        data = data.astype(type_map)
        name_map = {i: map[i]['beep_name'] for i in map}
        data.rename(name_map, axis="columns", inplace=True)

        # Temperatures with unicode symbol do not work on windows with mapping
        data['Temperature (°C)'] = data['Temperature (°C)'].astype('float')
        data['Circuit Temperature (°C)'] = data[
            'Circuit Temperature (°C)'].astype('float')
        data.rename(
            {
                "Temperature (°C)": "temperature",
                "Circuit Temperature (°C)": "circuit_temperature"
            },
            axis="columns",
            inplace=True
        )

        # ensure that there are not steps with step type numbers outside what is accounted
        # for within the schema
        STEP_NAME_IX_MAP = NOVONIX_CONFIG["step_names"]
        available_step_type_nums = data["step_type_num"].unique().tolist()
        unknown_step_types = []
        for astn in available_step_type_nums:
            if astn not in STEP_NAME_IX_MAP:
                unknown_step_types.append(astn)
        if unknown_step_types:
            raise ValueError(
                f"BEEP cannot process unknown Novonix step indices {unknown_step_types}. "
                f"Known step types by index are {STEP_NAME_IX_MAP}")

        # format capacity and energy
        STEP_IS_CHG_MAP = NOVONIX_CONFIG["step_is_chg"]

        data["step_type_name"] = data["step_type_num"].replace(STEP_NAME_IX_MAP)
        data["step_type"] = data["step_type_name"]. \
            replace(STEP_IS_CHG_MAP). \
            replace({True: "charge", False: "discharge"})

        chg_ix = data["step_type"] == "charge"
        dchg_ix = data["step_type"] == "discharge"

        data['charge_capacity'] = data[chg_ix]['capacity'].astype('float')
        data['discharge_capacity'] = data[dchg_ix]['capacity'].astype('float')
        data['charge_energy'] = data[chg_ix]['energy'].astype('float')
        data['discharge_energy'] = data[dchg_ix]['energy'].astype('float')
        data['date_time_iso'] = data['date_time'].map(
            lambda x: datetime.strptime(x, '%Y-%m-%d %I:%M:%S %p').isoformat())
        data.fillna(0)

        # Correct discharge capacities and energies for convention
        for convention in ("capacity", "energy"):
            data[f"cycle_chg_max_{convention}"] = \
                data.groupby("cycle_index")[f"charge_{convention}"].transform("max")

        # Correct convention for cycles without any charge step
        dchg_only_cycle_ix = []
        for cyc_ix in data["cycle_index"].unique():
            cyc_df = data[data["cycle_index"] == cyc_ix]
            if (cyc_df["step_type"] == "discharge").all():
                ix_range = data[data["cycle_index"] == cyc_ix].index
                dchg_only_cycle_ix.append(cyc_ix)
                for convention in ("capacity", "energy"):
                    data.loc[ix_range, f"cycle_chg_max_{convention}"] = cyc_df["capacity"].max()
        logger.warning(
            f"No charge steps found in cycles {dchg_only_cycle_ix}! "
            f"Using highest capacity reading to determine convention."
        )

        ix = data[(data["step_type"] == "discharge") &
                  (data["step_type_name"] != "rest")].index

        for target_column, max_reference_column in [
            ("discharge_capacity", "cycle_chg_max_capacity"),
            ("discharge_energy", "cycle_chg_max_energy")
        ]:
            cycle_metric_max = data[max_reference_column].loc[ix]
            discharge_data = data[target_column].loc[ix]
            data.loc[ix, target_column] = cycle_metric_max - discharge_data

        data.drop(columns=["cycle_chg_max_capacity", "cycle_chg_max_energy"],
                  inplace=True)

        summary = None
        if summary_path and os.path.exists(summary_path):
            summary = pd.read_csv(
                summary_path,
                index_col=0,
                encoding="utf-8"
            ).to_dict("list")
            if not summary:
                logger.warning(
                    "Summary file was loaded but no data was found. "
                    "Is it misformatted?")
        else:
            logger.warning(f"No associated summary file for Novonix: "
                           f"'{summary_path}': No external summary loaded.")

        # paths
        paths = {
            "raw": path,
            "metadata": path if metadata else None,
            "summary": summary_path if summary else None
        }
        # validation
        schema = os.path.join(VALIDATION_SCHEMA_DIR, "schema-novonix.yaml")
        obj = cls(data, metadata, paths=paths, schema=schema)
        obj.external_summary = summary
        return obj

    def iterate_steps_in_cycle(self, cycle_df, step_type):
        """
        A Novonix-specific method of filtering steps for interpolation
        since the charge/discharge changes are known via the step_type_num
        specification.

        For example, steps within a single cycle are not separated JUST
        by whether they are charge or discharge, they are separated by
        the KIND of charge/discharge.

        For example, a cycle with step type numbers 0, 7, and 8 would be
        broken up into three dataframes. If we are interested in the
        charge cycles, only the 7 data is returned. If we are interested
        in the discharge cycles, the 0 and 8 data is returned separately.

        Args:
            cycle_df (pd.Dataframe): The dataframe for a specific cycle
            step_type (str): "charge" or "discharge"

        Returns:
            (pd.Dataframe): Yields Novonix data as a dataframe
                for a particular step_type num if that step type num
                is the correct step type (chg/discharge)
        """
        gb = cycle_df.groupby("step_type_num")

        for _, step_df in gb:
            if (step_df["step_type"] == step_type).all():
                yield step_df
