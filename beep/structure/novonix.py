import pandas as pd
import os
from beep.structure.base import BEEPDatapath
from beep.conversion_schemas import NOVONIX_CONFIG
from beep import VALIDATION_SCHEMA_DIR, logger
from datetime import datetime


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
            search_lines = 200
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
                            metadata_value = l.split(":")[-1].strip().replace("\\n", "").replace("'", "")
                            metadata[k] = metadata_value if metadata_value else None
                            break
                i += 1
                if i > search_lines:
                    raise LookupError("Unable to find the header line in first {} lines of file".format(search_lines))
        raw = pd.read_csv(path, sep='\t', header=None)
        raw.dropna(axis=0, how='all', inplace=True)
        data = raw.iloc[header_starts_line - 1:]
        data = data[0].str.split(',', expand=True)
        headers = data.iloc[0]
        data = pd.DataFrame(data.values[1:], columns=headers, index=None)

        # format columns
        map = cls.conversion_config['data_columns']
        type_map = {j: map[j]['data_type'] for j in map}
        data = data.astype(type_map)
        data['Temperature (°C)'] = data['Temperature (°C)'].astype('float')
        data['Circuit Temperature (°C)'] = data['Circuit Temperature (°C)'].astype('float')
        name_map = {i: map[i]['beep_name'] for i in map}
        data.rename(name_map, axis="columns", inplace=True)

        # format capacity and energy
        rest = data['step_type_num'] == 0
        cc_charge = data['step_type_num'] == 1
        cc_discharge = data['step_type_num'] == 2
        cccv_charge = data['step_type_num'] == 7
        cv_hold_discharge = data['step_type_num'] == 8
        cccv_discharge = data['step_type_num'] == 9
        cccv_hold_discharge = data['step_type_num'] == 10

        data['charge_capacity'] = data[cc_charge | cccv_charge]['capacity'].astype('float')

        data['discharge_capacity'] = \
            data[rest | cc_discharge | cv_hold_discharge | cccv_discharge | cccv_hold_discharge][
            'capacity'].astype('float')
        data['charge_energy'] = data[cc_charge | cccv_charge]['energy'].astype('float')
        data['discharge_energy'] = data[cc_discharge | cv_hold_discharge | cccv_discharge | cccv_hold_discharge][
            'energy'].astype('float')
        data['date_time_iso'] = data['date_time'].map(
            lambda x: datetime.strptime(x, '%Y-%m-%d %I:%M:%S %p').isoformat())

        # add step type #todo set schema
        step_map = {0: 'discharge',
                    1: 'charge',
                    2: 'discharge',
                    7: 'charge',
                    8: 'discharge',
                    9: 'discharge',
                    10: 'discharge'}
        data['step_type'] = data['step_type_num'].replace(step_map)
        data.fillna(0)

        # Correct discharge capacities and energies for convention
        data["cycle_charge_max"] = data.groupby("cycle_index")["charge_capacity"].transform("max")

        ix = data[(data["step_type"] == "discharge") & (data["step_type_num"] != 0)].index
        cycle_chg_max = data["cycle_charge_max"].loc[ix]
        discharge_capacities = data["discharge_capacity"].loc[ix]
        data.loc[ix, "discharge_capacity"] = -1.0 * (cycle_chg_max - discharge_capacities)

        summary = None
        if summary_path and os.path.exists(summary_path):
            summary = pd.read_csv(summary_path, index_col=0).to_dict("list")
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


if __name__ == "__main__":
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.width = None
    fname = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/raw/test_Nova_Form-CH01-01_short.csv"
    md_fname = "/Users/ardunn/alex/tri/code/beep/beep/tests/test_files/raw/test_Nova_Form-CH01-01_short_metadata.csv"

    dp = NovonixDatapath.from_file(fname, summary_path=md_fname)

    print(dp.raw_data.columns)

    print(dp.metadata)

    print(dp.external_summary)

    dp.structure(
        charge_axis="test_time",
        discharge_axis="test_time",
        resolution=100
    )


    # print(dp.raw_data[["discharge_capacity", "step_type"]])