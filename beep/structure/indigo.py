"""Classes and functions for handling Indigo battery cycler data.

"""
from datetime import datetime

import pytz
import pandas as pd

from beep.structure.base import BEEPDatapath
from beep.conversion_schemas import INDIGO_CONFIG


class IndigoDatapath(BEEPDatapath):
    """Datapath for ingesting and structuring Indigo battery cycler data.

    """

    @classmethod
    def from_file(cls, path):
        """Creates a BEEPDatapath from an raw Indigo data file.

        Args:
            path (str, Pathlike): file path to data file

        Returns:
            (IndigoDatapath)
        """

        data = pd.read_hdf(path, "time_series_data")
        metadata = dict()

        if len(list(data["cell_id"].unique())) > 1:
            raise ValueError("More than 1 cell_id exists in {}".format(path))

        metadata["indigo_cell_id"] = int(data["cell_id"].iloc[0])
        metadata["filename"] = path
        metadata["_today_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # transformations
        data = (
            data.reset_index().reset_index()
        )  # twice in case old index is stored in file
        data = data.drop(columns=["index"])
        data.rename(columns={"level_0": "data_point"}, inplace=True)
        data.loc[data.half_cycle_count % 2 == 1, "charge_capacity"] = (
                abs(data.cell_coulomb_count_c) / 3600
        )
        data.loc[data.half_cycle_count % 2 == 0, "charge_capacity"] = 0
        data.loc[data.half_cycle_count % 2 == 0, "discharge_capacity"] = (
                abs(data.cell_coulomb_count_c) / 3600
        )
        data.loc[data.half_cycle_count % 2 == 1, "discharge_capacity"] = 0
        data.loc[data.half_cycle_count % 2 == 1, "charge_energy"] = abs(
            data.cell_energy_j
        )
        data.loc[data.half_cycle_count % 2 == 0, "charge_energy"] = 0
        data.loc[data.half_cycle_count % 2 == 0, "discharge_energy"] = abs(
            data.cell_energy_j
        )
        data.loc[data.half_cycle_count % 2 == 1, "discharge_energy"] = 0
        data["internal_resistance"] = data.cell_voltage_v / data.cell_current_a
        data["date_time_iso"] = data["system_time_us"].apply(
            lambda x: datetime.utcfromtimestamp(x / 1000000)
            .replace(tzinfo=pytz.UTC)
            .isoformat()
        )

        data.rename(INDIGO_CONFIG["data_columns"], axis="columns", inplace=True)

        metadata["start_datetime"] = data.sort_values(by="system_time_us")[
            "date_time_iso"
        ].iloc[0]

        paths = {
            "raw": path,
            "metadata": path
        }

        return cls(data, metadata, paths)
