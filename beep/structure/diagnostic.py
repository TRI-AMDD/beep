
from typing import Iterable, Optional


import pandas as pd
from monty.json import MSONable

from beep import logger


class DiagnosticConfigBasic(MSONable):
    """
    A class for representing diagnostic cycle configurations,
    their locations in cycle files, and information regarding
    their steps.

    Is basic because it only accounts for three kinds of cycles,
    HPPC, RPT, and Reset, and lumps all cycles into these catagories.

    All other indices are assumed to be normal.
    """

    def __init__(
            self,
            hppc_ix: Iterable,
            rpt_ix: Iterable,
            reset_ix: Iterable,

    ):
        self.hppc_ix = set(hppc_ix)
        self.rpt_ix = set(rpt_ix)
        self.reset_ix = set(reset_ix)

    @classmethod
    def from_step_numbers(
            cls,
            df_raw,
            hppc_match: Optional[Iterable[Iterable]] = None,
            hppc_match_type: str = "contains",
            rpt_match: Optional[Iterable[Iterable]] = None,
            rpt_match_type: str = "contains",
            reset_match: Optional[Iterable[Iterable]] = None,
            reset_match_type: str = "exact",
    ):
        """
        A method to automatically determine diagnostic cycle
        types by providing only step numbers unique to particular
        diagnostic cycles.

        For example, if step number "7" is always ONLY in HPPC cycles,
        we may assume all cycles containing step number 7.


        Similarly, we can match based on ex


        Args:
            df_raw:
            hppc_step_numbers:
            rpt_step_numbers:
            reset_step_numbers:

        Returns:

        """

        match_types = (hppc_match_type, rpt_match_type, reset_match_type)
        match_step_patterns = (hppc_match, rpt_match, reset_match)
        match_step_names = ("hppc", "rpt", "reset")

        target_column = "step_index"
        if target_column not in df_raw.columns:
            raise ValueError(f"Required column '{target_column}' not found in raw data!")

        all_diag_ix = [[], [], []]
        for cix in df_raw["cycle_index"].unique():
            df_cycle = df_raw[df_raw["cycle_index"] == cix]

            for i, cyc_match_list in enumerate(match_step_patterns):
                if cyc_match_list:
                    for cyc_match in cyc_match_list:

                        print(f"doing cycle match for {match_step_names[i]} on {cyc_match}")
                        unique = df_cycle[target_column].unique()
                        all_present = all([sn in unique for sn in cyc_match])
                        print(f"unique/present: {unique}, {all_present}")

                        if match_types[i] == "contains" and all_present:
                            print(f"Found match for contains on cycle {cix} for {match_step_names[i]}")
                            all_diag_ix[i].append(cix)
                            break
                        elif all_present and len(unique) == len(set(cyc_match)):
                            print(f"Found match for exact on cycle {cix} for {match_step_names[i]}")
                            all_diag_ix[i].append(cix)
                            break
        return cls(*all_diag_ix)

    def as_dict(self) -> dict:
        pass

    def from_dict(cls, d):
        pass
