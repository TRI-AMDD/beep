
from typing import Iterable, Optional, Tuple
from itertools import chain

import pandas as pd
from monty.json import MSONable

from beep import logger


class DiagnosticConfigBasic(MSONable):
    """
    A class for representing diagnostic cycle configurations,
    their locations in cycle files, and information regarding
    their steps.

    Is basic because it only accounts for three kinds of cycles,
    HPPC, RPT, and Reset, and lumps all cycles into these categories.

    - HPPC: hybrid power-pulse characterization
    - RPT: reference performance test
    - RESET: cycler-specific reset cycles

    All other indices are assumed to be normal.
    """
    HPPC = "hppc"
    RPT = "rpt"
    RESET = "reset"

    def __init__(
            self,
            hppc_ix: Iterable,
            rpt_ix: Iterable,
            reset_ix: Iterable,
            fallback_v_range: Optional[Tuple[float, float]] = None
    ):
        self.hppc_ix = frozenset(hppc_ix)
        self.rpt_ix = frozenset(rpt_ix)
        self.reset_ix = frozenset(reset_ix)
        self.dv_fallback = fallback_v_range

        all_ix = []
        for ix in chain(self.hppc_ix, self.rpt_ix, self.reset_ix):
            all_ix.append(ix)
        self.all_ix = frozenset(all_ix)

        if len(all_ix) != len(self.all_ix):
            raise ValueError(
                "There is overlap between cycles in the"
                "HPPC/RPT/Reset cycles! Each cycle must "
                "have exactly one diagnostic type."
            )
        self.cycle_to_type = {}
        for ctype, ix_list in {
                self.HPPC: self.hppc_ix,
                self.RPT: self.rpt_ix,
                self.RESET: self.reset_ix
        }.items():
            for ix in ix_list:
                self.cycle_to_type[ix] = ctype

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
            **kwargs
    ):
        """
        A method to automatically determine diagnostic cycle
        types and indices by providing only step numbers unique to
        particular diagnostic cycles.

        For example, if step number "7" is always ONLY in HPPC cycles,
        we may assume all cycles containing step number 7.

        Specify how to match via one of two methods: "exact" or "contains".
        In either of these methods, you also pass sets of step numbers
        which can be used to identify particular diagnostic cycle types.

        "Contains" will match if at least one set of step numbers
        passed is entirely found in a cycle. "Exact" will match if at least
        one set of step numbers passed is entirely found in a cycle
        with no other step types present.

        Sets of step numbers are assumed to all be matched based on OR,
        not AND. This method is not assumed to work for ALL potential
        cycler runs, but may be useful for many.

        Example 1:
            rpt_match=[(12, 13), (15, 16)],
            rpt_match_type="exact"

            Cycles are identified with RPT if they contain
            EXACTLY (only) step numbers 12 and 13 OR
            EXACTLY (only) step numbers 15 and 16.

        Example 2:
            hppc_match =[(1, 2, 3, 4, 6, 8)]
            hppc_match_type="contains"

            Cycles are identified as HPPC if they contain
            the full set of step numbers 1, 2, 3, 4, 6, and 8.
            Matching cycles may also contain other step numbers.

        Args:
            df_raw (pd.Dataframe): The raw data from a datapath.
            hppc_match: An iterable of sets of step numbers
                which can be used to match against potential HPPC cycles
                in order to automatically identify them.
            hppc_match_type (str): "contains" or "exact". "Contains" will
                match cycles containing at least one of the hppc_step_numbers
                sets entirely. "Exact" will match cycles only if at least
                one of the hppc_step_numbers sets is found entirely in a
                cycle with NO other step types.
            rpt_match: Iterable of sets of step numbers for matching
                on RPT cycles.
            rpt_match_type: Same syntax as hppc_match_type.
            reset_match: Iterable of the sets of step numbers for
                matching on reset cycles.
            reset_match_type: Same syntax as hppc_match_type and
                rpt_match_type.

        Returns:
            (DiagnosticConfigBasic)
        """

        match_types = (hppc_match_type, rpt_match_type, reset_match_type)
        match_step_patterns = (hppc_match, rpt_match, reset_match)
        target_column = "step_index"
        if target_column not in df_raw.columns:
            raise ValueError(f"Required column '{target_column}' not found in raw data!")

        all_diag_ix = ([], [], [])
        for cix in df_raw["cycle_index"].unique():
            df_cycle = df_raw[df_raw["cycle_index"] == cix]

            for i, cyc_match_list in enumerate(match_step_patterns):
                if cyc_match_list:
                    for cyc_match in cyc_match_list:
                        unique = df_cycle[target_column].unique()
                        all_present = all([sn in unique for sn in cyc_match])
                        if match_types[i] == "contains" and all_present:
                            all_diag_ix[i].append(cix)
                            break
                        elif all_present and len(unique) == len(set(cyc_match)):
                            all_diag_ix[i].append(cix)
                            break
        return cls(*all_diag_ix, **kwargs)

    def as_dict(self) -> dict:
        pass

    def from_dict(cls, d):
        pass


def legacy_conversion(diagnostic_available):
    """
    Converts a legacy "diagnostic available" dictionary to a new
    DiagnosticConfig object.

    Args:
        diagnostic_available:

    Returns:

    """
    pass