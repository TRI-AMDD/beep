
from typing import Iterable, Optional, Tuple, Dict
from itertools import chain
from dataclasses import dataclass

import pandas as pd
from monty.json import MSONable

from beep import logger


class DiagnosticConfig(MSONable):
    """
    Advanced configuration of diagnostic cycles.

    Simply holds and represents a dict; the keys in the dict
    are the diagnostic cycle types (e.g., RPT0.5), and the values
    are iterables (list/set/tuples) of cycle indices.

    Example:
        {
        "rpt0.5": {1,3,5,9,110...},
        "rpt1.0": {2,4,6,10,122...},
        "hppc": {20,40,60},
        "my_custom_cycle": {101,1048},
        }

    Cycles present but not in the diagnostic config will
    be assumed as regular (aging) cycles.

    Cycle types containing "rpt", "hppc", or "reset" strings will
    be considered RPT, HPPC, and RESET cycles respectively. This
    only affects the downstream structuring parameters, not the
    labelling of the individual cycles.

    Args:
        diagnostic_config (dict): Dict mapping cycle types (strings) to
            iterables of cycle indices (integers). Each cycle index
            should correspond with exactly one cycle type.
        **kwargs: Parameters that can be used by downstream structuring methods.

    Attributes:
        cycle_type_to_ix (dict): Map of the diagnostic cycle type (str)
            to a list of cycle indices (set of ints).
        ix_to_cycle_type (dict): Map of each cycle index (int) to the diagnostic
            cycle type (str).
        hppc_ix: indices of cycles that could be considered HPPC. Determined
            automatically if "hppc" is found in the name of a cycle type.
        rpt_ix: indices of cycles that could be considered RPT. Determined
            automatically if "rpt" is found in the name of a cycle type.
        reset_ix: indices of cycles that could be considered RESET. Determined
            automatically if "reset" is found in the name of a cycle type.
        params (dict): Parameters that can be used by downstream structuring
            methods.
    """

    def __init__(
            self,
            diagnostic_config: Dict[str, Iterable[int]],
            **kwargs
    ):

        self.cycle_type_to_ix = \
            {c: frozenset(ix) for c, ix in diagnostic_config.items()}

        ix_non_unique = list(self.cycle_type_to_ix.values())
        ix_unique = frozenset.union(*ix_non_unique)

        if sum([len(ix) for ix in ix_non_unique]) != len(ix_unique):
            raise ValueError(
                "There is overlap between cycles!"
                "Each cycle must have exactly one diagnostic type."
            )
        self.ix_to_cycle_type = {}
        for cycle_type, ixs in self.cycle_type_to_ix.items():
            for ix in ixs:
                self.ix_to_cycle_type[ix] = cycle_type

        hppc_cycle_types = {ct for ct in self.cycle_type_to_ix if "hppc" in ct.lower()}
        rpt_cycle_types = {ct for ct in self.cycle_type_to_ix if "rpt" in ct.lower()}
        reset_cycle_types = {ct for ct in self.cycle_type_to_ix if "reset" in ct.lower()}
        self.hppc_ix = frozenset.union(
            *[self.cycle_type_to_ix[hppc] for hppc in hppc_cycle_types] +
             [frozenset()]
        )
        self.rpt_ix = frozenset.union(
            *[self.cycle_type_to_ix[rpt] for rpt in rpt_cycle_types] +
             [frozenset()]
        )
        self.reset_ix = frozenset.union(
            *[self.cycle_type_to_ix[reset] for reset in reset_cycle_types] +
             [frozenset()]
        )
        self.params = kwargs

    @classmethod
    def from_dict(cls, d):
        """
        Create a DiagnosticConfig object from a dictionary.

        Args:
            d (dict): Dictionary to use to create diagnostic config.

        Returns:
            (DiagnosticConfig)

        """
        return cls(diagnostic_config=d["cycle_type_to_ix"])

    def as_dict(self) -> dict:
        """
        Convert a DiagnosticConfig object into a dictionary, e.g. for
        serialization purpose with monty.

        Returns:
            (dict)

        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "cycle_type_to_ix": self.cycle_type_to_ix,
            "ix_to_cycle_type": self.ix_to_cycle_type
        }


class DiagnosticConfigBasic(DiagnosticConfig):
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
    def __init__(
            self,
            hppc_ix: Iterable = tuple(),
            rpt_ix: Iterable = tuple(),
            reset_ix: Iterable = tuple(),
            fallback_v_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(
            diagnostic_config={
                "hppc": hppc_ix,
                "rpt": rpt_ix,
                "reset": reset_ix
            },
            fallback_v_range=fallback_v_range
        )

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


def legacy_conversion(diagnostic_available):
    """
    Converts a legacy "diagnostic available" dictionary to a new
    DiagnosticConfig object.

    Args:
        diagnostic_available:

    Returns:

    """
    pass



if __name__ == "__main__":

    configdict = {
        "rpt2.0": [1,11,21,31],
        "rpt5.0": [2,12,22,32],
        "hppc":[0,104],
    }

    dc = DiagnosticConfig(configdict)


    print(dc.rpt_ix)
    print(dc.ix_to_cycle_type[104])
    print(dc.ix_to_cycle_type[12])



