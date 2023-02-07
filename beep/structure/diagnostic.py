from typing import Iterable, Tuple, Dict

import pandas as pd
from monty.json import MSONable

"""
Classes defining diagnostic (e.g., RPT, HPPC) cycles
and their locations in cycler runs (Datapaths).
"""


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
            May only be python primitives str, int, or float.

    Attributes:
        cycles (dict): Map of the diagnostic cycle type (str)
            to a list of cycle indices (set of ints).
        by_ix (dict): Map of each cycle index (int) to the diagnostic
            cycle type (str). Does not map regular cycles.
        all_ix: All diagnostic cycle indices, regardless of types.
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
        if not diagnostic_config:
            raise ValueError(
                f"{self.__class__.__name__} needs at least one "
                f"diagnostic cycle type"
            )

        self.cycle_type_to_cycle_ix = \
            {c: frozenset(ix) for c, ix in diagnostic_config.items()}

        ix_non_unique = list(self.cycle_type_to_cycle_ix.values())
        ix_unique = frozenset.union(*ix_non_unique)

        if sum([len(ix) for ix in ix_non_unique]) != len(ix_unique):
            raise ValueError(
                "There is overlap between cycles!"
                "Each cycle must have exactly one diagnostic type."
            )
        self.cycle_ix_to_cycle_type = {}
        for cycle_type, ixs in self.cycle_type_to_cycle_ix.items():
            for ix in ixs:
                self.cycle_ix_to_cycle_type[ix] = cycle_type

        # Setting hppc_ix, rpt_ix, and reset_ix based on string recognition
        for ix_attr_name in ("hppc", "rpt", "reset"):
            cycle_types = {ct for ct in self.cycle_type_to_cycle_ix
                           if ix_attr_name in ct.lower()}
            frozen = frozenset.union(
                *[self.cycle_type_to_cycle_ix[cyc_typ] for cyc_typ in cycle_types] +
                [frozenset()]
            )
            setattr(self, f"{ix_attr_name}_ix", frozen)

        allowed_kwarg_types = (int, str, float, bool)
        for kw, arg in kwargs.items():
            if not isinstance(arg, allowed_kwarg_types):
                raise TypeError(f"Kwarg {kw} is type {type(arg)}; "
                                f"allowed types are {allowed_kwarg_types}")
        self.params = kwargs
        self.all_ix = frozenset(self.cycle_ix_to_cycle_type.keys())

        # Nice to have shorthand
        self.cycles = self.cycle_type_to_cycle_ix
        self.type_by_ix = self.cycle_ix_to_cycle_type

    @classmethod
    def from_step_numbers(
            cls,
            df_raw: pd.DataFrame,
            matching_criteria: Dict[str, Tuple[str, Iterable[Iterable[int]]]],
            **kwargs
    ):
        """
        Automatically determine diagnostic cycle types and indices by
        providing only step numbers unique to particular diagnostic cycles.

        For example, if step number "7" is always ONLY in HPPC cycles,
        we may assume all cycles containing step number 7.

        Specify how to match via one of two methods: "exact" or "contains".
        In either of these methods, you also pass sets of step numbers
        which can be used to identify particular diagnostic cycle types.

        "Contains" will match if at least one set of step numbers
        passed is entirely found in a cycle. "Exact" will match if at least
        one set of step numbers passed is entirely found in a cycle
        with no other step types present. Sets of step numbers are assumed
        to all be matched based on OR, not AND.

        Example 1:

        Our HPPC cycles always at least contain step numbers 1,2,4,6,8.
        Our low-rate RPT cycles are EXACTLY step numbers 12 and 13.
        Our high-rate RPT cycles are EXACTLY step numbers 15 and 16.

        dc = DiagnosticConfig.from_step_numbers(
            df,
            matching_criteria={
                "hppc": ("contains", [(1, 2, 4, 6, 8)]),
                "rpt_lowrate": ("exact", [(12, 13)]),
                "rpt_highrate": ("exact", [(15, 16)])
            }
        )


        Example 2:

        We have the same example as above but are grouping all RPT
        cycles together. So we'll have cycles labelled RPT which are
        either EXACTLY matching step numbers 12,13 or EXACTLY matching
        step numbers 15,16.


        dc = DiagnosticConfig.from_step_numbers(
            df,
            matching_criteria={
                "hppc": ("contains", [(1, 2, 4, 6, 8)]),
                "rpt_lowrate": ("exact", [(12, 13)]),
                "rpt_highrate": ("exact", [(15, 16)])
            }
        )

        Args:
            df_raw (pd.Dataframe): The raw data from a datapath.
            matching_criteria (dict): Keys are the names of the
                diagnostic cycle types. Values are 2-tuples of the form
                (rule_string, iterables) where rule_string is either
                "contains" or "exact" and the iterable contains one or
                more iterables of integers (cycle indices).

        Returns:
            (DiagnosticConfig)

        """
        target_column = "step_index"
        if target_column not in df_raw.columns:
            raise ValueError(
                f"Required column '{target_column}' not found in raw data!"
            )

        for matching_rule_pair in matching_criteria.values():
            if matching_rule_pair[0] not in ("contains", "exact"):
                raise ValueError(
                    f"Matching rule {matching_rule_pair[0]} not a valid rule."
                )

        all_diag_ix = {cycle_type: set() for cycle_type in matching_criteria.keys()}
        for cix in df_raw["cycle_index"].unique():
            df_cycle = df_raw[df_raw["cycle_index"] == cix]
            for cycle_type, (match_type, match_step_pattern) in matching_criteria.items():
                if match_step_pattern:
                    for step_pattern in match_step_pattern:
                        unique = df_cycle[target_column].unique()
                        all_present = all([sn in unique for sn in step_pattern])
                        if match_type == "contains" and all_present:
                            all_diag_ix[cycle_type].add(cix)
                            break
                        elif all_present and len(unique) == len(set(step_pattern)):
                            all_diag_ix[cycle_type].add(cix)
                            break
        return cls(all_diag_ix, **kwargs)

    @classmethod
    def from_dict(cls, d: dict):
        """
        Create a DiagnosticConfig object from a dictionary.

        Args:
            d (dict): Dictionary to use to create diagnostic config.

        Returns:
            (DiagnosticConfig)

        """
        return cls(diagnostic_config=d["cycle_type_to_ix"], **d["params"])

    def as_dict(self) -> dict:
        """
        Convert a DiagnosticConfig object into a dictionary, e.g. for
        serialization purpose with monty.

        Returns:
            (dict)

        """
        json_compatible_type2cix = {
            k: list(v) for k, v in self.cycle_type_to_cycle_ix.items()
        }
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "cycle_type_to_ix": json_compatible_type2cix,
            "params": self.params
        }
