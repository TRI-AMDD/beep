"""
Base classes for EIS.
"""

import pandas as pd
from monty.json import MSONable

from beep.structure.base import BEEPDatapath


class EIS(MSONable):
    """
    Class describing an Electrochemical Impedance Spectrum
    """

    def __init__(self, data, metadata):
        """

        Args:
            data (pd.DataFrame)
            metadata (pd.DataFrame)
        """
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        raise NotImplementedError

    def as_dict(self):
        """
        Method for serialization as dictionary

        Returns:
            ({}): dictionary representation of this object

        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "data": self.data.to_dict("list"),
            "metadata": self.metadata.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        """
        Method of invocation from dictionary

        Args:
            d ({}): dictionary from which to invoke

        Returns:
            (EISpectrum): object invoked from dictionary

        """
        data = pd.DataFrame(d["data"])
        data = data.sort_index()
        metadata = pd.DataFrame(d["metadata"])
        return cls(data, metadata)


class BEEPDatapathWithEIS(BEEPDatapath):

    def __init__(self, *args, **kwargs):
        self.eis = None
        super(BEEPDatapathWithEIS, self).__init__(*args, **kwargs)

    def load_eis(self, *args, **kwargs):
        raise NotImplementedError(
            "EIS containing datapath must implement 'load_eis' method.")
