"""Base classes for EIS.

Use these classes to create new EIS-capable BEEPDatapath classes for EIS-capable cyclers.

"""
import abc

import pandas as pd
from monty.json import MSONable

from beep.structure.base import BEEPDatapath


class EIS(MSONable):
    """Base class for an Electrochemical Impedance Spectrum

    EIS can be used by itself to hold EIS data; alternatively, it can be
    inherited by child *EIS classes (e.g., MaccorEIS) for specific cyclers
    which have their own ingestion or anaylsis formats.

    Each EIS instance corresponds to a single electrochemical impedance spectrum.

    This class does not need to implement a from_file method, though
    if you intend for it to be used with BEEPDatapathWithEIS, it should
    implement from_file.

    Attributes:
        data (pd.DataFrame): The EIS data, as a dataframe
        metadata (pd.DataFrame): The EIS metadata, as a dataframe
    """

    def __init__(self, data, metadata):
        """
        Args:
            data (pd.DataFrame): The EIS data, as a dataframe
            metadata (pd.DataFrame): The EIS metadata, as a dataframe
        """
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_file(cls, filename):
        """Create an EIS object from a raw data file.

        Args:
            filename (str, Pathlike): The filename of the raw EIS file.

        Returns:

        """
        raise NotImplementedError

    def as_dict(self):
        """Method for serialization as dictionary

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
        """Method of invocation from dictionary

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
    """An ABC for BEEPDatapaths capable of handling EIS data.

    BEEPDatapathWithEIS can be inherited by child cycler classes for handling and structuring
    cycler runs with EIS data. This does NOT mean that BEEPDatapathWithEIS must contain EIS data.
    All the capability of normal BEEPDatapathWithEIS is retained; extra features are added.

    All BEEPDatapathWithEIS MUST implement a load_eis method.

    Attributes:
        - eis ([EIS]): A list of EIS or EIS child objects; i.e., 0 or more electrochemical impedance spectra.
    """

    def __init__(self, *args, **kwargs):
        self.eis = None
        super(BEEPDatapathWithEIS, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def load_eis(self, *args, **kwargs):
        """Load one or more EIS files.

        Implement this method to load one or more EIS files.

        load_eis should do both of the following:
            - add all of the eis paths to the BEEPDatapathWithEIS.paths attribute like:
                self.paths["eis"] = <<list_of_eis_paths>>

            - update BEEPDatapathWithEIS.eis as a list of EIS or EIS-derived objects:
                self.eis = [EIS_obj1, EIS_obj2, EIS_obj3, ....]


        Args:
            *args:
            **kwargs:

        Returns:
            None
        """
        raise NotImplementedError(
            "EIS containing datapath must implement 'load_eis' method."
        )
