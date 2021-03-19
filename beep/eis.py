import abc

from monty.json import MSONable



class EISpectrum(MSONable):
    """
    Class describing an Electrochemical Impedance Spectrum
    """

    def __init__(self, data, metadata):
        """

        Args:
            data:
            metadata:
        """
        self.data = data
        self.metadata = metadata


    @classmethod
    def from_file(cls):
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
            "metadata": self.metadata.to_dict(),
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