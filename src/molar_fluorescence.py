import typing
import numpy as np
from .get_data import file_to_numpy


class MolarFluorescence:
    """
    Parameters
    ----------
    f : np.ndarray
        molar fluorescences for each cycle/well :math:`\\mathbf{f}`, n by number of wells matrix
        Determined in units of fluorescence divided by (pmol/L)
    df : np.ndarray
        standard deviation in molar fluorescences :math:`\\sigma`, n by number of wells matrix
        Determined in units of fluorescence divided by (pmol/L)
    cv : np.ndarray
        Coefficient of variation in molar fluorescences :math:`\\sigma/\\mathbf{f}`,
    q : int
        number of plates (different concentrations)
    n : int
        number of cycles
    m : int
        number of wells
    F : np.ndarray
        raw fluorescence data (scaled by 10^6 as described in get_data.py).
        Tensor of dimensions :math:`n\\times m \\times q`

    """
    def __init__(self, C: np.array, files: typing.List[str], name: str):
        """

        Parameters
        ----------
        C : np.array
            concentrations of reporter in pmol/L
        files : typing.List[str]
            list of file names (absolute paths)
        name : str
            name of reporter (e.g., FAM, Probe)
        """
        self.n = 45
        self.m = 96
        self.C = C
        self.q = C.shape[0]
        assert self.q > 1, "Must have more than one dataset!"
        self.F = np.zeros((self.n, self.m, self.q))
        for i, f in enumerate(files):
            self.F[:, :, i] = file_to_numpy(f)

        self.f = np.zeros((self.n, self.m))
        self.df = np.zeros((self.n, self.m))
        self.cv = np.zeros((self.n, self.m))
        self.name = name

    def calculate(self):
        """Perform calculations of :math:`\\mathbf{f}` and standard deviation :math:`\\sigma`"""
        for i in range(self.n):
            for w in range(self.m):
                self.f[i, w] = np.inner(self.F[i, w, :], self.C) / np.inner(self.C, self.C)
                R = self.F[i, w, :] - self.f[i, w] * self.C
                self.df[i, w] = np.sqrt(np.inner(R, R) / (self.q - 1))
                self.cv[i, w] = self.df[i, w] / self.f[i, w]
    
    def print_cv(self):
        """Print information relating to coefficient of variation
        """
        print("   CV min: %f max: %f mean: %f std: %f" % (self.cv.min(), self.cv.max(), self.cv.mean(), self.cv.std(ddof=1)))