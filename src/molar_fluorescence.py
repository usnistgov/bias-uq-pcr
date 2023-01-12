import os
import typing
import numpy as np
from .get_data import file_to_numpy


class MolarFluorescence:
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
        self.num_wells = 96
        self.C = C
        self.q = C.shape[0]
        assert self.q > 1, "Must have more than one dataset!"
        self.F = np.zeros((self.n, self.num_wells, self.q))
        for i, f in enumerate(files):
            self.F[:, :, i] = file_to_numpy(f)

        self.f = np.zeros((self.n, self.num_wells))
        self.df = np.zeros((self.n, self.num_wells))
        self.name = name

    def calculate(self):
        for i in range(self.n):
            for w in range(self.num_wells):
                self.f[i, w] = np.inner(self.F[i, w, :], self.C) / np.inner(self.C, self.C)
                R = self.F[i, w, :] - self.f[i, w] * self.C
                self.df[i, w] = np.sqrt(np.inner(R, R) / (self.q - 1))

    def save(self, dir: str):
        with open(os.path.join(dir, "f_iw_%s.npy" % self.name), "wb") as file:
            np.save(file, self.f)

        with open(os.path.join(dir, "df_iw_%s.npy" % self.name), "wb") as file:
            np.save(file, self.df)