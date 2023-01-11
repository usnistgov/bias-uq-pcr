import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def eval_R(p_fr: float, p_rf: float):
    return np.sqrt(p_rf / p_fr)


def get_pfr_prf(R: float, pbar: float):
    p_fr = pbar / R
    p_rf = R * pbar
    return p_fr, p_rf


def eval_E_S0(E_U0: np.ndarray, R: float):
    return E_U0[0, 0] + R * E_U0[1, 0]


def eval_E_D0(E_U0: np.ndarray, R: float):
    return E_U0[0, 0] - R * E_U0[1, 0]


class Amplification:
    def __init__(self, R: float, pbar: float, E_U0: np.ndarray, V_U0: np.ndarray,
                 farray=np.array, sqrt2=np.sqrt(2)) -> None:
        self.R = R
        self.pbar = pbar
        self.E_U0 = E_U0
        self.V_U0 = V_U0

        self.x1 = farray([[R], [1]]) / sqrt2
        self.x2 = farray([[R], [-1]]) / sqrt2
        self.z1 = farray([[1 / R], [1]]) / sqrt2
        self.z2 = farray([[1 / R], [-1]]) / sqrt2
        self.l1 = 1 + pbar
        self.l2 = 1 - pbar

        self.K1 = 1 / 2 * farray([
            [1 - R * pbar, 0],
            [0, 1 / R - pbar / R / R]
        ]
        )
        self.K2 = 1 / 2 * farray([
            [-1 + R * pbar, 0],
            [0, 1 / R - pbar / R / R]
        ]
        )

    def get_A(self) -> np.ndarray:
        return self.l1 * self.x1 @ self.z1.T + self.l2 * self.x2 @ self.z2.T

    def get_Atoi(self, i: int) -> np.ndarray:
        return pow(self.l1, i) * self.x1 @ self.z1.T + pow(self.l2, i) * self.x2 @ self.z2.T

    def get_E_Ui(self, i: int) -> np.ndarray:
        return self.get_Atoi(i) @ self.E_U0

    def get_x1z1TK1z1x1T(self):
        return self.l2 * (self.R + 1) / 8 * np.array([
            [1, 1 / self.R],
            [1 / self.R, 1 / self.R / self.R]
        ])

    def get_x1z1TK2z1x1T(self):
        return self.l1 * (self.R - 1) / 8 * np.array([
            [1, 1 / self.R],
            [1 / self.R, 1 / self.R / self.R]
        ])

    def get_V_Ui(self, i: int) -> np.ndarray:
        E_S0, E_D0 = eval_E_S0(self.E_U0, self.R), eval_E_D0(self.E_U0, self.R)
        return self.get_Atoi(i) @ self.V_U0 @ (self.get_Atoi(i).T) \
               + self.pbar * E_S0 * (
                       (pow(self.l1 * self.l1, i) - pow(self.l1, i)) / (
                           self.l1 * self.l1 - self.l1) * self.x1 @ self.z1.T @ self.K1 @ self.z1 @ self.x1.T
                       +
                       (pow(self.l1 * self.l2, i) - pow(self.l1, i)) / (
                                   self.l1 * self.l2 - self.l1) * self.x1 @ self.z1.T @ self.K1 @ self.z2 @ self.x2.T
                       +
                       (pow(self.l1 * self.l2, i) - pow(self.l1, i)) / (
                                   self.l1 * self.l2 - self.l1) * self.x2 @ self.z2.T @ self.K1 @ self.z1 @ self.x1.T
                       +
                       (pow(self.l2 * self.l2, i) - pow(self.l1, i)) / (
                                   self.l2 * self.l2 - self.l1) * self.x2 @ self.z2.T @ self.K1 @ self.z2 @ self.x2.T
               ) \
               + self.pbar * E_D0 * (
                       (pow(self.l1 * self.l1, i) - pow(self.l2, i)) / (
                           self.l1 * self.l1 - self.l2) * self.x1 @ self.z1.T @ self.K2 @ self.z1 @ self.x1.T
                       +
                       (pow(self.l1 * self.l2, i) - pow(self.l2, i)) / (
                                   self.l1 * self.l2 - self.l2) * self.x1 @ self.z1.T @ self.K2 @ self.z2 @ self.x2.T
                       +
                       (pow(self.l1 * self.l2, i) - pow(self.l2, i)) / (
                                   self.l1 * self.l2 - self.l2) * self.x2 @ self.z2.T @ self.K2 @ self.z1 @ self.x1.T
                       +
                       (pow(self.l2 * self.l2, i) - pow(self.l2, i)) / (
                                   self.l2 * self.l2 - self.l2) * self.x2 @ self.z2.T @ self.K2 @ self.z2 @ self.x2.T
               )

    def get_EXi_over_EYi(self, cycles: np.array) -> np.array:
        r"""Calculate :math:`\mathbb{E}\left[X_i\right]/\mathbb{E}\left[Y_i\right]` for a variety of cycles :math:`i`

        Parameters
        ----------
        cycles : np.array
            cycles (integers) to calculate for each i

        Returns
        -------
        np.array
            :math:`\mathbb{E}\left[X_i\right]/\mathbb{E}\left[Y_i\right]`
        """
        EXi_over_EYi = np.zeros(cycles.shape[0])
        for i in range(cycles.shape[0]):
            EUi = self.get_E_Ui(i)
            EXi_over_EYi[i] = EUi[0, 0] / EUi[1, 0]
        return EXi_over_EYi

    def get_nu(self):
        return self.V_U0[0, 0] + self.R * self.V_U0[1, 0] + self.R * self.V_U0[0, 1] + self.R * self.R * self.V_U0[1, 1] \
               + eval_E_S0(self.E_U0, self.R) * (1 + self.R) / 2 * self.l2 / self.l1 \
               + eval_E_D0(self.E_U0, self.R) * (1 - self.R) / 2 * self.l1 / (2 * self.l1 + self.l2)


class HydrolysisProbes(Amplification):
    """Specific subclass for hydrolysis probes

    >>> pbar = 0.9
    >>> R = 1
    >>> y = 5
    >>> E_U0 = np.array([[y], [y]])
    >>> V_U0 = np.array([[y, 0], [0, y]])
    >>> cls = HydrolysisProbes(R, pbar, E_U0, V_U0)
    >>> cls.get_V_DX(35)/cls.get_E_DX(35)/cls.get_E_DX(35)
    >>> cls.get_V_Ui(35)[0, 0]/cls.get_E_Ui(35)[0, 0]/cls.get_E_Ui(35)[0, 0]

    """

    def __init__(self, *args, **kwargs):
        Amplification.__init__(self, *args, **kwargs)

    def get_E_DX(self, i):
        return self.get_E_Ui(i)[0, 0] - self.E_U0[0, 0]

    def get_Cov_XiX0(self, i):
        # see rocketbook `Covariance with initial`
        return self.V_U0[0, 0] / 2. * (pow(self.l1, i) + pow(self.l2, i))

    def get_V_DX(self, i):
        return self.get_V_Ui(i)[0, 0] + self.V_U0[0, 0] - 2 * self.get_Cov_XiX0(i)


def formula_sympy():
    import sympy as s
    R = s.Symbol("R", positive=True, real=True)
    pbar = s.Symbol("p", positive=True, real=True)
    E = np.ones((2, 1))
    V = np.ones((2, 2))
    cls = Amplification(R, pbar, E, V, farray=s.Matrix, sqrt2=s.sqrt(2))
    s.pprint(s.simplify(cls.x1 @ cls.z1.T @ cls.K1 @ cls.z1 @ cls.x1.T - cls.get_x1z1TK1z1x1T()))

    s.pprint(s.simplify(cls.x1 @ cls.z1.T @ cls.K2 @ cls.z1 @ cls.x1.T - cls.get_x1z1TK2z1x1T()))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
    # formula_sympy()
