from .amplification import Amplification
import matplotlib.pyplot as plt
from .globals import N_av
import numpy as np


class HydrolysisProbes(Amplification):
    """Specific subclass for hydrolysis probes

    Parameters
    ----------
    n : int
        number of cycles
    m : int
        number of wells
    d : np.ndarray
        array of incremental increases in fluorescences, n by m
    b : np.ndarray
        array of background signals, n by m
    dd : np.ndarray
        array of standard deviation in incremental increases in fluorescences, n by m
    db : np.ndarray
        array of standard deviation in background signals, n by m
    E_F : np.ndarray
        expected value of fluorescence, n by m, :math:`\\mathbb{E}\\left[\\mathbf{F}\\right]`
    V_F : np.ndarray
        variance of fluorescence, n by m, :math:`\\mathsf{Var}\\left[\\mathbf{F}\\right]`
    E_DX : np.array
        expected value of change in X, length n, :math:`\\mathbf{E}\\left[\\Delta X_i\\right]`
    V_DX : np.array
        variance value of change in X, length n, :math:`\\mathsf{Var}\\left[\\Delta X_i\\right]`
    cycles : np.array
        cycle numbers, 1 to n


    """

    def __init__(self, C: float, Vol: float,
                 f_plus: np.ndarray, f_minus: np.ndarray,
                 R: float, pbar: float, E_U0: np.array, V_U0: np.ndarray,
                 s_plus: np.ndarray, s_minus: np.ndarray,
                 **kwargs) -> None:
        """

        Parameters
        ----------
        R : float
            ratio of amplification probabilities, :math:`R`
        pbar : float
            geometric mean of amplification probabilities, :math:`\\bar{p}`
        E_U0 : np.array
            initial expected values of both strands :math:`\\mathbb{E}\\left[\\mathbf{U}_0\\right]`, 2 array
        V_U0 : np.ndarray
            initial variances of both strands :math:`\\mathsf{Var}\\left[\\mathbf{U}_0\\right]`, 2 by 2 matrix
        f_plus : np.ndarray
            molar fluorescences for each cycle/well of active reporter :math:`\\mathbf{f}^+`, n by number of wells matrix
            Determined in units of fluorescence divided by (pmol/L)
        f_minus : np.ndarray
            molar fluorescences for each cycle/well of inactive reporter :math:`\\mathbf{f}^-`, n by number of wells matrix
            Determined in units of fluorescence divided by (pmol/L)
        C : float
            Total concentration of reporters in pmol/L, :math:`C`
        Vol : float
            Total volume of solution in L, :math:`\\mathcal{V}`
        s_plus : np.ndarray
            standard deviation in molar fluorescences for each cycle/well of active reporter :math:`\\sigma^+`, n by number of wells matrix
            Determined in units of fluorescence divided by (pmol/L)
        s_minus : np.ndarray
            standard deviation molar fluorescences for each cycle/well of inactive reporter :math:`\\sigma^-`, n by number of wells matrix
            Determined in units of fluorescence divided by (pmol/L)
        kwargs : dict
            extra kwargs for Amplification class
        """
        Amplification.__init__(self, R, pbar, E_U0, V_U0)
        self.b = f_minus * C
        self.db = s_minus * C
        Vol = Vol * 1e-12  # Tera Liter (TL). 1 pmol/L = 1 mol / TL
        self.d = (f_plus - f_minus) / Vol / N_av
        self.dd = np.sqrt(s_plus*s_plus + s_minus*s_minus) / Vol / N_av

        n, m = self.b.shape
        assert ((n == self.d.shape[0]) and (m == self.d.shape[1])), "Inconsistent shapes"
        self.n = n
        self.m = m
        self.E_F = np.zeros((self.n, self.m))
        self.V_F = np.zeros((self.n, self.m))
        self.E_DX = np.zeros(self.n)
        self.V_DX = np.zeros(self.n)
        self.cycles = np.array([i + 1 for i in range(self.n)])

    def get_E_DX(self, i) -> float:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        float
            :math:`\\mathbb{E}\\left[\\Delta X_i\\right]`

        """
        return self.get_E_Ui(i)[0] - self.E_U0[0]

    def get_Cov_XiX0(self, i):
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        float
            :math:`\\mathsf{Cov}\\left[X_i, X_0\\right]`, see (43)

        """
        return self.V_U0[0, 0] / 2. * (pow(self.l1, i) + pow(self.l2, i))

    def get_V_DX(self, i):
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        float
            :math:`\\mathsf{Var}\\left[\\Delta X_i\\right]`

        """
        return self.get_V_Ui(i)[0, 0] + self.V_U0[0, 0] - 2 * self.get_Cov_XiX0(i)

    def calculate(self):
        """
        Performs calculations, filling in :code:`E_F`, :code:`V_F`, :code:`E_DX`, and :code:`V_DX`.

        """
        for im1 in range(self.n):
            self.E_DX[im1] = self.get_E_DX(im1 + 1)
            self.V_DX[im1] = self.get_V_DX(im1 + 1)
            self.E_F[im1, :] = self.b[im1, :] + self.d[im1, :]*self.E_DX[im1]
            self.V_F[im1, :] = self.d[im1, :] * self.d[im1, :] * self.V_DX[im1]


def plot_fluorescence_curve(kls: HydrolysisProbes, ax: plt.axis, wm1: int):
    """

    Parameters
    ----------
    kls : HydrolysisProbes
        Contains all amplification data. Has already computed values
    ax : matplotlib.axes
        axes to plot on
    wm1 : int
        well number


    """

    s = np.sqrt(kls.V_F[:, wm1])
    kwargs = dict(lw=1)
    ax.plot(kls.cycles, kls.E_F[:, wm1], '-', color='magenta', label='$\\kappa = 0$', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] + s, '--', color='C0', label='$\\kappa = 1$', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] - s, '--', color='C0', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] + 2*s, '-.', color='C1', label='$\\kappa = 2$', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] - 2*s, '-.', color='C1', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] + 3*s, ls='dotted', color='tab:red', label='$\\kappa = 3$', **kwargs)
    ax.plot(kls.cycles, kls.E_F[:, wm1] - 3*s, ls='dotted', color='tab:red', **kwargs)
    ax.fill_between(kls.cycles, kls.E_F.min(axis=1), kls.E_F.max(axis=1), color='lightgrey')
