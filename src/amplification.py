import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def eval_R(p_fr: float, p_rf: float):
    """Calculate :math:`R` from :math:`p_\\mathrm{rf}` and :math:`p_\\mathrm{fr}`

    Parameters
    ----------
    p_fr : float
        probability of forward to reverse amplification, :math:`p_\\mathrm{fr}`
    p_rf : float
        probability of reverse to forward amplification, :math:`p_\\mathrm{rf}`

    Returns
    -------
    float
        :math:`R = \\sqrt{  \\dfrac{p_\\mathrm{rf}}{p_\\mathrm{fr}}   }`

    """
    return np.sqrt(p_rf / p_fr)


def get_pfr_prf(R: float, pbar: float):
    """Get :math:`p_\\mathrm{fr}` and :math:`p_\\mathrm{rf}` from :math:`R` and :math:`\\bar{p}`

    Parameters
    ----------
    R : float
        square root of ratio of probabilities, :math:`R`
    pbar : float
        geometric mean of probabilities, :math:`\\bar{p}`

    Returns
    -------
    tuple(p_fr, p_rf)

    """
    p_fr = pbar / R
    p_rf = R * pbar
    return p_fr, p_rf


def eval_E_S0(E_U0: np.ndarray, R: float):
    """Calculates :math:`\\mathbb{E}\\left[S_0\\right]` from :math:`\\mathbb{E}\\left[\\mathbf{U}_0\\right]` and :math:`R`

    Parameters
    ----------
    E_U0 : np.ndarray
        Initial expected value vector
    R : float
        square root of ratio of amplification probabilities

    Returns
    -------
    float
        :math:`\\mathbb{E}\\left[S_0\\right] := \\mathbb{E}\\left[X_i + RY_i\\right]`

    """
    return E_U0[0, 0] + R * E_U0[1, 0]


def eval_E_D0(E_U0: np.ndarray, R: float):
    """Calculates :math:`\\mathbb{E}\\left[D_0\\right]` from :math:`\\mathbb{E}\\left[\\mathbf{U}_0\\right]` and :math:`R`

    Parameters
    ----------
    E_U0 : np.ndarray
        Initial expected value vector
    R : float
        square root of ratio of amplification probabilities

    Returns
    -------
    float
        :math:`\\mathbb{E}\\left[D_0\\right] := \\mathbb{E}\\left[X_i - RY_i\\right]`

    """
    return E_U0[0, 0] - R * E_U0[1, 0]


class Amplification:
    """
    Parameters
    ---------
    l1 : float
        Largest eigenvalue of :math:`\\mathbf{A}`, :math:`\\lambda_1`
    l2 : float
        Smallest eigenvalue of :math:`\\mathbf{A}`, :math:`\\lambda_2`
    x1 : np.ndarray
        Right eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_1`, :math:`\\mathbf{x}_1`
    x2 : np.ndarray
        Right eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_2`, :math:`\\mathbf{x}_2`
    z1 : np.ndarray
        Left eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_1`, :math:`\\mathbf{z}_1`
    z2 : np.ndarray
        Left eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_2`, :math:`\\mathbf{z}_2`
    K1 : np.ndarray
        Matrix :math:`\\mathbf{K}_1` defined in text to calculate variance
    K2 : np.ndarray
        Matrix :math:`\\mathbf{K}_2` defined in text to calculate variance
    """
    def __init__(self, R: float, pbar: float, E_U0: np.ndarray, V_U0: np.ndarray,
                 farray=np.array, sqrt2=np.sqrt(2)) -> None:
        """

        Parameters
        ----------
        R : float
            ratio of amplification probabilities, :math:`R`
        pbar : float
            geometric mean of amplification probabilities, :math:`\\bar{p}`
        E_U0 : np.ndarray
            initial expected values of both strands :math:`\\mathbb{E}\\left[\\mathbf{U}_0\\right]`, 2 by 1 matrix
        V_U0 : np.ndarray
            initial variances of both strands :math:`\\mathsf{Var}\\left[\\mathbf{U}_0\\right]`, 2 by 2 matrix
        farray : callable, optional
            function to convert to array type, defaults to np.array
        sqrt2
            square root of 2, defaults to float calculated by np.sqrt(2)
        """
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
        """

        Returns
        -------
        np.ndarray
            matrix :math:`\\mathbf{A}` calculated by decomposition (10)

        """
        return self.l1 * self.x1 @ self.z1.T + self.l2 * self.x2 @ self.z2.T

    def get_Atoi(self, i: int) -> np.ndarray:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.ndarray
            matrix :math:`\\mathbf{A}^i` calculated by decomposition (10)

        """
        return pow(self.l1, i) * self.x1 @ self.z1.T + pow(self.l2, i) * self.x2 @ self.z2.T

    def get_E_Ui(self, i: int) -> np.ndarray:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.ndarray
            matrix :math:`\\mathbb{E}\\left[\\mathbf{U}_i\\right]=\\mathbf{A}^i\\mathbb{E}\\left[\\mathbf{U}_0\\right]`

        """
        return self.get_Atoi(i) @ self.E_U0

    def get_x1z1TK1z1x1T(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            computes :math:`\\mathbf{x}_1\\mathbf{z}_1^\\top\\mathbf{K}_1\\mathbf{z}_1\\mathbf{x}_1^\\top`

        """
        return self.l2 * (self.R + 1) / 8 * np.array([
            [1, 1 / self.R],
            [1 / self.R, 1 / self.R / self.R]
        ])

    def get_x1z1TK2z1x1T(self):
        """

        Returns
        -------
        np.ndarray
            computes :math:`\\mathbf{x}_1\\mathbf{z}_1^\\top\\mathbf{K}_2\\mathbf{z}_1\\mathbf{x}_1^\\top`

        """
        return self.l1 * (self.R - 1) / 8 * np.array([
            [1, 1 / self.R],
            [1 / self.R, 1 / self.R / self.R]
        ])

    def get_V_Ui(self, i: int) -> np.ndarray:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.ndarray
            complicated expression to calculate :math:`\\mathsf{Var}\\left[\\mathbf{U}_i\\right]`

        """
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
        """

        Returns
        -------
        float
            Computes :math:`\\nu`, see Equation (22).

        """
        return self.V_U0[0, 0] + self.R * self.V_U0[1, 0] + self.R * self.V_U0[0, 1] + self.R * self.R * self.V_U0[1, 1] \
               + eval_E_S0(self.E_U0, self.R) * (1 + self.R) / 2 * self.l2 / self.l1 \
               + eval_E_D0(self.E_U0, self.R) * (1 - self.R) / 2 * self.l1 / (2 * self.l1 + self.l2)
