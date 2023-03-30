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


class Amplification:
    """
    Parameters
    ---------
    l1 : float
        Largest eigenvalue of :math:`\\mathbf{A}`, :math:`\\lambda_1`
    l2 : float
        Smallest eigenvalue of :math:`\\mathbf{A}`, :math:`\\lambda_2`
    x1 : np.array
        Right eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_1`, :math:`\\mathbf{x}_1`
    x2 : np.array
        Right eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_2`, :math:`\\mathbf{x}_2`
    z1 : np.array
        Left eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_1`, :math:`\\mathbf{z}_1`
    z2 : np.array
        Left eigenvector of :math:`\\mathbf{A}` corresponding to :math:`\\lambda_2`, :math:`\\mathbf{z}_2`
    K1 : np.ndarray
        Matrix :math:`\\mathbf{K}_1` defined in (25)
    K2 : np.ndarray
        Matrix :math:`\\mathbf{K}_2` defined in (25)
    """

    def __init__(self, R: float, pbar: float, E_U0: np.array, V_U0: np.ndarray):
        """

        Parameters
        ----------
        R : float
            ratio of amplification probabilities, :math:`R`
        pbar : float
            geometric mean of amplification probabilities, :math:`\\bar{p}`
        E_U0 : np.array
            initial expected values of both strands :math:`\\mathbb{E}\\left[\\mathbf{U}_0\\right]`, 2 by 1 matrix
        V_U0 : np.ndarray
            initial variances of both strands :math:`\\mathsf{Var}\\left[\\mathbf{U}_0\\right]`, 2 by 2 matrix

        """
        self.R = R
        self.pbar = pbar
        self.E_U0 = E_U0.reshape((2,))
        self.V_U0 = V_U0

        self.X = np.array([[R, R], [1, -1]]) / np.sqrt(2)
        self.Z = np.array([[1, R], [1, -R]]) / R / np.sqrt(2)
        self.x1 = self.X[:, 0]
        self.x2 = self.X[:, 1]
        self.z1 = self.Z[0, :].T
        self.z2 = self.Z[1, :].T
        self.l1 = 1 + pbar
        self.l2 = 1 - pbar
        self.lamda = np.array([[self.l1, 0], [0, self.l2]])
        self.e1 = np.array([1, 0])
        self.e2 = np.array([0, 1])
        self.initialize()

    def initialize(self):
        """Initialize :math:`\\mathbf{K}_\\ell` via (25), :math:`\\nu_{j,k}` via (28a),
        and :math:`\\eta_{j,k}^\\ell` via (28b).


        """
        self.K1 = np.inner(self.z1, self.E_U0) * np.array([
            [self.pbar * self.R * (1 - self.pbar*self.R) * np.inner(self.e2, self.x1), 0],
            [0, self.pbar/ self.R * (1 - self.pbar/self.R) * np.inner(self.e1, self.x1)]
        ])
        self.K2 = np.inner(self.z2, self.E_U0) * np.array([
            [self.pbar*self.R* (1 - self.pbar*self.R) * np.inner(self.e2, self.x2), 0],
            [0, self.pbar/ self.R * (1 - self.pbar/ self.R) * np.inner(self.e1, self.x2)]
        ])
        self.eta_jkl = np.zeros((2, 2, 2))
        self.nu_jk = np.zeros((2, 2))
        for j in range(2):
            for k in range(2):
                self.nu_jk[j, k] = np.inner(self.Z[j, :], self.V_U0 @ self.Z[k, :])
                self.eta_jkl[j, k, 0] = np.inner(self.Z[j, :], self.K1 @ self.Z[k, :]) / (
                            self.lamda[j, j] * self.lamda[k, k] - self.l1)
                self.eta_jkl[j, k, 1] = np.inner(self.Z[j, :], self.K2 @ self.Z[k, :]) / (
                            self.lamda[j, j] * self.lamda[k, k] - self.l2)

    def get_A(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            matrix :math:`\\mathbf{A}` calculated by decomposition (13)

        """
        return self.l1 * np.outer(self.x1, self.z1) + self.l2 * np.outer(self.x2, self.z2)

    def get_Atoi(self, i: int) -> np.ndarray:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.ndarray
            matrix :math:`\\mathbf{A}^i` calculated by decomposition (15)

        """
        return pow(self.l1, i) * np.outer(self.x1, self.z1) + pow(self.l2, i) * np.outer(self.x2, self.z2)

    def get_E_Ui(self, i: int) -> np.array:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.array
            matrix :math:`\\mathbb{E}\\left[\\mathbf{U}_i\\right]=\\mathbf{A}^i\\mathbb{E}\\left[\\mathbf{U}_0\\right]`

        """
        return self.get_Atoi(i) @ self.E_U0

    def get_V_Ui(self, i: int) -> np.ndarray:
        """

        Parameters
        ----------
        i : int
            cycle number

        Returns
        -------
        np.ndarray
            :math:`\\mathsf{Var}\\left[\\mathbf{U}_i\\right]` obtained by Equation (27)

        """
        def prefactor(j: int, k: int):
            return (
                    (self.nu_jk[j, k] + self.eta_jkl[j, k, 0] + self.eta_jkl[j, k, 1]
                    ) * pow(self.lamda[j, j] * self.lamda[k, k], i)
                    - self.eta_jkl[j, k, 0] * pow(self.l1, i)
                    - self.eta_jkl[j, k, 1] * pow(self.l2, i)
            )

        return sum(
            prefactor(j, k) * np.outer(self.X[:, j], self.X[:, k]) for j in range(2) for k in range(2)
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
            if EUi[1] > 0:
                EXi_over_EYi[i] = EUi[0] / EUi[1]
            else:
                EXi_over_EYi[i] = np.inf
        return EXi_over_EYi
