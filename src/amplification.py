import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def eval_R(p_fr: float, p_rf: float):
    return np.sqrt(p_rf/p_fr)

def get_pfr_prf(R: float, pbar: float):
    p_fr = pbar / R
    p_rf = R*pbar
    return p_fr, p_rf

def eval_E_Ui_brute_force(i: int, A: np.ndarray, E_U0: np.ndarray) -> np.ndarray:
    if i == 1:
        return A@E_U0
    
    return A@eval_E_Ui_brute_force(i-1, A, E_U0)

def eval_E_S0(E_U0: np.ndarray, R: float):
    return E_U0[0, 0] + R*E_U0[1, 0]

def eval_E_D0(E_U0: np.ndarray, R: float):
    return E_U0[0, 0] - R*E_U0[1, 0]

def eval_V_Ui_old(V_Uim1: np.ndarray, E_Uim1: np.ndarray, R: float, pbar:float):
    p_fr, p_rf = get_pfr_prf(R, pbar)
    V_Xi = V_Uim1[0, 0] + 2*p_rf*V_Uim1[1, 0] + p_rf*p_rf*V_Uim1[1, 1] + p_rf*(1-p_rf)*E_Uim1[1, 0]
    V_Yi = V_Uim1[1, 1] + 2*p_fr*V_Uim1[1, 0] + p_fr*p_fr*V_Uim1[0, 0] + p_fr*(1-p_fr)*E_Uim1[0, 0]
    C_YXi = V_Uim1[1, 0]*(1 + p_rf*p_fr) + p_fr*V_Uim1[0, 0] + p_rf*V_Uim1[1, 1]
    return np.array([
        [V_Xi, C_YXi],
        [C_YXi, V_Yi]
    ])


def eval_V_Ui_new(V_Uim1: np.ndarray, E_Uim1: np.ndarray, R: float, pbar:float):
    p_fr, p_rf = get_pfr_prf(R, pbar)
    A = np.array([
        [1, p_rf],
        [p_fr, 1]
    ])
    E_Sim1, E_Dim1 = eval_E_S0(E_Uim1, R), eval_E_D0(E_Uim1, R)
    return A@V_Uim1@A.T \
        + pbar*E_Sim1/2*np.array([
        [(1-R*pbar), 0],
        [0, (1-pbar/R)/R]
    ]) \
        + pbar*E_Dim1/2*np.array([
        [-(1-R*pbar), 0],
        [0, (1-pbar/R)/R]
    ])

def eval_V_Ui_brute_force(
        i: int, A: np.ndarray, V_U0: np.ndarray, E_U0: np.ndarray,
        K1: np.ndarray, K2: np.ndarray,
        p: float, R: float
    ) -> np.ndarray:
    E_S0, E_D0 = eval_E_S0(E_U0, R), eval_E_D0(E_U0, R)
    V_Binomials = p*(E_S0*pow(1 + p, i-1)*K1 + E_D0*pow(1 - p, i-1)*K2)
    if i == 1:
        return A@V_U0@A.T + V_Binomials
    
    return A@eval_V_Ui_brute_force(i-1, A, V_U0, E_U0, K1, K2, p, R)@A.T + V_Binomials


class AmplificationNew:
    """Base class for amplification

    >>> p_fr, p_rf = 0.87, 0.92
    >>> R = eval_R(p_fr, p_rf)
    >>> pbar = np.sqrt(p_rf*p_fr)
    >>> E_U0 = np.array([[3], [4]])
    >>> V_U0 = np.array([[3, 0], [0, 4]])
    >>> cls = AmplificationNew(R, pbar, E_U0, V_U0)
    >>> A = np.array([[1, p_rf],[p_fr, 1]])
    >>> np.max(np.abs(A - cls.get_A())) < 1e-14
    True
    >>> np.max(np.abs(A - cls.get_Atoi(1))) < 1e-14
    True
    >>> np.max(np.abs(A@A - cls.get_Atoi(2))) < 1e-14
    True

    >>> np.max(np.abs(eval_E_Ui_brute_force(5, A, E_U0) - cls.get_E_Ui(5))) < 1e-10
    True
    >>> args = A, cls.V_U0, cls.E_U0, cls.K1, cls.K2, pbar, R
    >>> np.max(np.abs(eval_V_Ui_brute_force(5, *args) - cls.get_V_Ui(5))) < 1e-10
    True
    
    # test nu
    >>> E_S0 = eval_E_S0(cls.E_U0, R)
    >>> cls.get_nu()/E_S0/E_S0 - cls.get_V_Ui(40)/cls.get_E_Ui(40)/cls.get_E_Ui(40)

    >>> cls.V_U0 = np.random.random((2, 2))*10
    >>> cls.V_U0[0, 1] = cls.V_U0[1, 0]
    >>> cls.E_U0 = np.random.random((2, 1))*10
    >>> np.max(np.abs(eval_V_Ui_new(cls.V_U0, cls.E_U0, R, pbar) - eval_V_Ui_old(cls.V_U0, cls.E_U0,  R, pbar))) < 1e-10
    True
    >>> np.max(np.abs(cls.get_V_Ui(1) - eval_V_Ui_old(cls.V_U0, cls.E_U0, R, pbar))) < 1e-10
    True

    """
    def __init__(self, R: float, pbar: float, E_U0: np.ndarray, V_U0: np.ndarray, farray=np.array, sqrt2=np.sqrt(2)) -> None:
        self.R = R
        self.pbar = pbar
        self.E_U0 = E_U0
        self.V_U0 = V_U0

        self.x1 = farray([[R], [1]])/sqrt2
        self.x2 = farray([[R], [-1]])/sqrt2
        self.z1 = farray([[1/R], [1]])/sqrt2
        self.z2 = farray([[1/R], [-1]])/sqrt2
        self.l1 = 1 + pbar
        self.l2 = 1 - pbar

        self.K1 = 1/2*farray([
            [1-R*pbar, 0],
            [0, 1/R - pbar/R/R]
        ]
        )
        self.K2 = 1/2*farray([
            [-1 + R*pbar, 0],
            [0, 1/R - pbar/R/R]
        ]
        )
    
    def get_A(self) -> np.ndarray:
        return self.l1*self.x1@self.z1.T + self.l2*self.x2@self.z2.T
    
    def get_Atoi(self, i: int) -> np.ndarray:
        return pow(self.l1, i)*self.x1@self.z1.T + pow(self.l2, i)*self.x2@self.z2.T
    
    def get_E_Ui(self, i: int) -> np.ndarray:
        return self.get_Atoi(i)@self.E_U0
    
    def get_x1z1TK1z1x1T(self):
        return self.l2*(self.R + 1)/8*np.array([
            [1, 1/self.R],
            [1/self.R, 1/self.R/self.R]
        ])
    
    def get_x1z1TK2z1x1T(self):
        return self.l1*(self.R - 1)/8*np.array([
            [1, 1/self.R],
            [1/self.R, 1/self.R/self.R]
        ])
    
    def get_V_Ui(self, i: int) -> np.ndarray:
        E_S0, E_D0 = eval_E_S0(self.E_U0, self.R), eval_E_D0(self.E_U0, self.R)
        return self.get_Atoi(i)@self.V_U0@(self.get_Atoi(i).T) \
            + self.pbar*E_S0*(
                (pow(self.l1*self.l1, i) - pow(self.l1, i))/(self.l1*self.l1 - self.l1)*self.x1@self.z1.T@self.K1@self.z1@self.x1.T
                +
                (pow(self.l1*self.l2, i) - pow(self.l1, i))/(self.l1*self.l2 - self.l1)*self.x1@self.z1.T@self.K1@self.z2@self.x2.T
                +
                (pow(self.l1*self.l2, i) - pow(self.l1, i))/(self.l1*self.l2 - self.l1)*self.x2@self.z2.T@self.K1@self.z1@self.x1.T
                +
                (pow(self.l2*self.l2, i) - pow(self.l1, i))/(self.l2*self.l2 - self.l1)*self.x2@self.z2.T@self.K1@self.z2@self.x2.T
                ) \
            + self.pbar*E_D0*(
                (pow(self.l1*self.l1, i) - pow(self.l2, i))/(self.l1*self.l1 - self.l2)*self.x1@self.z1.T@self.K2@self.z1@self.x1.T
                +
                (pow(self.l1*self.l2, i) - pow(self.l2, i))/(self.l1*self.l2 - self.l2)*self.x1@self.z1.T@self.K2@self.z2@self.x2.T
                +
                (pow(self.l1*self.l2, i) - pow(self.l2, i))/(self.l1*self.l2 - self.l2)*self.x2@self.z2.T@self.K2@self.z1@self.x1.T
                +
                (pow(self.l2*self.l2, i) - pow(self.l2, i))/(self.l2*self.l2 - self.l2)*self.x2@self.z2.T@self.K2@self.z2@self.x2.T
                ) 
    
    def get_EXi_over_EYi(self, cycles: np.array) -> np.array:
        r"""Calculate :math:`\mathbb{E}\left[X_i\right]/\mathbb{E}\left[Y_i\right]` for a variety of cycles :math:`i`

        >>> R = 0.9
        >>> pbar = 0.85
        >>> l1 = 1 + pbar
        >>> l2 = 1 - pbar
        >>> for (EX0, EY0) in [(10, 10), (10, 0), (0, 10)]:
        >>> EX0, EY0 = 10, 10
        >>> a = EX0 - R*EY0
        >>> 



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
            EXi_over_EYi[i] = EUi[0, 0]/EUi[1, 0]
        return EXi_over_EYi

    
    def get_nu(self):
        return self.V_U0[0, 0] + self.R*self.V_U0[1, 0] + self.R*self.V_U0[0, 1] + self.R*self.R*self.V_U0[1, 1] \
            + eval_E_S0(self.E_U0, self.R)*(1 + self.R)/2*self.l2/self.l1 \
            + eval_E_D0(self.E_U0, self.R)*(1 - self.R)/2*self.l1/(2*self.l1 + self.l2)


class HydrolysisProbes(AmplificationNew):
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
        AmplificationNew.__init__(self, *args, **kwargs)
    
    def get_E_DX(self, i):
        return self.get_E_Ui(i)[0, 0] - self.E_U0[0, 0]
    
    def get_Cov_XiX0(self, i):
        # see rocketbook `Covariance with initial`
        return self.V_U0[0, 0]/2.*(pow(self.l1, i) + pow(self.l2, i))
    
    def get_V_DX(self, i):
        return self.get_V_Ui(i)[0, 0] + self.V_U0[0, 0] - 2*self.get_Cov_XiX0(i)


def formula_sympy():
    import sympy as s
    R = s.Symbol("R", positive=True, real=True)
    pbar = s.Symbol("p", positive=True, real=True)
    E = np.ones((2, 1))
    V = np.ones((2, 2))
    cls = AmplificationNew(R, pbar, E, V, farray=s.Matrix, sqrt2=s.sqrt(2))
    s.pprint(s.simplify(cls.x1@cls.z1.T@cls.K1@cls.z1@cls.x1.T - cls.get_x1z1TK1z1x1T()))

    s.pprint(s.simplify(cls.x1@cls.z1.T@cls.K2@cls.z1@cls.x1.T - cls.get_x1z1TK2z1x1T()))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # formula_sympy()