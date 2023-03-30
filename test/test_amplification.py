from src.amplification import eval_R, Amplification, get_pfr_prf
import numpy as np


def eval_E_Ui_brute_force(i: int, A: np.ndarray, E_U0: np.ndarray) -> np.ndarray:
    if i == 1:
        return A @ E_U0

    return A @ eval_E_Ui_brute_force(i - 1, A, E_U0)


def eval_V_Ui_brute_force(
        i: int, A: np.ndarray, V_U0: np.ndarray, E_U0: np.ndarray,
        K1: np.ndarray, K2: np.ndarray,
        p: float, R: float
) -> np.ndarray:
    V_Binomials = pow(1 + p, i-1)*K1 + pow(1 - p, i - 1)*K2
    if i == 1:
        return A @ V_U0 @ A.T + V_Binomials

    return A @ eval_V_Ui_brute_force(i - 1, A, V_U0, E_U0, K1, K2, p, R) @ A.T + V_Binomials


p_fr, p_rf = 0.87, 0.92
R = eval_R(p_fr, p_rf)
pbar = np.sqrt(p_fr * p_rf)
E_U0 = np.array([3, 4])
V_U0 = np.array([[3, 0], [0, 4]])
cls = Amplification(R, pbar, E_U0, V_U0)
A = np.array([[1, p_rf], [p_fr, 1]])


def infty_norm(err):
    return np.max(np.max(err))


def test_A():
    assert infty_norm(A - cls.get_A()) < 1e-14, "A is not correct"
    assert infty_norm(A - cls.get_Atoi(1)) < 1e-14, "A is not correct"
    assert infty_norm(A@A - cls.get_Atoi(2)) < 1e-14, "A^2 is not correct"


def test_A_new():
    X = 1/np.sqrt(2)*np.array([[R, R], [1, -1]])
    X_inv = 1/(R*np.sqrt(2))*np.array([[1, R], [1, -R]])
    L = np.array([[1 + pbar, 0], [0, 1-pbar]])
    A_new = X@L@X_inv
    assert infty_norm(A_new - cls.get_A()) < 1e-14, "A is not correct"


def test_E_Ui(i=5):
    error = eval_E_Ui_brute_force(i, A, E_U0) - cls.get_E_Ui(i)
    assert infty_norm(error) < 1e-10, "E Ui not correct"


def test_V_Ui(i=5):
    args = A, V_U0, E_U0, cls.K1, cls.K2, pbar, R
    error = eval_V_Ui_brute_force(i, *args) - cls.get_V_Ui(i)
    assert infty_norm(error) < 1e-10, "V Ui not correct"


def test_EXi_over_EYi():
    R = 0.9
    pbar = 0.85
    l1 = 1 + pbar
    l2 = 1 - pbar

    i = np.array((list(range(5))))

    for (EX0, EY0) in [(10, 10), (10, 0), (0, 10)]:
        a = EX0 + R*EY0
        b = EX0 - R*EY0
        kls = Amplification(R, pbar, np.array([EX0, EY0]), np.random.random((2, 2)))
        EXi = a/2*l1**i + b/2*l2**i
        EYi = a/2/R*l1**i - b/2/R*l2**i
        indices = EYi > 0
        val1 = EXi[indices]/EYi[indices]
        val2 = kls.get_EXi_over_EYi(i)
        error = val1 - val2[indices]
        assert infty_norm(error) < 1e-10, "EXi/EYi not correct!"