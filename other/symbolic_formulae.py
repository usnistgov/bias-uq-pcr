import numpy as np

from src.amplification import Amplification


def formula_sympy():
    import sympy as s
    R = s.Symbol("R", positive=True, real=True)
    pbar = s.Symbol("p", positive=True, real=True)
    E = np.ones((2, 1))
    V = np.ones((2, 2))
    cls = Amplification(R, pbar, E, V, farray=s.Matrix, sqrt2=s.sqrt(2))
    s.pprint(s.simplify(cls.x1 @ cls.z1.T @ cls.K1 @ cls.z1 @ cls.x1.T - cls.get_x1z1TK1z1x1T()))

    s.pprint(s.simplify(cls.x1 @ cls.z1.T @ cls.K2 @ cls.z1 @ cls.x1.T - cls.get_x1z1TK2z1x1T()))