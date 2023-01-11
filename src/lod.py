import numpy as np


def my_int(x):
    """

    >>> my_int(1.1)
    2
    >>> my_int(1.)
    1
    >>> my_int(1.999)
    2

    """
    if np.abs(int(x) - x) < np.finfo(float).eps:
        return int(x)
    
    return int(np.ceil(x))


def L_ds(R: float, p: float):
    l1 = 1 + p
    l2 = 1 - p
    return my_int(
        9*(
            (1 + R*R)/(1+R)/(1+R)
            + l2/2/l1
            + (1 - R)*(1-R)/(1+R)/(1+R)*l1/(4*l1 + 2*l2)
        )
    )


def L_rt1(r: float, R: float, p: float):
    l1 = 1 + p
    l2 = 1 - p
    return my_int(
        9/r*(
            1
            + (1 + R)/2 * l2/l1
            + (1 - R)/2*l1/(2*l1 + l2)
        )
    )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    L_min = 100
    L_max = -100
    for R in np.linspace(0.9, 1.1):
        for p in np.linspace(0.8, 1):
            L = L_ds(R, p)
            if L < L_min:
                L_min = L
            if L > L_max:
                L_max = L
    
    print('lds', L_min, L_max)
    L_min = 100
    L_max = -100
    for r in np.linspace(0.2, 1.):
        for R in np.linspace(0.9, 1.1):
            for p in np.linspace(0.8, 1):
                L = L_rt1(r, R, p)
                if L < L_min:
                    L_min = L
                if L > L_max:
                    L_max = L
    
    print('lrt', L_min, L_max)