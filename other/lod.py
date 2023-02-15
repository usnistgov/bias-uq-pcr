import numpy as np


def my_int(x):
    """

    >>> my_int(1.1)
    2
    >>> my_int(1.)
    1
    >>> my_int(2.)
    2
    >>> my_int(1.999)
    2

    """
    return np.max( (1, int(np.ceil(x)))  )


class Base:
    def beta(self, kappa, R, pbar, r):
        raise NotImplemented

    def alpha(self, kappa, R):
        raise NotImplemented

    def get_L(self, chi, kappa, R, pbar, r):
        return my_int(
            chi*self.alpha(kappa, R) + self.beta(kappa, R, pbar, r)
        )


class DNA(Base):
    def beta(self, kappa, R, pbar, r):
        l1 = 1 + pbar
        l2 = 1 - pbar
        return kappa*kappa*(
            l2/2/l1 + l1/(2*l1 + l2)*(1-R)*(1-R)/(1+R)/(1+R)
        )
    
    def alpha(self, kappa, R):
        return kappa*kappa*(1 + R*R)/(1 + R)/(1 + R)

class fRNA(Base):
    def beta(self, kappa, R, pbar, r):
        l1 = 1 + pbar
        l2 = 1 - pbar
        return kappa*kappa*(
            l2/l1*(1 + R)/2/R/r
            - l1/(2*l1 + l2)*(1-R)/2/R/r
            + (1-r)/r
        )
    
    def alpha(self, kappa, R):
        return kappa*kappa


class rRNA(fRNA):
    def beta(self, kappa, R, pbar, r):
        l1 = 1 + pbar
        l2 = 1 - pbar
        return kappa*kappa*(
            l2/l1*(1 + R)/2/r
            + l1/(2*l1 + l2)*(1-R)/2/r
            + (1-r)/r
        )



if __name__ == '__main__':
    import doctest
    doctest.testmod()


    Rs = np.linspace(0.9, 1.1)
    ps = np.linspace(0.8, 0.99)
    rs = np.linspace(0.2, 0.99)
    L_min = 100
    L_max = -100
    kappa = 3; chi = 1
    d = DNA()
    for R in Rs:
        for p in ps:
            L = d.get_L(chi, kappa, R, p, None)
            if L < L_min:
                L_min = L
            if L > L_max:
                L_max = L
    print('lds', L_min, L_max, np.sqrt(chi/L_min), np.sqrt(chi/L_max))

    
    d = fRNA()
    L_min = 100
    L_max = -100
    for r in rs:
        for R in Rs:
            for p in ps:
                L = d.get_L(chi, kappa, R, p, r)
                if L < L_min:
                    L_min = L
                if L > L_max:
                    L_max = L
    
    print('frna', L_min, L_max, np.sqrt(chi/L_min), np.sqrt(chi/L_max))

    d = rRNA()
    L_min = 100
    L_max = -100
    for r in rs:
        for R in Rs:
            for p in ps:
                L = d.get_L(chi, kappa, R, p, r)
                if L < L_min:
                    L_min = L
                if L > L_max:
                    L_max = L
    
    print('rrna', L_min, L_max, np.sqrt(chi/L_min), np.sqrt(chi/L_max))