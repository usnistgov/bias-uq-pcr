import numpy as np


def my_int(x):
    """

    Parameters
    ----------
    x : float

    Returns
    -------
    int
        :math:`\\min\\left\{y \\in \\mathbb{N} \\mid y \\ge x\\right\}`

    Notes
    -----
    The set :math:`\\mathbb{N} := \{1, 2, \\ldots\}`.


    >>> my_int(0.9)
    1
    >>> my_int(1.1)
    2
    >>> my_int(1.)
    1
    >>> my_int(2.)
    2
    >>> my_int(2.01)
    3
    >>> my_int(1.999)
    2


    """
    return np.max( (1, int(np.ceil(x)))  )


class LOD:
    """
    Parameters
    ----------
    input_type : str
        Type of nucleic acids input. Either :code:`"ds-DNA"` (see (4)),
        :code:`"fs-RNA"` (see (5)), or :code:`"rs-RNA"` (see (6)).

    """
    def __init__(self, input_type: str):
        self.input_type = input_type
        assert self.input_type in ("fs-RNA", "rs-RNA", "ds-DNA"), "Input type not found"

    def alpha(self, R):
        """

        Parameters
        ----------
        R : float
            square-root of amplification efficiency ratio

        Returns
        -------
        float
            :math:`\\alpha` as in (35a)

        """
        if self.input_type == "ds-DNA":
            return (R*R + 1)/(R + 1)/(R + 1)

        return 1.

    def beta(self, R, pbar, r):
        """

        Parameters
        ----------
        R : float
            square-root of amplification efficiency ratio (see (3))
        pbar : float
            geometric mean of amplification efficiencies (see (2))
        r : float
            probability of reverse-transcription

        Returns
        -------
        float
            :math:`\\beta` as in (35b)

        """
        l1 = 1. + pbar
        l2 = 1. - pbar
        if self.input_type == "ds-DNA":
            return l2/2/l1 - (pbar*l1)/(l1*l1 - l2)*(R - 1)*(R-1)/(R+1)/(R + 1)/2
        if self.input_type == "fs-RNA":
            return (1 - r) / r + l2*(R + 1)/l1/2/R/r - (pbar*l1)/(l1*l1 - l2)*(R - 1)/2/R/r

        return (1 - r) / r + l2*(R + 1)/l1/2/r + pbar*l1/(l1*l1 - l2)*(R - 1)/2/r

    def L(self, chi, kappa, R, pbar, r):
        """

        Parameters
        ----------
        chi : float
            parameter relating :math:`\\mathbb{E}[I]` to :math:`\\mathsf{Var}[I]` (see (50))
        kappa : float
            number of standard deviations allowed to be above background, :math:`\\kappa` see (51)
        R : float
            square-root of amplification efficiency ratio (see (3))
        pbar : float
            geometric mean of amplification efficiencies (see (2))
        r : float
            probability of reverse-transcription

        Returns
        -------
        int
            :math:`L` as in (56)

        """
        return my_int(
            kappa*kappa*(chi*self.alpha(R) + self.beta(R, pbar, r))
        )

    def M(self, chi, L):
        """

        Parameters
        ----------
        chi : float
            relationship between expected value and variance
        L : int
            limit of detection from (56)

        Returns
        -------
        float
            :math:`M` as in (57)

        """
        return np.sqrt(chi / L)


def main(chi=1, kappa=3, N=100):
    """Estimate range of LOD due to different assays. Prints out results.

    Parameters
    ----------
    chi : float, optional
        Relationship between expected value an variance, defaults to 1 (Poisson distribution)
    kappa : float, optional
        Number of standard deviations for statistical significance, defaults to 3
    N : int, optional
        number of points in interval, defaults to 100

    """

    print("Type, Lmin, Lmax, Mmin, Mmax")
    for input_type in ["ds-DNA", "fs-RNA", "rs-RNA"]:
        kls = LOD(input_type)
        L_min = 1e6
        L_max = 0
        for R in np.linspace(0.9, 1.1, N):
            for p in np.linspace(0.8, 0.99, N):
                for r in np.linspace(0.2, 0.99, N):
                    L = kls.L(chi, kappa, R, p, r)
                    if L > L_max:
                        L_max = L
                    if L < L_min:
                        L_min = L

        M_max = kls.M(chi, L_min)
        M_min = kls.M(chi, L_max)
        # print out results
        print("%s, %i, %i, %5.3f, %5.3f" % (kls.input_type, L_min, L_max, M_min, M_max))


if __name__ == '__main__':
    main()