# Overview

The purpose of this code and data is to enable reproduction
and facilitate extension of the computational
results associated with Ref. [1].


# Installation, Testing, and Reproduction of Manuscript

Python can be obtained at [Python.org](https://python.org).
The dependencies required can be installed via

```bash
pip install -r requirements.txt
```

To test the installation, navigate to the parent directory and invoke
```bash
python3 -m pytest --doctest-modules
```

For step-by-step instructions on reproducing the manuscript, run

```bash
python plot_figures.py
```
(also see [here](plot_figures.py)).

# Experimental Data

The raw data can be found in the directory [data/](data/).

# Documentation

The documentation can be compiled using [sphinx](https://www.sphinx-doc.org) in [doc/](doc/).
A pdf version of the documentation is available [here](doc/manual.pdf).

# Citing This Work

To cite the manuscript, use Ref. [1].
To cite the software or experimental data, use Ref. [2].

## References

  1. DeJaco, R. F.; Roberts, M. J.; Romsos, E. L.; Vallone, P. M.; Kearsley, A. J. Reducing Bias and Quantifying Uncertainty in Fluorescence Produced by PCR, *Bulletin of Mathematical Biology*, 2023, [doi: 10.1007/s11538-023-01182-z](https://doi.org/10.1007/s11538-023-01182-z).

  2. DeJaco, R. F. Software and Data associated with ``Reducing Bias and Quantifying Uncertainty in Fluorescence Produced by PCR,'' National Institute of Standards and Technology, 2023, [doi: 10.18434/mds2-2910](https://doi.org/10.18434/mds2-2910).
