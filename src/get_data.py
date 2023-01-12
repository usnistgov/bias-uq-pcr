import pandas as pd
from .wells import ORDERED_WELLS
import numpy as np

F_SCALE = 1e6


def file_to_numpy(file_abs_path: str) -> np.ndarray:
    """

    Parameters
    ----------
    file_abs_path : str
        path to file name

    Notes
    -----
    All fluorescence values are divided by :math:`10^6` before any analysis

    Returns
    -------
    np.ndarray
        :math:`n` by :math:`m` matrix of fluorescence, :math:`\\mathbf{F}`

    """
    sheet = pd.ExcelFile(file_abs_path)
    df = sheet.parse(
        sheet_name="Raw Data", header=7
    )
    df = df.pivot(index='Cycle', columns="Well", values='1') / F_SCALE
    df = df[ORDERED_WELLS]
    return df.to_numpy()


def cycle_to_index(cycle: int) -> int:
    """

    Parameters
    ----------
    cycle : int
        cycle number :math:`i`, 0 to n

    Returns
    -------
    int
        index in matrix or array


    """
    return cycle - 1


def index_to_cycle(index: int) -> int:
    """

    Parameters
    ----------
    index : int
        index of cycle (0 to n-1)

    Returns
    -------
    int
        cycle number (1 to n)

    """
    return index + 1