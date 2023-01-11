import typing


def well_to_number(well: str) -> int:
    """Return number of well
    >>> well_to_number("A1")
    0
    >>> well_to_number("A12")
    11
    >>> well_to_number("H12")
    95
    >>> well_to_number("H1")
    84
    """
    return well_to_column(well) + well_to_row(well) * len(COLUMNS)


def number_to_well(number: int) -> str:
    """Return number of well
    >>> number_to_well(0)
    'A1'
    >>> number_to_well(11)
    'A12'
    >>> number_to_well(95)
    'H12'
    >>> number_to_well(84)
    'H1'
    """
    return column_row_to_well(number_to_column(number), number_to_row(number))


def number_to_row(number: int) -> int:
    """
    >>> number_to_row(0)
    0
    >>> number_to_row(1)
    0
    >>> number_to_row(11)
    0
    >>> number_to_row(12)
    1
    >>> number_to_row(95)
    7
    >>> number_to_row(84)
    7
    """
    return number // 12


def number_to_column(number: int) -> int:
    """
    >>> number_to_column(1)
    1
    >>> number_to_column(0)
    0
    >>> number_to_column(11)
    11
    >>> number_to_column(95)
    11
    >>> number_to_column(84)
    0
    """
    return number % 12


def column_row_to_well(ix: int, iy: int) -> str:
    """Convert indices to well
    >>> column_row_to_well(0, 0)
    'A1'
    >>> column_row_to_well(0, 7)
    'H1'
    >>> column_row_to_well(11, 0)
    'A12'
    >>> column_row_to_well(11, 7)
    'H12'
    >>> well_to_row(column_row_to_well(11, 7))
    7
    >>> well_to_column(column_row_to_well(11, 7))
    11
    Parameters
    ----------
    ix : int
        a index
    iy : int
        y index
    Returns
    -------
    well : str
    """
    return ROWS[iy] + COLUMNS[ix]


def well_to_column_row(well: str) -> typing.Tuple[int]:
    return well_to_column(well), well_to_row(well)


def well_to_row(well: str) -> int:
    """Well to y index
    >>> well_to_row('H100')
    7
    >>> well_to_row('H1')
    7
    >>> well_to_row('A1')
    0
    >>> well_to_row('B1')
    1
    >>> well_to_row('Z1')
    Traceback (most recent call last):
        ...
    ValueError: 'Z' is not in list
    Parameters
    ----------
    well : str
        well name
    Returns
    -------
    iy : int
        y-index of well
    """
    return ROWS.index(well[0])


def well_to_column(well: str) -> int:
    """Convert well name to column
    >>> well_to_column('H100')
    Traceback (most recent call last):
        ...
    ValueError: '100' is not in list
    >>> well_to_column('H1')
    0
    >>> well_to_column('A1')
    0
    >>> well_to_column('B12')
    11
    >>> well_to_column('B13')
    Traceback (most recent call last):
        ...
    ValueError: '13' is not in list
    >>> well_to_column('Z1')
    0
    Parameters
    ----------
    well : str
        name of well
    Returns
    -------
    ix : int
        a-index for well
    """
    return COLUMNS.index(well[1:])


ROWS = list("ABCDEFGH")
COLUMNS = list("%i" % i for i in range(1, 13))
ORDERED_WELLS = [None] * len(ROWS) * len(COLUMNS)
for i in range(len(ORDERED_WELLS)):
    ORDERED_WELLS[i] = number_to_well(i)