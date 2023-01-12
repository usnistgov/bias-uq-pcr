import typing


def well_to_number(well: str) -> int:
    """Returns number of well

    Parameters
    ----------
    well : str
        name of well (e.g., :code:`"A1"`)

    Returns
    -------
    int
        well number (0 to 95)

    """
    return well_to_column(well) + well_to_row(well) * len(COLUMNS)


def number_to_well(number: int) -> str:
    """Get well name from number

    Parameters
    ----------
    number : int
        well number (0 to 95)

    Returns
    -------
    str
        well name (A1 to H12)

    """
    return column_row_to_well(number_to_column(number), number_to_row(number))


def number_to_row(number: int) -> int:
    """Get the row number associated with a well number

    Parameters
    ----------
    number : int
        well number

    Returns
    -------
    int
        row number (0 to 7)

    """
    return number // 12


def number_to_column(number: int) -> int:
    """Get column index associated with well number

    Parameters
    ----------
    number : int
        well number (0 to 95)

    Returns
    -------
    int
        column (0 to 11)

    """
    return number % 12


def column_row_to_well(ix: int, iy: int) -> str:
    """Convert indices to well

    Parameters
    ----------
    ix : int
        x (column) index
    iy : int
        y (row) index

    Returns
    -------
    str
        name of well

    """
    return ROWS[iy] + COLUMNS[ix]


def well_to_column_row(well: str) -> typing.Tuple[int]:
    """Convert well name to column, row

    Parameters
    ----------
    well : str
        name of well

    Returns
    -------
    tuple(int, int)
        column index, row index

    """
    return well_to_column(well), well_to_row(well)


def well_to_row(well: str) -> int:
    """Well to y index

    Parameters
    ----------
    well : str
        well name

    Returns
    -------
    iy : int
        y-index of well (row)

    """
    return ROWS.index(well[0])


def well_to_column(well: str) -> int:
    """Convert well name to column

    Parameters
    ----------
    well : str
        name of well

    Returns
    -------
    ix : int
        x-index for well (row)

    """
    return COLUMNS.index(well[1:])


ROWS = list("ABCDEFGH")
COLUMNS = list("%i" % i for i in range(1, 13))
ORDERED_WELLS = [None] * len(ROWS) * len(COLUMNS)
for i in range(len(ORDERED_WELLS)):
    ORDERED_WELLS[i] = number_to_well(i)