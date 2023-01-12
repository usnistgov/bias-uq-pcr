from src.wells import (well_to_number, number_to_well,
                    number_to_row, number_to_column,
                       column_row_to_well, well_to_row, well_to_column)


def test_well_to_number():
    assert well_to_number("A1") == 0
    assert well_to_number("A12") == 11
    assert well_to_number("H12") == 95
    assert well_to_number("H1") == 84


def test_number_to_well():
    assert "A1" == number_to_well(0)
    assert "A12" == number_to_well(11)
    assert "H12" == number_to_well(95)
    assert "H1" == number_to_well(84)


def test_number_to_row():
    assert number_to_row(0) == 0
    assert number_to_row(1) == 0
    assert number_to_row(11) == 0
    assert number_to_row(12) == 1
    assert number_to_row(95) == 7
    assert number_to_row(84) == 7


def test_number_to_column():
    assert number_to_column(0) == 0
    assert number_to_column(1) == 1
    assert number_to_column(11) == 11
    assert number_to_column(95) == 11
    assert number_to_column(84) == 0


def test_column_row_to_well():
    assert column_row_to_well(0, 0) == "A1"
    assert column_row_to_well(0, 7) == "H1"
    assert column_row_to_well(11, 0) == "A12"
    assert column_row_to_well(11, 7) == "H12"


def test_well_to_row():
    assert well_to_row(column_row_to_well(11, 7)) == 7
    assert well_to_row("H100") == 7
    assert well_to_row("H1") == 7
    assert well_to_row("A1") == 0
    assert well_to_row("B1") == 1


def test_well_to_column():
    assert well_to_column(column_row_to_well(11, 7)) == 11
    assert well_to_column("H1") == 0
    assert well_to_column("A1") == 0
    assert well_to_column("B12") == 11
    assert well_to_column("Z1") == 0
