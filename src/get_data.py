import pandas as pd
from .wells import ORDERED_WELLS
import numpy as np

F_SCALE = 1e6


def file_to_numpy(file_abs_path: str) -> np.ndarray:
    sheet = pd.ExcelFile(file_abs_path)
    df = sheet.parse(
        sheet_name="Raw Data", header=7
    )
    df = df.pivot(index='Cycle', columns="Well", values='1') / F_SCALE
    df = df[ORDERED_WELLS]
    return df.to_numpy()


def cycle_to_index(cycle: int) -> int:
    return cycle - 1


def index_to_cycle(index: int) -> int:
    return index + 1


def conc_dir_to_conc(conc_dir: str) -> float:
    return float(conc_dir[:conc_dir.find("_")]) / 1000.


if __name__ == '__main__':
    import os

    file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "data", "FAM", "25_vol_pt_TE", "70_nM", "CDC_HID2-1.xls"
    )
    df = file_to_numpy(file)
    print(df)
    # file = os.path.join("")
