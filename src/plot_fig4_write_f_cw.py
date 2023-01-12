from get_data import file_to_numpy, conc_dir_to_conc
import typing
import matplotlib.pyplot as plt
from globals import CYCLE_SYMBOL
from wells import number_to_well, well_to_number
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Reporter:
    def __init__(self, C: np.array, files: typing.List[str], name: str):
        """

        Parameters
        ----------
        C : np.array
            concentrations of reporter in pmol/L
        files : typing.List[str]
            list of file names (absolute paths)
        name : str
            name of reporter (e.g., FAM, Probe)
        """
        self.n = 45
        self.num_wells = 96
        self.C = C
        self.q = C.shape[0]
        assert self.q > 1, "Must have more than one dataset!"
        self.F = np.zeros((self.n, self.num_wells, self.q))
        for i, f in enumerate(files):
            self.F[:, :, i] = file_to_numpy(f)

        self.f = np.zeros((self.n, self.num_wells))
        self.df = np.zeros((self.n, self.num_wells))
        self.name = name

    def calculate(self):
        for i in range(self.n):
            for w in range(self.num_wells):
                self.f[i, w] = np.inner(self.F[i, w, :], self.C) / np.inner(self.C, self.C)
                R = self.F[i, w, :] - self.f[i, w] * self.C
                self.df[i, w] = np.sqrt(np.inner(R, R) / (self.q - 1))

    def save(self, dir: str):
        with open(os.path.join(dir, "f_iw_%s.npy" % self.name), "wb") as file:
            np.save(file, self.f)

        with open(os.path.join(dir, "df_iw_%s.npy" % self.name), "wb") as file:
            np.save(file, self.df)


def plot_figure4(plus: Reporter, minus: Reporter):
    fig, axes = plt.subplots(figsize=(4.68504, 2.5), ncols=3, sharex=False, sharey=True)
    cycles_to_plot = [1, 20, 40]
    wells_to_plot = (
        'A1', 'C10', 'E5', 'G3', 'H11'
    )

    for i, ax in zip(
            list(i - 1 for i in cycles_to_plot),
            axes
    ):
        for w, marker, color in zip(
                list(map(well_to_number, wells_to_plot)),
                ['^', 'o', 'v', '*', 'd'],
                ['C0', 'C1', 'C2', 'C3', 'C4']
        ):
            kwargs = dict(color=color, clip_on=False)
            ax.plot(plus.C, plus.F[i, w], marker, mfc='None', **kwargs)
            C = np.linspace(0, plus.C.max())
            ax.fill_between(C, (plus.f[i, w] - 3 * plus.df[i, w]) * C,
                                (plus.f[i, w] + 3 * plus.df[i, w]) * C,
                                alpha=0.3, **kwargs)
            ax.plot(C, plus.f[i, w] * C, '-', label="%s" % number_to_well(w), **kwargs)

            ax.plot(minus.C, minus.F[i, w], marker, mfc='None', **kwargs)
            C = np.linspace(0, minus.C.max())
            ax.fill_between(C, (minus.f[i, w] - 3 * minus.df[i, w]) * C,
                                (minus.f[i, w] + 3 * minus.df[i, w]) * C,
                                alpha=0.3, **kwargs)
            ax.plot(C, minus.f[i, w] * C, '-', **kwargs)

    axes[0].legend(loc=(0.01, 0.97), edgecolor="None", facecolor='None', ncol=len(wells_to_plot),
                 handleheight=1.5, labelspacing=0.3, columnspacing=1.5,  # handlelength=1.
                 )

    axes[0].set_ylabel("$\\frac{F_{%s,w}}{10^6}$" % CYCLE_SYMBOL,
                     rotation=0, fontsize=16, labelpad=16)
    for i in range(3):
        axes[i].set_xlabel("$C$ [pmol/L]")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_ylim([0., 1.6])
        axes[i].set_yticks([0., 0.4, 0.8, 1.2, 1.6])
        axes[i].set_xticks([0., 0.08, 0.16])
        axes[i].set_xlim([0., 0.16])
        axes[i].tick_params(which='both', direction='in')
        axes[i].annotate("$%s=%i$" % (CYCLE_SYMBOL, cycles_to_plot[i]), xy=(0.6, 0.04), xycoords='axes fraction')
    axes[0].annotate("Inactive", xy=(0.73, 0.55), xycoords='axes fraction',
                   xytext=(0.55, 0.25), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle="->"))
    axes[0].annotate("Active", xy=(0.3, 0.85), xycoords='axes fraction',
                   xytext=(0.05, 0.95), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle="-[", connectionstyle="angle"))
    axes[1].set_xticklabels(["", "0.08", "0.16"])
    axes[2].set_xticklabels(["", "0.08", "0.16"])
    fig.subplots_adjust(left=0.14, right=0.96, top=0.94, bottom=0.16, wspace=0.04)
    fig.savefig(os.path.join(BASE_DIR, '..', 'out', "Fig4.png"), transparent=True, dpi=300)


if __name__ == '__main__':
    plus = Reporter(
        np.array([0.01, 0.02, 0.04, 0.07]),
        [
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "10_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "20_nM", "CDC_HID2-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "40_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "70_nM", "CDC_HID2-1.xls")
        ], "+"
    )
    minus = Reporter(
        np.array([0.04, 0.08, 0.16]),
        [
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "40_nM", "CDC_HID2-1.xls"),
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "80_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "160_nM", "CDC_HID2-1.xls"),
        ], "-"
    )

    plus.calculate()
    minus.calculate()
    plus.save(os.path.join(BASE_DIR, "..", "out"))
    minus.save(os.path.join(BASE_DIR, "..", "out"))

    plot_figure4(plus, minus)
