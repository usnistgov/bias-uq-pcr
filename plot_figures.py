import numpy as np
import os
from matplotlib import pyplot as plt
from src.globals import CYCLE_SYMBOL
from src.amplification import Amplification
from src.molar_fluorescence import MolarFluorescence
from src.wells import well_to_number, number_to_well


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_figure2(R=0.9, pbar=0.85, max_cycle=4):
    r"""

    Parameters
    ----------
    R : float, optional
        choice for value of :math:`R`, defaults to 0.9
    pbar : float, optional
        choice for value of :math:`\bar{p}`, defaults to 0.85
    max_cycle : int, optional
        choice for maximum cycle to plot, defaults to 4

    """
    cycles = np.array(list(range(max_cycle + 1)))

    fig, ax = plt.subplots(figsize=(3.25, 3.25))

    for (EX0, EY0, label, marker) in [
        (10, 10, r"$\mathbb{E}\left[N_0\right] = \mathbb{E}\left[X_0\right]/2$", 'x'),
        (10, 0, r"$\mathbb{E}\left[N_0\right] = \mathbb{E}\left[X_0\right]$", 'o'),
        (0, 10, r"$\mathbb{E}\left[N_0\right] = \mathbb{E}\left[Y_0\right]$", 'd')
    ]:
        cls = Amplification(R, pbar, np.array([[EX0], [EY0]]), np.random.random((2, 2)))
        ax.plot(cycles, cls.get_EXi_over_EYi(cycles), marker, label=label, mfc='None', ls='-')

    ax.legend(loc=(0.4, 0.4), edgecolor='None', facecolor='None')
    ax.set_xlabel("Cycle, $%s$" % CYCLE_SYMBOL)
    ax.set_ylabel(
        r"$\dfrac{\mathbb{E}\left[X_%s\right]}{\mathbb{E}\left[Y_%s\right]}$" % (CYCLE_SYMBOL, CYCLE_SYMBOL),
        rotation=0., labelpad=16
    )
    ax.annotate("$R=%3.2f$" % R, xy=(2, 0.2), xycoords='data')
    ax.annotate("$\\bar{p}=%3.2f$" % pbar, xy=(2, 0.1), xycoords='data')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.subplots_adjust(left=0.24, top=0.97, bottom=0.14, right=0.97)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(BASE_DIR, "out", "Fig2.png"), transparent=True, dpi=300)


def plot_figure4(plus: MolarFluorescence, minus: MolarFluorescence):
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
    fig.savefig(os.path.join(BASE_DIR, 'out', "Fig4.png"), transparent=True, dpi=300)


if __name__ == '__main__':
    plot_figure2()
    Plus = MolarFluorescence(
        np.array([0.01, 0.02, 0.04, 0.07]),
        [
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "10_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "20_nM", "CDC_HID2-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "40_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "FAM", "25_vol_pt_TE", "70_nM", "CDC_HID2-1.xls")
        ], "+"
    )
    Minus = MolarFluorescence(
        np.array([0.04, 0.08, 0.16]),
        [
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "40_nM", "CDC_HID2-1.xls"),
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "80_nM", "CDC_HID1-1.xls"),
            os.path.join(BASE_DIR, "data", "Probe", "25_vol_pt_TE", "160_nM", "CDC_HID2-1.xls"),
        ], "-"
    )

    Plus.calculate()
    Minus.calculate()
    Plus.save(os.path.join(BASE_DIR, "out"))
    Minus.save(os.path.join(BASE_DIR, "out"))

    plot_figure4(Plus, Minus)
