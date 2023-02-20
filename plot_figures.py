import numpy as np
import os
from matplotlib import pyplot as plt

from plot_si_figures import plot_SI_figures
from src.globals import CYCLE_SYMBOL
from src.amplification import Amplification
from src.kinetic_PCR import HydrolysisProbes, plot_fluorescence_curve
from src.molar_fluorescence import MolarFluorescence
from src.wells import number_to_well


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


def plot_figure4(cv_plus: np.ndarray, cv_minus: np.ndarray):
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    kwargs = dict(clip_on=False, zorder=100)

    edges = np.histogram_bin_edges(cv_plus, bins='fd') # Freedman Diaconis Estimator
    vals, edges = np.histogram(cv_plus, bins=edges)
    for i in range(1, edges.shape[0]-1):
        ax.plot([edges[i-1], edges[i], edges[i]], [vals[i-1], vals[i-1], vals[i]], '-', color='C0', **kwargs)
    i = edges.shape[0] - 2
    ax.plot([edges[i-1], edges[i]], [vals[i-1], vals[i-1]], '-', color='C0', **kwargs)

    edges = np.histogram_bin_edges(cv_minus, bins='fd') # Freedman Diaconis Estimator
    vals, edges = np.histogram(cv_minus, bins=edges)
    for i in range(1, edges.shape[0]-1):
        ax.plot([edges[i-1], edges[i], edges[i]], [vals[i-1], vals[i-1], vals[i]], ls='dotted', color='C1', **kwargs)
    i = edges.shape[0] - 2
    ax.plot([edges[i-1], edges[i]], [vals[i-1], vals[i-1]], ls='dotted', color='C1', **kwargs)
    ax.annotate("Active, $\\dfrac{\\sigma_{i,w}^{+}}{f_{i,w}^{+}}$", xy=(0.5, 0.8), xycoords="axes fraction", color='C0')
    ax.annotate("Inactive, $\\dfrac{\\sigma_{i,w}^{-}}{f_{i,w}^{-}}$", xy=(0.5, 0.5), xycoords="axes fraction", color='C1')
    ax.set_ylabel("Count")
    ax.set_xlabel("$\\dfrac{\\sigma_{i,w}^{-}}{f_{i,w}^{-}}$ or $\\dfrac{\\sigma_{i,w}^{+}}{f_{i,w}^{+}}$ ")

    fig.subplots_adjust(left=0.18, bottom=0.22, top=0.98, right=0.99)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([0., 50., 100, 150, 200, 250, 300])
    ax.set_ylim([0., 300.])
    fig.savefig(os.path.join(BASE_DIR, "out", "Fig4.png"), transparent=True, dpi=300)


def plot_figure5(model: HydrolysisProbes):
    # plot background signals and such
    fig, axes = plt.subplots(figsize=(4.68504, 4.5), nrows=2, sharex=True, sharey=False)
    for w, color in [(0, 'C0'), (13, "C1"), (26, "C2"), (39, "C3"),
                     (52, "C4"), (65, "C5"), (78, "C6"), (91, "C7")]:
        bscale = 1.
        axes[0].plot(model.cycles, model.b[:, w] / bscale,
                     color=color, label=number_to_well(w))
        axes[0].fill_between(model.cycles,
                             (model.b[:, w] - 2 * model.db[:, w]) / bscale,
                             (model.b[:, w] + 2 * model.db[:, w]) / bscale,
                             alpha=0.3, color=color)
        dscale = 1e-6
        axes[1].plot(model.cycles, model.d[:, w] / dscale, color=color)
        axes[1].fill_between(model.cycles,
                             (model.d[:, w] - 2 * model.dd[:, w]) / dscale,
                             (model.d[:, w] + 2 * model.dd[:, w]) / dscale,
                             alpha=0.3, color=color)
    axes[0].legend(loc=(0.05, -0.25), ncol=4, edgecolor='None', facecolor='None')

    axes[0].set_ylabel("$b_{i,w}$", rotation=0, labelpad=14)
    axes[1].set_ylabel("$\\dfrac{d_{i,w}}{10^6}$", rotation=0, labelpad=14)
    axes[1].set_xlabel("Cycle, $i$")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.99)
    for ax in (axes[0], axes[1]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45])
        ax.set_xlim([1., 45])
        ax.set_xticks([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19,
                       21, 22, 23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39,
                       41, 42, 43, 44], minor=True)
        ax.tick_params(axis="x", direction='in', which='both')
    fig.savefig(os.path.join(BASE_DIR, "out", "Fig5.png"), transparent=True, dpi=300)


def plot_figure6(model: HydrolysisProbes, wm1: int):
    """

    Parameters
    ----------
    model : HydrolysisProbes
        Contains all of the parameters for calculation of fluorescence curves
    wm1 : int
        well number (0 to 95)

    Notes
    -----
    The methods :code:`model.E_U0` and :code:`model.V_U0` are updated in place during plotting of each subplot


    """
    def fmt_ax(ax):
        ax.semilogy()
        ax.set_ylim([0.3, 100])
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.set_xticks([1, 6, 11, 16, 21, 26, 31])
        ax.set_xlim([1, 26])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(4.68504, 3.5), sharex=True, sharey=True)
    for I, icol in zip((64, 16, 8, 4), ((0, 0), (0, 1), (1, 0), (1, 1))):
        model.E_U0 = np.ones((2, 1))*I
        model.V_U0 = np.eye(2)*I
        model.calculate()

        plot_fluorescence_curve(model, ax[icol], wm1)
        ax[icol].annotate("$\\mathbb{E}\\left[I\\right]=%i$" % I, xy=(0.05, 0.06),
                          xycoords='axes fraction')

    for icol in ((0, 0), (0, 1), (1, 0), (1, 1)):
        fmt_ax(ax[icol])
        if icol[1] == 0:
            ax[icol].set_ylabel(
                r"$\mathbb{E}\left[F_{%s,%i}\right] \pm \kappa\sqrt{\mathsf{Var}\left[F_{%s,%i}\right]}$"
                % (CYCLE_SYMBOL, wm1 + 1, CYCLE_SYMBOL, wm1 + 1)
            )
            if icol[0] == 0:
                ax[icol].legend(edgecolor='None', facecolor='None')
        if icol[0] == 1:
            ax[icol].set_xlabel("Cycle, $%s$" % CYCLE_SYMBOL)

    fig.subplots_adjust(right=0.98, top=0.98, left=0.12, hspace=0.0, wspace=0.09, bottom=0.11)
    fig.savefig(os.path.join(BASE_DIR, "out", "Fig6.png"), transparent=True, dpi=300)


def main():
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

    # Plot another figure
    plot_figure4(Plus.cv, Minus.cv)

    C = 0.125
    Vol = 20e-6
    R = 1.0
    pbar = 0.9
    model = HydrolysisProbes(
        C, Vol, Plus.f, Minus.f, R, pbar,
        np.zeros((2, 1)), np.zeros((2, 2)),
        Plus.df, Minus.df
    )

    plot_figure5(model)

    plot_figure6(model, 0)

    plot_SI_figures(Plus, Minus)


if __name__ == '__main__':
    main()