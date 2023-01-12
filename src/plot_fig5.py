import numpy as np
from globals import CYCLE_SYMBOL
import matplotlib.pyplot as plt
import os
from src.kinetic_PCR import HydrolysisProbes

from wells import well_to_number

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_curve(kls: HydrolysisProbes, ax, well="A1"):
    w = well_to_number(well)

    ax.plot(kls.cycles, kls.E_F.max(axis=1), ls='dotted', color="blue")
    ax.plot(kls.cycles, kls.E_F.min(axis=1), ls='dotted', color="blue")

    s = np.sqrt(kls.V_F[:, w])
    ax.plot(kls.cycles, kls.E_F[:, w], '-', color='k')
    ax.fill_between(kls.cycles, kls.E_F[:, w] - 3 * s, kls.E_F[:, w] + 3 * s, color='lightgrey')


if __name__ == '__main__':

    with open(os.path.join(BASE_DIR, '..', 'out', "f_iw_+.npy"), 'rb') as file:
        f_plus = np.load(file)

    with open(os.path.join(BASE_DIR, '..', 'out', "f_iw_-.npy"), 'rb') as file:
        f_minus = np.load(file)


    def fmt_ax(ax):
        ax.semilogy()
        ax.set_ylim([0.3, 100])
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.set_xticks([1, 6, 11, 16, 21, 26, 31])
        ax.set_xlim([1, 26])


    C = 0.125  # pM
    Vol = 20e-6  # L solution

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(4.68504, 3.5), sharex=True, sharey=True)
    R, pbar = 1.0, 0.9
    p = 0
    for N_0, icol in zip((128, 32, 16, 8), ((0, 0), (0, 1), (1, 0), (1, 1))):
        cls = HydrolysisProbes(
            C, Vol, f_plus, f_minus, R, pbar,
            np.array([[N_0 / 2], [N_0 / 2]]),
            np.array([[N_0 / 2, 0.], [0., N_0 / 2]])
        )
        cls.calculate()

        plot_curve(cls, ax[icol])
        ax[icol].annotate("$\\mathbb{E}\\left[N_0\\right]=%i$" % N_0, xy=(0.05, 0.06),
                          xycoords='axes fraction')
        print(N_0 / 2)

    for icol in ((0, 0), (0, 1), (1, 0), (1, 1)):
        fmt_ax(ax[icol])
        if icol[1] == 0:
            ax[icol].set_ylabel(
                r"$\mathbb{E}\left[F_{%s,w}\right] \pm 3\sqrt{\mathsf{Var}\left[F_{%s,w}\right]}$"
                % (CYCLE_SYMBOL, CYCLE_SYMBOL)
            )
        if icol[0] == 1:
            ax[icol].set_xlabel("Cycle, $%s$" % CYCLE_SYMBOL)

    # plot background inset
    axb = fig.add_axes([0.14, 0.265, 0.19, 0.25])
    cls = HydrolysisProbes(
        C, Vol, f_plus, f_minus, R, pbar,
        np.zeros((2, 1)), np.zeros((2, 2))
    )
    axb.plot(cls.cycles, cls.b.max(axis=1), ls='dotted', color="purple")
    axb.plot(cls.cycles, cls.b.min(axis=1), ls='dotted', color="purple")
    axb.plot(cls.cycles, cls.b.mean(axis=1), ls='solid', color="purple")
    axb.tick_params(left=True, right=True, top=True,
                    which='both', direction='in', labelleft=False, labelright=True)
    axb.set_xticks([1, 11, 21])
    axb.set_xticks([6, 16], minor=True)
    axb.set_xlim([1, 16])
    axb.set_ylim([0.625, 1.25])
    axb.set_yticks([0.8, 1.0, 1.2])

    fig.subplots_adjust(right=0.98, top=0.98, left=0.12, hspace=0.0, wspace=0.09, bottom=0.11)
    fig.savefig(os.path.join(BASE_DIR, "..", "out", "Fig5.png"), transparent=True, dpi=300)
