import numpy as np
from globals import CYCLE_SYMBOL
import matplotlib.pyplot as plt
import os
from amplification import HydrolysisProbes

from wells import well_to_number

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_E_V(R, pbar, E_Y0, E_X0, V_Y0, V_X0, f_cw_plus, f_cw_minus):
    # def __init__(self, R: float, pbar: float, E_U0: np.ndarray, V_U0: np.ndarray, farray=np.array, sqrt2=np.sqrt(2)) -> None:
    cls = HydrolysisProbes(
        R, pbar, np.array([[E_X0], [E_Y0]]),
        np.array([
            [V_X0, 0],
            [0, V_Y0]
        ])
    )

    C = 0.125  # pM
    Vol = 20e-6  # L solution
    Vol = Vol * 1e-12  # Tera Liter (TL). 1 pmol/L = 1 mol / TL
    N_av = 6.022e23
    n, m = f_cw_plus.shape

    cycles = np.array([i + 1 for i in range(n)])
    b_cw = f_cw_minus * C
    d_cw = (f_cw_plus - f_cw_minus) / Vol / N_av
    print("mean d_cw", d_cw.mean())
    print("mean b_cw", b_cw.mean())

    E = np.zeros((n, m))
    V = np.zeros((n, m))
    E_DXc = np.array([cls.get_E_DX(i + 1) for i in range(n)])
    V_DXc = np.array([cls.get_V_DX(i + 1) for i in range(n)])
    for w in range(m):
        E[:, w] = b_cw[:, w] + d_cw[:, w] * E_DXc
        V[:, w] = d_cw[:, w] * d_cw[:, w] * V_DXc

    well = 'A1'
    iwell = well_to_number(well)
    if E_Y0 + E_X0 > 0:
        print("last cycle", E[-1, iwell] - 3 * np.sqrt(V[-1, iwell]) - b_cw[-1, iwell])
        print("CV, final", V_DXc[-1] / E_DXc[-1] / E_DXc[-1], "1/9=", 1 / 9)
        V_Un, E_Un = cls.get_V_Ui(n), cls.get_E_Ui(n)
        print("CV, new,", V_Un[0, 0] / E_Un[0, 0] / E_Un[0, 0])

    return cycles, E[:, iwell], V[:, iwell], E.max(axis=1), E.min(axis=1), b_cw


def plot_curve(R, pbar, E_Y0, E_X0, V_Y0, V_X0, f_cw_plus, f_cw_minus, ax,
               background_only=False, background=True):
    cycles, E, V, Emax, Emin, b_cw = calculate_E_V(
        R, pbar, E_Y0, E_X0, V_Y0, V_X0, f_cw_plus, f_cw_minus
    )

    if not background_only:
        if background:
            ax.plot(cycles, Emax, ls='dotted', color="blue")
            ax.plot(cycles, Emin, ls='dotted', color="blue")

        s = np.sqrt(V)
        ax.plot(cycles, E, '-', color='k')
        ax.fill_between(cycles, E - 3 * s, E + 3 * s, color='lightgrey')

        # print CT line
        F_t = (E[0] + 3 * s[0]) * 1.2
        it = 0
        while E[it] < F_t:
            it += 1
        print("CT = ", it + 1, np.log(E_Y0 + E_X0))
    else:
        ax.plot(cycles, b_cw.max(axis=1), ls='dotted', color="purple")
        ax.plot(cycles, b_cw.min(axis=1), ls='dotted', color="purple")
        ax.plot(cycles, b_cw.mean(axis=1), ls='solid', color="purple")


if __name__ == '__main__':

    with open(os.path.join(BASE_DIR, '..', 'out', "f_cw_FAM.npy"), 'rb') as file:
        f_cw_plus = np.load(file)

    with open(os.path.join(BASE_DIR, '..', 'out', "f_cw_Probe.npy"), 'rb') as file:
        f_cw_minus = np.load(file)


    def fmt_ax(ax):
        ax.semilogy()
        ax.set_ylim([0.3, 100])
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.set_xticks([1, 6, 11, 16, 21, 26, 31])
        ax.set_xlim([1, 26])


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(4.68504, 3.5), sharex=True, sharey=True)
    R, pbar = 1.0, 0.9
    p = 0
    for N_0, icol in zip((128, 32, 16, 8), ((0, 0), (0, 1), (1, 0), (1, 1))):
        plot_curve(R, pbar, N_0 / 2, N_0 / 2, N_0 / 2, N_0 / 2, f_cw_plus, f_cw_minus,
                   ax[icol], background=True)
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
    plot_curve(R, pbar, 0, 0, 0, 0, f_cw_plus, f_cw_minus, axb, background_only=True)
    axb.tick_params(left=True, right=True, top=True,
                    which='both', direction='in', labelleft=False, labelright=True)
    axb.set_xticks([1, 11, 21])
    axb.set_xticks([6, 16], minor=True)
    axb.set_xlim([1, 16])
    axb.set_ylim([0.625, 1.25])
    axb.set_yticks([0.8, 1.0, 1.2])

    fig.subplots_adjust(right=0.98, top=0.98, left=0.12, hspace=0.0, wspace=0.09, bottom=0.11)
    fig.savefig(os.path.join(BASE_DIR, "..", "out", "Fig5.png"), transparent=True, dpi=300)

    # for icol in ((0, 0), (0, 1), (1, 0), (1, 1)):
    #     ax[icol].set_yscale('linear')
    #     ax[icol].set_ylim([0.75, 2])

    # fig.savefig(os.path.join(BASE_DIR, "..", "fig", "amplification-ylinear.png"), transparent=True)

    # plot LOD
    # fig, ax = plt.subplots(figsize=(4.68504, 3.25), ncols=2, sharex=True, sharey=True)
    # for (N_0, iax) in [(42, 0), (40, 1)]:
    #     cycles, E, rV_baseline, rV_probe, rV_ampl, rv_combined, V = calculate_E_V(
    #         e, pbar, N_0//2, N_0 //2, N_0 //2, N_0 //2
    #     )
    #     s = np.sqrt(V)
    #     ax[iax].plot(cycles, E, '-', color='k')
    #     ax[iax].fill_between(cycles, E - 3*s, E + 3*s, color='lightgrey')
    #     ax[iax].set_xlabel("$c$")
    #     ax[iax].tick_params(which='both', direction='in')

    # ax[0].annotate(r"$\mathbb{E}\left[N_0\right]=2L=42$", xy=(0.05, 0.9), xycoords='axes fraction')
    # ax[1].annotate(r"$\mathbb{E}\left[N_0\right]=40$", xy=(0.08, 0.9), xycoords='axes fraction')
    # ax[0].semilogy()
    # ax[0].set_ylim([0.4, 100])
    # ax[0].set_ylabel("$\\mu\\left[F_c\\right]$\n$\\pm$\n$3\\sigma\\left[F_c\\right]$", rotation=0., labelpad=14)
    # ax[0].set_xticks([0., 10., 20., 30., 40., 50])
    # ax[0].set_xlim([0., 35.])
    # fig.subplots_adjust(right=0.99, top=0.97, left=0.15, bottom=0.14, wspace=0.05)

    # fig.savefig(os.path.join(BASE_DIR, "..", "fig", "lod.png"), transparent=True)
