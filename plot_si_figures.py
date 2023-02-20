import os

import numpy as np
from matplotlib import pyplot as plt

from src.molar_fluorescence import MolarFluorescence
from src.wells import number_to_well

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_table_header(wm1):
    return """
    \\begin{table}
        \\caption{Molar Fluorescence Parameters for Well %s ($w=%i$)}
        \\centering
        \\begin{tabular}{c|ll|ll}
            Cycle & \\multicolumn{2}{c|}{Inactive} & \\multicolumn{2}{c}{Active} \\\\
            \\hline
            $i$ & $f_{i,%i}^{-}$ & $\\sigma_{i,%i}^{-}$ &  $f_{i,%i}^{+}$ & $\\sigma_{i,%i}^{+}$ \\\\
            \\hline
    """ % (number_to_well(wm1), wm1 + 1, wm1 + 1, wm1 + 1, wm1 + 1, wm1 + 1)


table_footer = \
    """               \\hline
        \\end{tabular}
    \\end{table}
    \\clearpage
"""


def plot_SI_figures(Plus: MolarFluorescence, Minus: MolarFluorescence):
    """Plot figures and write tables for Supplementary Information

    Parameters
    ----------
    Plus : MolarFluorescence
        fluorescence data associated with active probe
    Minus : MolarFluorescence
        fluorescence data associated with inactive probe

    """
    cycles_to_plot = [1, 5, 10, 15, 20, 25, 30, 35, 40]

    # write text file
    with open(os.path.join(BASE_DIR, 'out', "si-figures-tables.tex"), 'w') as f:
        f.write(
            """
            \\begin{figure}
                \\centering
                \\includegraphics{si-figs/FigS1.png}
                \\caption{
                    Experimental data points (symbols, $(C^\ell, F_{i,1}^\ell)$ for each $\ell = 1$ to $q$
                    at fixed cycle $i$)
                    compared to model for inactive probe (blue, dashed lines)
                    and active probe (orange, straight lines).
                    The subplots depict cycles $i=1, 5, 10, 15, 20, 25, 30,$ and 40
                    in left-to-right and top-to-bottom order.
                    Well $w=1$ is also called well A1.
                }
                \\label{fig:S1}
            \\end{figure}
            \\clearpage
            """
        )
        wm1 = 0
        f.write(get_table_header(wm1))
        for im1 in range(Plus.n):
            dec_plus = 1 + int(-1 * np.floor(np.log10(Plus.df[im1, wm1])))
            dec_minus = 1 + int(-1 * np.floor(np.log10(Minus.df[im1, wm1])))
            f.write(
                "{i:d} & {fm:.{dm1}f} & {dfm:.{dec_minus}f} & {fp:.{dp1}f} & {dfp:.{dec_plus}f} \\\\\n".format(
                    dec_plus=dec_plus,
                    dec_minus=dec_minus,
                    dm1=dec_minus - 1,
                    dp1=dec_plus - 1,
                    i=im1 + 1,
                    fm=Minus.f[im1, wm1],
                    dfm=Minus.df[im1, wm1],
                    fp=Plus.f[im1, wm1],
                    dfp=Plus.df[im1, wm1],
                )
            )
        f.write(table_footer)
        for wm1 in range(1, 96):
            f.write(
                """
                \\begin{figure}
                    \\centering
                    \\includegraphics{si-figs/FigS%i.png}
                    \\caption{
                        As Figure~\\ref{fig:S1} with well $w=%i$ (or %s).
                    }
                \\end{figure}
                \\clearpage""" % (wm1 + 1, wm1 + 1, number_to_well(wm1))
            )

            f.write(get_table_header(wm1))
            for im1 in range(Plus.n):
                dec_plus = 1 + int(-1 * np.floor(np.log10(Plus.df[im1, wm1])))
                dec_minus = 1 + int(-1 * np.floor(np.log10(Minus.df[im1, wm1])))
                f.write(
                    "{i:d} & {fm:.{dm1}f} & {dfm:.{dec_minus}f} & {fp:.{dp1}f} & {dfp:.{dec_plus}f} \\\\\n".format(
                        dec_plus=dec_plus,
                        dec_minus=dec_minus,
                        dm1=dec_minus - 1,
                        dp1=dec_plus - 1,
                        i=im1 + 1,
                        fm=Minus.f[im1, wm1],
                        dfm=Minus.df[im1, wm1],
                        fp=Plus.f[im1, wm1],
                        dfp=Plus.df[im1, wm1],
                    )
                )
            f.write(table_footer)

    for wm1 in range(96):
        fig, axs = plt.subplots(figsize=(6., 6.), ncols=3, nrows=3)
        axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2], axs[2, 0], axs[2, 1], axs[2, 2]]
        for i, ax in zip(
                list(i - 1 for i in cycles_to_plot),
                axes
        ):
            kwargs = dict(color='C1', clip_on=False)
            ax.set_ylabel("$F_{%i,%i}^q$ or $f_{%i,%i}^{\\pm}C$" % (i + 1, wm1 + 1, i + 1, wm1 + 1))
            ax.plot(Plus.C, Plus.F[i, wm1], 'o', mfc='None', **kwargs)
            C = np.linspace(0, Plus.C.max())
            ax.fill_between(C, (Plus.f[i, wm1] - 3 * Plus.df[i, wm1]) * C,
                            (Plus.f[i, wm1] + 3 * Plus.df[i, wm1]) * C,
                            alpha=0.3, **kwargs)
            ax.plot(C, Plus.f[i, wm1] * C, '-', label="%s" % number_to_well(wm1), **kwargs)

            kwargs['color'] = 'C0'
            ax.plot(Minus.C, Minus.F[i, wm1], 'x', mfc='None', **kwargs)
            C = np.linspace(0, Minus.C.max())
            ax.fill_between(C, (Minus.f[i, wm1] - 3 * Minus.df[i, wm1]) * C,
                            (Minus.f[i, wm1] + 3 * Minus.df[i, wm1]) * C,
                            alpha=0.3, **kwargs)
            ax.plot(C, Minus.f[i, wm1] * C, '--', **kwargs)

        for i in range(9):
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].set_ylim([0., 1.6])
            axes[i].set_yticks([0., 0.4, 0.8, 1.2, 1.6])
            axes[i].set_xticks([0., 0.08, 0.16])
            axes[i].set_xlim([0., 0.16])
            axes[i].tick_params(which='both', direction='in')
            axes[i].grid()
            # if i in (1, 2, 4, 5, 7, 8):
            #     plt.setp(axes[i].get_yticklabels(), visible=False)
            axes[i].set_xlabel("$C^q$ or $C$ [pmol/L]")
            # if i > 5:
            # else:
            #     plt.setp(axes[i].get_xticklabels(), visible=False)

        axes[0].annotate("Inactive", xy=(0.53, 0.15), xycoords='axes fraction', color='C0')
        axes[0].annotate("Active", xy=(0.1, 0.85), xycoords='axes fraction', color='C1')
        axes[1].set_xticklabels(["", "0.08", "0.16"])
        axes[2].set_xticklabels(["", "0.08", "0.16"])
        fig.subplots_adjust(left=0.09, right=0.97, top=0.99, bottom=0.07, wspace=0.4, hspace=0.30)
        fig.savefig(os.path.join(BASE_DIR, 'out', "FigS%i.png" % (wm1 + 1)), transparent=True, dpi=300)
        plt.close(fig)

    # print max cvs
    print("Active probe:")
    Plus.print_cv()
    print("Inactive Probe:")
    Minus.print_cv()
