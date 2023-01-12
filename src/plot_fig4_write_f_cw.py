from get_data import file_to_numpy, conc_dir_to_conc
import matplotlib.pyplot as plt
from globals import CYCLE_SYMBOL
from wells import number_to_well
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_dataset(reporter_dir, TE_dir):
    data_dir = os.path.join(BASE_DIR, "data", reporter_dir, TE_dir)
    data = {}

    for conc_dir in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, conc_dir)):
            if file.endswith('.xls'):
                assert conc_dir not in data.keys(), "Already have conc!"
                data[conc_dir_to_conc(conc_dir)] = file_to_numpy(
                    os.path.join(data_dir, conc_dir, file)
                )
    return data


def find_index(val, group):
    i = 0
    while i < len(group):
        if val == group[i]:
            return i
        i += 1

    return i


def main(data):
    fig, ax = plt.subplots(figsize=(4.68504, 2.5), ncols=3, sharex=False, sharey=True)
    cycles_to_plot = [1, 20, 40]
    wells_to_plot = (
        'A1', 'C10', 'E5', 'G3', 'H11'
    )
    markers = ['^', 'o', 'v', '*', 'd']
    lines = ['-', '--']

    for reporter, values in data.items():

        # ignore high concentration points
        for C in list(values.keys()):
            if C > 0.2:
                values.pop(C)

        # get fit
        f = np.zeros((45, 96))
        df = np.zeros((45, 96))
        q = len(values.keys())

        for icycle in range(45):
            for iwell in range(96):
                Cwell = np.zeros(q)
                Fwell = np.zeros(q)
                i = 0
                for C, val in values.items():
                    Cwell[i] = C
                    Fwell[i] = val[icycle, iwell]
                    i += 1

                f[icycle, iwell] = np.inner(Fwell, Cwell) / np.inner(Cwell, Cwell)
                R = Fwell - f[icycle, iwell] * Cwell
                df[icycle, iwell] = np.sqrt(np.inner(R, R) / (q - 1))

                well_index = find_index(number_to_well(iwell), wells_to_plot)
                cycle_index = find_index(icycle, cycles_to_plot)
                if cycle_index < len(cycles_to_plot) and well_index < len(wells_to_plot):
                    ireporter = find_index(reporter, ['FAM', 'Probe'])
                    kwargs = dict(color='C%i' % well_index, clip_on=False)
                    ax[cycle_index].plot(Cwell, Fwell, markers[well_index], mfc='None', **kwargs)
                    C = np.linspace(0, Cwell.max())
                    ax[cycle_index].fill_between(C,
                                                 (f[icycle, iwell] - 3 * df[icycle, iwell]) * C,
                                                 (f[icycle, iwell] + 3 * df[icycle, iwell]) * C,
                                                 alpha=0.3, **kwargs)
                    if reporter == "FAM":
                        kwargs['label'] = "%s" % number_to_well(iwell)
                    ax[cycle_index].plot(C, f[icycle, iwell] * C, lines[ireporter], **kwargs)
                    if 'label' in kwargs.keys():
                        kwargs.pop('label')

        with open(os.path.join(BASE_DIR, "..", "out", "f_cw_%s.npy" % reporter), "wb") as file:
            np.save(file, f)

        with open(os.path.join(BASE_DIR, "..", "out", "df_cw_%s.npy" % reporter), "wb") as file:
            np.save(file, df)

    ax[0].legend(loc=(0.01, 0.97), edgecolor="None", facecolor='None', ncol=len(wells_to_plot),
                 handleheight=1.5, labelspacing=0.3, columnspacing=1.5,  # handlelength=1.
                 )

    ax[0].set_ylabel("$\\frac{F_{%s,w}}{10^6}$" % CYCLE_SYMBOL,
                     rotation=0, fontsize=16, labelpad=16)
    for i in range(3):
        ax[i].set_xlabel("$C$ [pmol/L]")
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim([0., 1.6])
        ax[i].set_yticks([0., 0.4, 0.8, 1.2, 1.6])
        ax[i].set_xticks([0., 0.08, 0.16])
        ax[i].set_xlim([0., 0.16])
        ax[i].tick_params(which='both', direction='in')
        ax[i].annotate("$%s=%i$" % (CYCLE_SYMBOL, cycles_to_plot[i]), xy=(0.6, 0.04), xycoords='axes fraction')
    ax[0].annotate("Inactive", xy=(0.73, 0.55), xycoords='axes fraction',
                   xytext=(0.55, 0.25), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle="->"))
    ax[0].annotate("Active", xy=(0.3, 0.85), xycoords='axes fraction',
                   xytext=(0.05, 0.95), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle="-[", connectionstyle="angle"))
    ax[1].set_xticklabels(["", "0.08", "0.16"])
    ax[2].set_xticklabels(["", "0.08", "0.16"])
    fig.subplots_adjust(left=0.14, right=0.96, top=0.94, bottom=0.16, wspace=0.04)
    fig.savefig(os.path.join(BASE_DIR, '..', 'out', "Fig4.png"), transparent=True, dpi=300)


if __name__ == '__main__':
    data = {
        key: get_dataset(key, "25_vol_pt_TE") for key in ("FAM", "Probe")
    }

    main(data)
