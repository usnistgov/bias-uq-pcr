import matplotlib.pyplot as plt
import numpy as np
import os
from globals import CYCLE_SYMBOL
from amplification import Amplification

cycles = np.array(list(range(5)))
R = 0.9
pbar = 0.85

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
fig.savefig(os.path.join(BASE_DIR, "..", "out", "Fig2.png"), transparent=True, dpi=300)
