import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_USA as L
from sgwt import COORD_USA as C

X = 2.5 * impulse(L, n=22000)
X += impulse(L, n=72000)
s = [.5, 10, 100]

with Convolve(L) as conv:
    Y = conv.lowpass(X, s, order=8)[0]
# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(Y[:,0], C, 'managua')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_filters_5.png'), dpi=500, bbox_inches='tight')
plt.show()
