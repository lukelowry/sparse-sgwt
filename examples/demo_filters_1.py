import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C

X = impulse(L, n=600)
s = [1e-1]

with Convolve(L) as conv:
    LP = conv.lowpass(X, s)
    BP = conv.bandpass(X, s)
    HP = conv.highpass(X, s)
# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(BP[0][:,0], C, 'berlin')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_filters_1_bandpass.png'), dpi=400, bbox_inches='tight')
plt.show()
