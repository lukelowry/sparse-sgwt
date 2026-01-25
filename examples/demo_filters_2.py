import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_EASTWEST as L
from sgwt import COORD_EASTWEST as C

X = impulse(L, n=65000)
X -= impulse(L, n=35000)
s = 3e0

with Convolve(L) as conv:
    BP = conv.bandpass(X, s, order=3)
# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(BP, C, 'coolwarm')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_filters_2_bandpass.png'), dpi=400, bbox_inches='tight')
plt.show()
