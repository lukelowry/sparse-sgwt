import os
import matplotlib.pyplot as plt

import time
# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_EASTWEST as L
from sgwt import COORD_EASTWEST as C

# Impulse
X  = impulse(L, n=65000)
X  -= impulse(L, n=35000)

# Scales
s = [3e0]

# Third Order Band-Pass
with Convolve(L) as conv:
    BP = conv.bandpass(X, s, order=3)[0]

# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(BP[:,0], C, 'coolwarm')

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_dir = os.path.join(project_root, 'docs', '_static')

# Save image
images_dir = os.path.join(static_dir, 'images')
os.makedirs(images_dir, exist_ok=True)
save_path = os.path.join(images_dir, 'demo_filters_2_bandpass.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')

plt.show()
