import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, impulse
from sgwt import DELAY_USA as L
from sgwt import COORD_USA as C

# Impulse
X  = impulse(L, n=35000)

# Scales
s = [10]

# Memory Efficient Context
with Convolve(L) as conv:

    #  Fourth Order Band-Pass
    Y = conv.bandpass(X, s)[0]
    Y = conv.bandpass(Y, s)[0]
    Y = conv.bandpass(Y, s)[0]
    Y = conv.bandpass(Y, s)[0]

# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal
plot_signal(Y[:,0], C, 'berlin')

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')

# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)

save_path = os.path.join(static_images_dir, 'demo_filters_3_bandpass.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
