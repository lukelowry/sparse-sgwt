import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS

from sgwt import Convolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C


# Impulse
X  = impulse(L, n=1200)
X += impulse(L, n=600)

# Scales
s = [1e-1]

# Memory Efficient Context
with Convolve(L) as conv:

    LP = conv.lowpass(X, s)
    BP = conv.bandpass(X, s)
    HP = conv.highpass(X, s)

# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# DOC_END_CODE_EXCLUDE_PLOT
# Assuming plot_signal creates and sets the current matplotlib figure
from demo_plot import plot_signal
plot_signal(BP[0][:,0], C, 'seismic')

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__)) # e.g., .../sparse-sgwt/examples
project_root = os.path.abspath(os.path.join(script_dir, '..')) # e.g., .../sparse-sgwt
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')

# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)

save_path = os.path.join(static_images_dir, 'demo_filters_1_bandpass.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show() # Display the plot when the script is run
