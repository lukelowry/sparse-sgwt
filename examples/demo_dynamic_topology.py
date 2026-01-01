import os
import matplotlib.pyplot as plt
import numpy as np
from sgwt import DyConvolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C

# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Poles
scales = [0.1, 1, 10]
poles = [1/s for s in scales]

# The tradeoff for efficient graph updates is that poles cannot change
with DyConvolve(L, poles) as conv:

    # Pre-Close Convolution
    Y_before = conv.bandpass(X)
 
    # Add Branch, effectively making Bus 1200 and 600 Neighbors.
    # We should expect Bus 600 to have positive value similar to 1200.
    tau = 1e-3
    conv.addbranch(1200, 600, 1/tau**2)

    # Post-Close Convolution
    Y_after = conv.bandpass(X)
    
# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal

# Combine plots into a single figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Dynamic Topology Update: Band-pass Filtered Signal', fontsize=14, fontweight='bold')

# Plot Y_before
plt.sca(ax1) # Set current axes
plot_signal(Y_before[0][:,0], C, 'seismic')
ax1.set_title('Before Branch Added (Bus 1200)')

# Plot Y_after
plt.sca(ax2) # Set current axes
plot_signal(Y_after[0][:,0], C, 'seismic')
ax2.set_title('After Branch Added (Bus 1200 <-> 600)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_dynamic_topology.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
