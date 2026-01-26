import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import DyConvolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C

X = impulse(L, n=1200)
scales = [0.1, 1, 10]
poles = [1/s for s in scales]

with DyConvolve(L, poles) as conv:
    Y_before = conv.bandpass(X)
    conv.addbranch(1200, 600, 1/(1e-3)**2)
    Y_after = conv.bandpass(X)
# DOC_END_CODE_EXCLUDE_PLOT
from demo_plot import plot_signal

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Dynamic Topology Update: Band-pass Filtered Signal', fontsize=14, fontweight='bold')

plt.sca(ax1)
plot_signal(Y_before[0], C, 'seismic')
ax1.set_title('Before Branch Added (Bus 1200)')

plt.sca(ax2)
plot_signal(Y_after[0], C, 'seismic')
ax2.set_title('After Branch Added (Bus 1200 <-> 600)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_dynamic_topology.png'), dpi=400, bbox_inches='tight')
plt.show()
