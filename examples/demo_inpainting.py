import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import DyConvolve
from sgwt import DELAY_USA as L
from sgwt import COORD_USA as C
import numpy as np

SAMPLE_FRACTION = 0.005
N_ITERATIONS = 100
SMOOTHING_SCALE = 50.0
STEP_SIZE = 1
n_nodes = L.shape[0]

X_true = C[:, 0:1].copy(order='F')

n_samples = int(n_nodes * SAMPLE_FRACTION)
sample_indices = np.random.choice(n_nodes, n_samples, replace=False)
J_mask = np.isin(np.arange(n_nodes), sample_indices)
X_sampled = np.zeros_like(X_true)
X_sampled[J_mask] = X_true[J_mask]

Xh = np.zeros_like(X_true, order='F')

with DyConvolve(L, poles=[1/SMOOTHING_SCALE]) as conv:
    for i in range(N_ITERATIONS):
        error = np.zeros_like(Xh)
        error[J_mask] = X_sampled[J_mask] - Xh[J_mask]
        smoothed_error = conv.lowpass(error)[0]
        Xh += STEP_SIZE * smoothed_error * SMOOTHING_SCALE
# DOC_END_CODE_EXCLUDE_PLOT

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
fig.suptitle(f'Graph Signal Inpainting from {SAMPLE_FRACTION:.1%} of Data', fontsize=14, fontweight='bold')

vmin, vmax = np.min(X_true), np.max(X_true)

ax1.set_title('Ground Truth Signal', fontsize=12)
ax1.scatter(C[:, 0], C[:, 1], c=X_true, s=10, vmin=vmin, vmax=vmax, cmap='viridis')

ax2.set_title('Input: Sparse Samples', fontsize=12)
ax2.scatter(C[:, 0], C[:, 1], c='#e0e0e0', s=8, zorder=1)
ax2.scatter(C[J_mask, 0], C[J_mask, 1], c=X_true[J_mask],
            s=35, vmin=vmin, vmax=vmax, cmap='viridis', zorder=2, edgecolors='black', linewidths=0.75)

ax3.set_title('Output: Reconstructed Signal', fontsize=12)
ax3.scatter(C[:, 0], C[:, 1], c=Xh, s=10, vmin=vmin, vmax=vmax, cmap='viridis')

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('white')
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'inpainting_reconstruction.png'), dpi=400, bbox_inches='tight')
plt.show()
