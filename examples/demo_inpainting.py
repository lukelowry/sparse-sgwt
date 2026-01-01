import os
import numpy as np
import matplotlib.pyplot as plt
from sgwt import DyConvolve, DELAY_USA, COORD_USA

# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# --- Configuration ---
# Note: Parameters tuned for fast convergence.
# Original: N_ITERATIONS=500, SMOOTHING_SCALE=10.0, STEP_SIZE=1.0
SAMPLE_FRACTION = 0.005  # Use 0.1% of nodes as sensors
N_ITERATIONS = 100      # Number of reconstruction steps
SMOOTHING_SCALE = 50.0  # Larger scale propagates info faster
STEP_SIZE = 1         # Larger step size accelerates convergence

# 1. Setup: Ground Truth Signal and Graph
L = DELAY_USA
C = COORD_USA
n_nodes = L.shape[0]

# Use longitude as a smooth signal over the graph
X_true = C[:, 0:1].copy(order='F')

# 2. Create Sparse Samples
n_samples = int(n_nodes * SAMPLE_FRACTION)
sample_indices = np.random.choice(n_nodes, n_samples, replace=False)

# A boolean mask efficiently identifies sensor locations
J_mask = np.zeros(n_nodes, dtype=bool)
J_mask[sample_indices] = True

# The sampled signal is zero everywhere except at sensor locations
X_sampled = np.zeros_like(X_true)
X_sampled[J_mask] = X_true[J_mask]

# 3. Iterative Reconstruction
Xh = np.zeros_like(X_true, order='F')  # Start with a zero-signal guess

# DyConvolve is ideal here, as it pre-factors the system for a fixed scale.
with DyConvolve(L, poles=[1/SMOOTHING_SCALE]) as conv:
    print(f"Reconstructing signal from {n_samples} samples ({SAMPLE_FRACTION:.1%})...")

    for i in range(N_ITERATIONS):
        # Calculate error at sensor locations
        error = np.zeros_like(Xh)
        error[J_mask] = X_sampled[J_mask] - Xh[J_mask]

        # Propagate error across the graph using a low-pass filter
        smoothed_error = conv.lowpass(error)[0]

        # Update the signal. The DyConvolve.lowpass filter has a 1/scale factor,
        # so we multiply by the scale to get the pure solver response.
        Xh += STEP_SIZE * smoothed_error * SMOOTHING_SCALE

        if (i + 1) % 100 == 0:
            print(f"  Iteration {i+1}/{N_ITERATIONS}")

print("Reconstruction complete.")

# DOC_END_CODE_EXCLUDE_PLOT
# 4. Visualize Results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
fig.suptitle(f'Graph Signal Inpainting from {SAMPLE_FRACTION:.1%} of Data', fontsize=14, fontweight='bold')

# Use ground truth for consistent color mapping
vmin, vmax = np.min(X_true), np.max(X_true)

# --- Plot 1: Ground Truth ---
ax1.set_title('Ground Truth Signal', fontsize=12)
ax1.scatter(C[:, 0], C[:, 1], c=X_true, s=10, vmin=vmin, vmax=vmax, cmap='viridis')

# --- Plot 2: Sparse Input ---
ax2.set_title('Input: Sparse Samples', fontsize=12)
# Plot all nodes as a faint background
ax2.scatter(C[:, 0], C[:, 1], c='#e0e0e0', s=8, zorder=1)
# Highlight sampled nodes
ax2.scatter(C[J_mask, 0], C[J_mask, 1], c=X_true[J_mask],
            s=35, vmin=vmin, vmax=vmax, cmap='viridis', zorder=2, edgecolors='black', linewidths=0.75)

# --- Plot 3: Reconstructed Output ---
ax3.set_title('Output: Reconstructed Signal', fontsize=12)
# Use same color map for comparison
ax3.scatter(C[:, 0], C[:, 1], c=Xh, s=10, vmin=vmin, vmax=vmax, cmap='viridis')

# --- Plot Formatting ---
for ax in [ax1, ax2, ax3]:
    ax.set_facecolor('white')
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure for documentation
# Determine the project root and target static images directory
script_dir = os.path.dirname(os.path.abspath(__file__)) # e.g., .../sparse-sgwt/examples
project_root = os.path.abspath(os.path.join(script_dir, '..')) # e.g., .../sparse-sgwt
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'inpainting_reconstruction.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')

plt.show()