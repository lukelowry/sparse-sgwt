import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS

from sgwt import Convolve, DyConvolve, impulse
from sgwt import DELAY_USA as L
import numpy as np
import time 

# DOC_START_CODE_EXCLUDE_IMPORTS
# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Polesp
scales = np.geomspace(1e-5, 1e2, 20)
poles = 1/scales

with Convolve(L) as conv:

    start = time.time()
    for i in range(2):
        Y = conv.bandpass(X, scales)
    T1 = time.time() - start


with DyConvolve(L, poles) as conv:

    start = time.time()
    for i in range(2):
        Y = conv.bandpass(X)
    T2 = time.time() - start

# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# DOC_END_CODE_EXCLUDE_PLOT
# Create a bar chart to visualize the performance comparison
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Static (Convolve)', 'Dynamic (DyConvolve)']
times_ms = [T1 * 1000, T2 * 1000]
colors = ['skyblue', 'lightcoral']

ax.bar(labels, times_ms, color=colors)
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Performance Comparison: Static vs. Dynamic Convolution')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of bars
for i, v in enumerate(times_ms):
    ax.text(i, v + max(times_ms)*0.05, f"{v:.2f} ms", ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_dynamic_time.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"Static: {T1*1000:.3f} ms")
print(f"Dynamic: {T2*1000:.3f} ms")
