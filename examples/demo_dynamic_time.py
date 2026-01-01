import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve, DyConvolve, impulse
from sgwt import DELAY_USA as L
import numpy as np
import time 

# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Polesp
scales = np.geomspace(1e-5, 1e2, 20)
poles = 1/scales

with Convolve(L) as conv:
    start = time.time()
    for i in range(10):
        Y = conv.bandpass(X, scales)
    T1 = time.time() - start


with DyConvolve(L, poles) as conv:
    start = time.time()
    for i in range(10):
        Y = conv.bandpass(X)
    T2 = time.time() - start
# DOC_END_CODE_EXCLUDE_PLOT
# Set font to Times New Roman for a professional look (already set above, but kept for clarity if code block was different)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Create a bar chart to visualize the performance comparison
fig, ax = plt.subplots(figsize=(9, 6)) # Increased figure size for better readability
labels = ['Convolve', 'DyConvolve']
times_ms = [T1 * 1000, T2 * 1000]
colors = ['#66c2a5', '#fc8d62'] # A more subtle color palette

bars = ax.bar(labels, times_ms, color=colors, width=0.6) # Added width for better spacing
ax.set_ylabel('Execution Time (ms)', fontsize=12)
ax.set_title('80k Bus Convolution with 20 Scales For 10 Signals', fontsize=14, pad=15) # Added padding to title
ax.tick_params(axis='x', labelsize=11) # Increased rotation and labelsize
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.6) # Slightly reduced alpha for grid

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + yval*0.03, # Adjusted y-offset
            f"{yval:.2f} ms", ha='center', va='bottom', fontsize=10, color='black')

# Add a frame around the plot area
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

plt.tight_layout(pad=2.0) # Increased padding for tight_layout

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
