# -*- coding: utf-8 -*-
"""
Example: Chebyshev vs. Analytical Timing
---------------------------------------
This demo compares the execution time of the Chebyshev polynomial approximation
against the direct analytical solver. Timing is performed inside the context
to exclude symbolic and numeric factorization overhead.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sgwt
from sgwt import ChebyConvolve, DyConvolve, Convolve, impulse
from sgwt import IMPEDANCE_TEXAS as L

# Professional Plotting Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})

# DOC_START_CODE_EXCLUDE_IMPORTS
# --- Configuration ---
SCALES = [0.01, 0.1, 1, 10]
ORDER = 450      # Polynomial order
N_ITER = 200      # Number of iterations for averaging
X = impulse(L, n=1200)

# 1. Setup Filter Kernel
def f(x): return np.stack([sgwt.functions.bandpass(x, scale=s, order=1) for s in SCALES], axis=1)

# Pre-calculate spectral bound and fit kernel (Setup phase)
lbnd = 1e-3
kernel = sgwt.ChebyKernel.from_function_on_graph(L, f, ORDER, min_lambda=lbnd)

print(f"Benchmarking on {L.shape[0]} nodes...")

# 2. Time Chebyshev Convolution
with ChebyConvolve(L) as conv_cheb:
    # Warm up to ensure JIT/cache effects are minimized
    _ = conv_cheb.convolve(X, kernel)
    
    start = time.time()
    for _ in range(N_ITER):
        _ = conv_cheb.convolve(X, kernel)
    t_cheb = (time.time() - start) / N_ITER

# 3. Time Analytical Convolution (Direct Solver)
# We use DyConvolve because it performs numeric factorization in __enter__,
# allowing us to time ONLY the back-substitution (solve) phase.
with DyConvolve(L, poles=[1.0/s for s in SCALES]) as conv_dy:
    # Warm up
    _ = conv_dy.bandpass(X)
    
    start = time.time()
    for _ in range(N_ITER):
        _ = conv_dy.bandpass(X)
    t_analytical = (time.time() - start) / N_ITER

# 4. Time Standard Convolution (Static)
# This includes numeric factorization in every call, representing the 
# "standard" overhead when scales are not pre-determined.
with Convolve(L) as conv_std:
    # Warm up
    _ = conv_std.bandpass(X, SCALES)
    
    start = time.time()
    for _ in range(N_ITER):
        _ = conv_std.bandpass(X, SCALES)
    t_std = (time.time() - start) / N_ITER

# --- Visualization ---
print(f"Chebyshev (Order {ORDER}): {t_cheb*1000:.2f} ms")
print(f"Convolve:  {t_std*1000:.2f} ms")
print(f"DyConvolve:  {t_analytical*1000:.2f} ms")
# DOC_END_CODE_EXCLUDE_PLOT

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chebyshev vs. Analytical Timing Comparison", fontsize=16, fontweight='bold')
fig.text(0.5, 0.92, f"Polynomial Order $k={ORDER}$ | Scales: {len(SCALES)}", ha='center', fontsize=12, style='italic')

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Plot 1: Spectral Response
x_plot = np.geomspace(lbnd , kernel.spectrum_bound, 1000)
y_true = f(x_plot)
y_approx = kernel.evaluate(x_plot)

ax1.plot(x_plot, y_true, 'k--', alpha=0.5)
ax1.plot(x_plot, y_approx, alpha=0.8)

ax1.set_xscale('log')
ax1.set_title('Spectral Response Comparison')
ax1.set_xlabel('Eigenvalue (λ)')
ax1.set_ylabel('Filter Gain')

ax1.plot([], [], 'k--', alpha=0.5, label='Target (Analytical)')
ax1.plot([], [], 'gray', alpha=0.8, label=f'Chebyshev (O={ORDER})')
ax1.legend()

ax1.set_xlim(np.min(x_plot), np.max(x_plot))
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot 2: Timing Comparison
labels = [f'Cheb (O={ORDER})', 'Convolve', 'DyConvolve']
times = [t_cheb * 1000, t_std * 1000, t_analytical * 1000]
bars = ax2.bar(labels, times, color=['#fc8d62', '#8da0cb', '#66c2a5'])

ax2.set_ylabel('Runtime [ms]')
ax2.set_title(f'Benchmark Runtime per Convolution')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), 
            f"{yval:.2f} ms", ha='center', va='bottom')

plt.tight_layout(rect=[0, 0, 1, 0.91])

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_cheby_time.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()