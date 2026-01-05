# -*- coding: utf-8 -*-
"""
Example: Chebyshev Polynomial Convolution
-----------------------------------------
Compares multi-scale Chebyshev polynomial approximations against the 
analytical DyConvolve solver.
"""

import os, time, numpy as np, matplotlib.pyplot as plt
# DOC_START_CODE_EXCLUDE_IMPORTS
import sgwt
from sgwt import ChebyConvolve, DyConvolve, impulse
from sgwt import DELAY_EASTWEST as L, COORD_EASTWEST as C
from demo_plot import plot_signal, plot_spectral

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})

SCALES, ORDER, N_ITER = [3], 500, 2
XMIN = 1e-5
X = impulse(L, n=65000) - impulse(L, n=35000)

def f(x): return np.stack([sgwt.functions.bandpass(x, scale=s, order=3) for s in SCALES], axis=1)

# --- Calculations ---
kernel = sgwt.ChebyKernel.from_function_on_graph(L, f, ORDER, min_lambda=XMIN)

# 1. Convolve signals for spatial plots
with ChebyConvolve(L) as conv_cheb:
    Y_cheb = conv_cheb.convolve(X, kernel)
with DyConvolve(L, [1.0/s for s in SCALES]) as conv: 
    Y_dy = conv.bandpass(X, order=3)

# 2. Time convolutions for performance plot
with ChebyConvolve(L) as conv_cheb:
    _ = conv_cheb.convolve(X, kernel) # warm-up
    start = time.time()
    for _ in range(N_ITER): _ = conv_cheb.convolve(X, kernel)
    t_cheb = (time.time() - start) / N_ITER

with DyConvolve(L, [1.0/s for s in SCALES]) as conv:
    _ = conv.bandpass(X) # warm-up
    start = time.time()
    for _ in range(N_ITER): _ = conv.bandpass(X, order=3)
    t_dy = (time.time() - start) / N_ITER

# DOC_END_CODE_EXCLUDE_PLOT

# 3. Calculate spectral approximation error
x_eval = np.geomspace(XMIN, kernel.spectrum_bound, 1000)
y_true = f(x_eval)
y_approx = kernel.evaluate(x_eval)
errors = np.mean((y_true - y_approx)**2, axis=0) # MSE for each scale


# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
gs_main = fig.add_gridspec(1, 2, width_ratios=[0.8, 2], wspace=0.05)
fig.suptitle("Chebyshev vs. DyConvolve", fontsize=16, fontweight='bold')
fig.text(0.5, 0.935, f"Polynomial Order $k={ORDER}$", ha='center', fontsize=12, style='italic')

# --- Left Column: Analysis Plots ---
gs_left = gs_main[0, 0].subgridspec(3, 1, hspace=0.5)

# Spectral plot
ax_spec = fig.add_subplot(gs_left[0, 0])
plot_spectral(ax_spec, kernel, f, lbnd=XMIN)
ax_spec.set_title("Spectral Response", fontsize=12, fontweight='bold', pad=10)

# Timing plot
ax_time = fig.add_subplot(gs_left[1, 0])
labels = ['Chebyshev', 'DyConvolve']
times_ms = [t_cheb * 1000, t_dy * 1000]
bars = ax_time.bar(labels, times_ms, color=['#fc8d62', '#66c2a5'], width=0.4)
ax_time.set_ylabel('Runtime [ms]', fontsize=10)
ax_time.set_title('Convolution Runtime', fontsize=12, fontweight='bold', pad=10)
ax_time.grid(axis='y', linestyle='--', alpha=0.6)
ax_time.set_xlim(-0.8, 1.8)
ax_time.set_ylim(0, max(times_ms) * 1.35)
for bar in bars:
    yval = bar.get_height()
    ax_time.annotate(f"{yval:.2f}", 
                     xy=(bar.get_x() + bar.get_width()/2, yval),
                     xytext=(0, 12), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color='black',
                     arrowprops=dict(arrowstyle='-', color='black', lw=0.7))
ax_time.spines[['top', 'right']].set_visible(False)

# Error plot
ax_err = fig.add_subplot(gs_left[2, 0])
scale_labels = [f'{s}' for s in SCALES]
bars_err = ax_err.bar(scale_labels, errors, color=plt.cm.viridis(np.linspace(0, 0.8, len(SCALES))), width=0.4)
ax_err.set_yscale('log')
ax_err.set_ylabel('Mean Squared Error', fontsize=10)
ax_err.set_xlabel('Filter Scale', fontsize=10)
ax_err.set_title('Chebyshev Error', fontsize=12, fontweight='bold', pad=10)
ax_err.grid(axis='y', which='both', linestyle='--', alpha=0.6)
ax_err.set_xlim(-0.8, len(SCALES)-0.2)
ax_err.set_ylim(top=max(errors) * 50)
ax_err.spines[['top', 'right']].set_visible(False)

# --- Right Column: Spatial Wavelets ---
gs_right = gs_main[0, 1].subgridspec(len(SCALES), 2, hspace=0.0, wspace=0.0)

for i, s in enumerate(SCALES):
    # Chebyshev
    ax_c = fig.add_subplot(gs_right[i, 0])
    plot_signal(Y_cheb[:, 0, i], C, 'coolwarm', ax=ax_c)
    if i == 0: ax_c.set_title("Chebyshev", fontsize=12, fontweight='bold', pad=10)

    # DyConvolve
    ax_d = fig.add_subplot(gs_right[i, 1])
    plot_signal(Y_dy[i][:, 0], C, 'coolwarm', ax=ax_d)
    if i == 0: ax_d.set_title("DyConvolve", fontsize=12, fontweight='bold', pad=10)
    ax_d.text(1.02, 0.5, f"Scale {s}", transform=ax_d.transAxes, rotation=-90, va='center', ha='left', fontweight='bold')

plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.93])
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_cheby_convolve_2.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
