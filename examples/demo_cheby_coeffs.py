# -*- coding: utf-8 -*-
"""
Example: Chebyshev Approximation Quality
---------------------------------------
This demo illustrates how the order of the Chebyshev polynomial affects the 
accuracy of the filter approximation across the graph spectrum.
"""

import os
import sgwt
import numpy as np
import matplotlib.pyplot as plt
from sgwt import IMPEDANCE_TEXAS as L

# Professional Plotting Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
})

# DOC_START_CODE_EXCLUDE_IMPORTS
# 1. Define target filter function
def bandpass(x):
    return x / (1 + x)**2

def f(x):
    return np.array([bandpass(x)]).T

# 2. Get spectrum bound from the module
conv = sgwt.ChebConvolve(L)
ubnd = conv.spectrum_bound

# 3. Setup evaluation points
x_eval = np.geomspace(1e-4, ubnd, 1000)
y_true = f(x_eval)

# 4. Compare different orders
orders = [10, 50, 200]
# DOC_END_CODE_EXCLUDE_PLOT

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Chebyshev Approximation Quality", fontsize=14)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.plot(x_eval, y_true, 'k--', label='Target Filter', alpha=0.6)

for order in orders:
    # Fit kernel using the module's function (uses quadratic sampling by default)
    kernel = sgwt.ChebyKernel.from_function(f, order, ubnd)
    y_approx = kernel.evaluate(x_eval)
    
    ax1.plot(x_eval, y_approx, label=f'Order {order}')
    
    # Plot absolute error
    error = np.abs(y_true.flatten() - y_approx.flatten())
    ax2.plot(x_eval, error, label=f'Order {order} Error')

xlim = np.min(x_eval), np.max(x_eval)
ax1.set_xlim(*xlim)
ax2.set_xlim(*xlim)

ax1.set_title('Chebyshev Approximation Quality (Quadratic Sampling)')
ax1.set_ylabel('Filter Gain')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both', linestyle='--')

ax2.set_title('Approximation Error (Absolute)')
ax2.set_ylabel('Error')
ax2.set_xlabel('Eigenvalue (λ)')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_cheby_coeffs.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()