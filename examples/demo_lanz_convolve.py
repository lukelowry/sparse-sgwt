# -*- coding: utf-8 -*-
"""
Example: Lanczos Method Convolution
-----------------------------------
Compares the Lanczos method approximation against the analytical DyConvolve solver.
"""

import os, time, numpy as np, matplotlib.pyplot as plt
import sgwt
from sgwt import LanzConvolve, DyConvolve, ChebConvolve, impulse
from sgwt import IMPEDANCE_TEXAS as L, COORD_TEXAS as C
from demo_plot import plot_signal

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})

SCALES, ORDER, N_ITER, XMIN = [0.05, 0.5, 5.0], 200, 400, 1e-2
X = impulse(L, n=600) # Input signal
def f(x): return np.stack([((4.0/s)*x / (x + 1.0/s)**2)**2 for s in SCALES], axis=1)

# --- Calculations & Benchmarking ---
def run_bench(ctx, func):
    with ctx:
        res = func() # Warm up
        start = time.time()
        for _ in range(N_ITER): _ = func()
        return res, (time.time() - start) / N_ITER

conv_cheb = ChebConvolve(L)
kernel = sgwt.ChebyKernel.from_function(f, ORDER, conv_cheb.spectrum_bound, min_lambda=XMIN)

methods = [
    {"name": "Lanczos", "color": "#e78ac3", "ctx": LanzConvolve(L), "run": lambda c: c.convolve(X, f, ORDER)},
    {"name": "Chebyshev", "color": "#fc8d62", "ctx": conv_cheb, "run": lambda c: c.convolve(X, kernel)},
    {"name": "DyConvolve", "color": "#66c2a5", "ctx": DyConvolve(L, [1.0/s for s in SCALES]), "run": lambda c: c.bandpass(X, order=2)}
]

for m in methods:
    m['res'], m['time'] = run_bench(m['ctx'], lambda: m['run'](m['ctx']))

# --- Plotting ---
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.3, hspace=0.4)
fig.suptitle(f"Graph Convolution Comparison (Order {ORDER})", fontsize=16, fontweight='bold')

# 1. Spectral Response (Ritz Values)
ax_spec = fig.add_subplot(gs[0:2, 0])
ubnd = conv_cheb.spectrum_bound
ritz_vals = methods[0]['ctx'].ritz_values(X, ORDER)
x_eval = np.geomspace(XMIN, ubnd, 1000)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(SCALES)))

for i, s in enumerate(SCALES):
    ax_spec.plot(x_eval, f(x_eval)[:, i], 'k--', alpha=0.2)
    ax_spec.scatter(ritz_vals, f(ritz_vals)[:, i], color=colors[i], s=8, alpha=0.6, label=f'Scale {s}')
ax_spec.set_xscale('log'); ax_spec.set_title("Spectral Response (Ritz Values)", fontweight='bold')
ax_spec.legend(fontsize=8); ax_spec.set_xlabel('λ'); ax_spec.set_ylabel('Gain')

# 2. Timing Comparison
ax_time = fig.add_subplot(gs[2, 0])
names = [m['name'] for m in methods]
times = [m['time']*1000 for m in methods]
colors_bar = [m['color'] for m in methods]
bars = ax_time.bar(names, times, color=colors_bar, width=0.6)
ax_time.set_ylabel('ms'); ax_time.set_title('Runtime', fontweight='bold')
for b in bars: ax_time.text(b.get_x()+b.get_width()/2, b.get_height(), f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=9)

# 3. Spatial Wavelets
for i, s in enumerate(SCALES):
    for j, m in enumerate(methods):
        ax = fig.add_subplot(gs[i, j+1])
        data = m['res']
        sig = data[i][:, 0] if isinstance(data, list) else data[:, 0, i]
        plot_signal(sig, C, 'berlin', ax=ax)
        if i == 0: ax.set_title(m['name'], fontweight='bold')
        if j == len(methods)-1: ax.text(1.05, 0.5, f"Scale {s}", transform=ax.transAxes, rotation=-90, va='center', fontweight='bold')

plt.tight_layout(pad=0.1, rect=[0, 0, 1, 0.93])
img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', '_static', 'images', 'demo_lanz_convolve.png'))
os.makedirs(os.path.dirname(img_path), exist_ok=True)
plt.savefig(img_path, dpi=400, bbox_inches='tight')
plt.show()