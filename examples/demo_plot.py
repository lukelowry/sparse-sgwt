import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Set font to Times New Roman for a professional look
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def plot_signal(f, C, cmap='Spectral', ax=None, dot_size=15):
    '''
    Parameters
        f: Signal to plot, (nVertex, nTime)
        C: Coordinats
    '''

    L1, L2 = C[:, 0], C[:, 1]

    mx = np.sort(np.abs(f))[-20] 
    norm = Normalize(-mx, mx)
    if ax is None: ax = plt.gca()
    ax.scatter(L1, L2 , c=f, edgecolors='none', cmap=cmap, norm=norm, s=dot_size)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.margins(0.05)

def plot_spectral(ax, kernel, target_func, lbnd=1e-5, dim=None):
    """Plots target vs approximated spectral response. Plots all if dim is None."""
    ubnd = kernel.spectrum_bound
    x = np.geomspace(lbnd, ubnd, 1000)
    y_true = target_func(x)
    y_approx = kernel.evaluate(x)

    if dim is None:
        dims = range(y_true.shape[1])
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(dims)))
    else:
        dims, colors = [dim], ['#1f77b4']

    for i, d in enumerate(dims):
        ax.plot(x, y_true[:, d], 'k--', alpha=0.3, label='Target' if i==0 else None)
        ax.plot(x, y_approx[:, d], color=colors[i], alpha=0.8, label=f'Scale Index {d}' if dim is None else 'Cheby')

    ax.set_xscale('log')
    ax.set_xlabel('Eigenvalue (λ)', fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Filter Gain', fontsize=10)
    ax.margins(x=0.01)