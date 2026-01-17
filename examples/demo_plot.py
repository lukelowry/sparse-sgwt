import os
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as PathEffects
from scipy.stats import gaussian_kde

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

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

def save_figure(fig, folder_path, filename, dpi=300):
    """
    Saves the figure to a specified folder with customizable DPI.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    full_path = os.path.join(folder_path, filename)
    
    # Save with tight bounding box and matching facecolor
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {full_path} (DPI={dpi})")

def overlay_peaks(ax, peaks):
    """Plots numbered markers with high-contrast outlines."""
    ax.scatter(peaks['Wavelength'], peaks['Frequency'], marker='x', s=120, lw=4, c='black', zorder=15)
    ax.scatter(peaks['Wavelength'], peaks['Frequency'], marker='x', s=120, lw=2, c='white', zorder=16)
    
    n_peaks = peaks['Wavelength'].size
    for i in range(n_peaks):
        w, f = peaks['Wavelength'][i], peaks['Frequency'][i]
        txt = ax.annotate(f"{i+1}", (w, f),
                          xytext=(8, 8), textcoords='offset points', 
                          color='white', fontsize=13, weight='bold', zorder=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

def plot_contour(ax, x, y, Y_mag, cmap='magma', levels=20):
    """Renders dark-mode contour plot with continuous colorbar."""
    ax.set_facecolor('black')
    text_c = 'white'
    
    X, Y = np.meshgrid(x, y)
    CS = ax.contour(X, Y, Y_mag.T, levels=levels, cmap=cmap, linewidths=1.5)
    
    ax.set_xscale("log") 
    ax.set_yscale("linear") 
    ax.grid(False) 
    
    ax.set_xlabel("Wavelength [km]", color=text_c)
    ax.set_ylabel("Frequency [Hz]", color=text_c)
    ax.tick_params(colors=text_c, which='both')
    for spine in ax.spines.values(): spine.set_color(text_c)

    norm = plt.Normalize(vmin=Y_mag.min(), vmax=Y_mag.max())
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Magnitude', color=text_c, rotation=270, labelpad=15)
    cbar.ax.yaxis.set_tick_params(color=text_c, labelcolor=text_c)
    cbar.outline.set_visible(False)



def plot_peak_heatmap(master_df, wavlen, freqs, cmap='inferno', output_dir=None, filename="mode_clusters.png", dpi=300):
    """
    Plots a Kernel Density Estimate (KDE) heatmap of system-wide mode clusters.
    
    Parameters:
    output_dir : str (Optional)
        If provided, the plot will be saved to this folder.
    filename : str (Optional)
        The filename for the saved image.
    dpi : int (Optional)
        Resolution for saving (default 300).
    """
    # --- 1. Data Preparation ---
    x_val = np.log10(master_df['Wavelength'])
    y_val = master_df['Frequency']
    
    values = np.vstack([x_val, y_val])
    kernel = gaussian_kde(values)
    
    # --- 2. Evaluation Grid ---
    xmin, xmax = np.log10(wavlen.min()), np.log10(wavlen.max())
    ymin, ymax = freqs.min(), freqs.max()
    
    X_grid, Y_grid = np.meshgrid(
        np.linspace(xmin, xmax, 100),
        np.linspace(ymin, ymax, 100)
    )
    
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    Z = np.reshape(kernel(positions).T, X_grid.shape)

    # --- 3. Setup Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#2b2b2b')
    ax.set_facecolor('black')
    text_c = 'white'

    # --- 4. Plot Heatmap (Smooth Density) ---
    cf = ax.contourf(10**X_grid, Y_grid, Z, levels=50, cmap=cmap, extend='min')

    # --- 5. Plot Individual Peaks (Refined) ---
    ax.scatter(
        master_df['Wavelength'], 
        master_df['Frequency'], 
        color='white', 
        s=20,           
        alpha=0.5,      
        linewidths=0.3, 
        edgecolors='black', 
        label='Detected Peak'
    )

    # --- 6. Formatting ---
    ax.set_xscale("log")
    ax.set_yscale("linear") 
    ax.grid(False)

    ax.set_xlabel("Wavelength [km]", color=text_c)
    ax.set_ylabel("Frequency [Hz]", color=text_c)
    ax.set_title("System-Wide Mode Clusters (Density)", color=text_c, pad=15)
    
    ax.tick_params(colors=text_c, which='both')
    for spine in ax.spines.values(): spine.set_color(text_c)

    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mode Density', color=text_c, rotation=270, labelpad=20)
    cbar.ax.yaxis.set_tick_params(color=text_c, labelcolor=text_c)
    cbar.outline.set_visible(False)

    # --- 7. Optional Saving ---
    if output_dir:
        save_figure(fig, output_dir, filename, dpi=dpi)

    plt.show()