import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# Custom Plotting Library
import demo_plot as splt

# SGWT Imports
from sgwt import Convolve, impulse
from sgwt import DELAY_WECC as L
from sgwt.functions import gaussian_wavelet

# --- Helper Logic Functions ---

def get_signal(fname, t_range):
    """Loads parquet, reconstructs voltage, and removes DC offset."""
    Vdata = pd.read_parquet(fname).to_numpy()
    nbus = Vdata.shape[1] // 2
    
    # Reconstruct complex signal
    V = (Vdata[:, :nbus] * np.exp(1j * Vdata[:, nbus:] / 180 * np.pi)).T
    
    # Detrending
    signal = np.cumsum(np.diff(V, axis=1), axis=1)
    time = np.linspace(t_range[0], t_range[1], signal.shape[1])
    return signal, time

def build_transform_matrices(t, s, Ts, bus_target, time_target):
    """Constructs the spatial (A) and temporal (B) filter matrices."""
    B = np.stack([gaussian_wavelet(t, a=sc, b=time_target, w0=2*np.pi) for sc in Ts]).T

    X_imp = impulse(L, n=bus_target)
    with Convolve(L) as conv:
        A = np.stack(conv.bandpass(X_imp, s, order=10), axis=1)[:,:,0].T
        
    return A, B

def find_peaks(x, y, Y_mag, top_n=5, min_dist=5):
    """Returns top N local maxima coordinates and magnitudes."""
    coords = peak_local_max(Y_mag, min_distance=min_dist)
    mags = Y_mag[coords[:, 0], coords[:, 1]]
    
    df = pd.DataFrame({
        'Wavelength': x[coords[:, 0]],
        'Frequency': y[coords[:, 1]],
        'Magnitude': mags
    })
    return df.sort_values('Magnitude', ascending=False).head(top_n).reset_index(drop=True)

# --- Batch Processing Functions ---

def extract_all_peaks(V, t, s, Ts, wavlen, freqs, time_target):
    """Computes SGWT for all buses and returns a DataFrame of all detected peaks."""
    n_buses = V.shape[0]
    all_peaks_list = []

    print(f"Extracting peaks for {n_buses} buses...")

    # Pre-compute Temporal Matrix B (Constant)
    B = np.stack([gaussian_wavelet(t, a=sc, b=time_target, w0=2*np.pi) for sc in Ts]).T

    for bus_idx in range(n_buses):
        # Spatial Matrix
        X_imp = impulse(L, n=bus_idx)
        with Convolve(L) as conv:
            A = np.stack(conv.bandpass(X_imp, s, order=10), axis=1)[:,:,0].T 

        Y = A @ V @ B
        Ymag = np.sqrt(np.abs(Y)) # Consistent with your main script
        
        peaks_df = find_peaks(wavlen, freqs, Ymag, top_n=5)
        peaks_df['Bus_ID'] = bus_idx
        all_peaks_list.append(peaks_df)
    
    master_df = pd.concat(all_peaks_list, ignore_index=True)
    print(f"Extraction complete. Found {len(master_df)} total peaks.")
    
    return master_df

def generate_all_bus_plots(V, t, s, Ts, wavlen, freqs, time_target, output_dir):
    """Iterates through all buses, generates plots, and saves them."""
    n_buses = V.shape[0]
    print(f"Starting batch plotting for {n_buses} buses...")

    B = np.stack([gaussian_wavelet(t, a=sc, b=time_target, w0=2*np.pi) for sc in Ts]).T

    for bus_idx in range(n_buses):
        X_imp = impulse(L, n=bus_idx)
        with Convolve(L) as conv:
            A = np.stack(conv.bandpass(X_imp, s, order=10), axis=1)[:,:,0].T 

        Y = A @ V @ B
        Ymag = np.sqrt(np.abs(Y))
        
        peaks_df = find_peaks(wavlen, freqs, Ymag, top_n=5)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#2b2b2b')
        
        splt.plot_contour(ax, wavlen, freqs, Ymag, cmap='Spectral', levels=15)
        splt.overlay_peaks(ax, peaks_df)
        ax.set_title(f"Bus {bus_idx}", color='white', fontsize=14, pad=10)

        fname = f"bus_{bus_idx:03d}.png"
        splt.save_figure(fig, output_dir, fname)
        plt.close(fig)

    print("Batch plotting complete.")

# --- Main Execution Block ---

# 1. Configuration
DIR = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations\Modal SGWT (Journal)\DETECTION_WECC_240"
FILEPATH = f"{DIR}\signal.parquet"

BUS_TARGET = 40     
TIME_TARGET = 2      
T_RANGE = (0, 60)    

# 2. Data Loading
V, t = get_signal(FILEPATH, T_RANGE)

# 3. Domain Definition (Linear Frequency)
s = np.geomspace(1e-3, 1e1, 150)        # Spatial
freqs = np.linspace(0.02, 2.0, 100)     # Frequency
Ts = 1.0 / freqs                        # Temporal Scales
wavlen = np.sqrt(s)

# --- Mode 1: Single Bus Visualization ---
print(f"Running Single Bus Analysis for Bus {BUS_TARGET}...")
A, B = build_transform_matrices(t, s, Ts, BUS_TARGET, TIME_TARGET)
Y = A @ V @ B
Ymag = np.sqrt(np.abs(Y))

peaks_df = find_peaks(wavlen, freqs, Ymag, top_n=5)
#splt.master_plot(wavlen, freqs, Ymag, peaks_df)

# --- Mode 2: System-Wide Heatmap ---
print("Running System-Wide Peak Extraction...")
all_peaks = extract_all_peaks(V, t, s, Ts, wavlen, freqs, TIME_TARGET)
splt.plot_peak_heatmap(all_peaks, wavlen, freqs, dpi=600, output_dir=f"{DIR}\\Contours")

# --- Mode 3: Batch Save All Plots ---
# print("Running Batch Plot Generation...")
# BATCH_DIR = f"{DIR}\\Contours"
# generate_all_bus_plots(V, t, s, Ts, wavlen, freqs, TIME_TARGET, BATCH_DIR)