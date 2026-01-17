import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DIR = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations\Modal SGWT (Journal)\DETECTION_WECC_240"

# DOC_START_CODE_EXCLUDE_IMPORTS
import demo_plot as splt
from sgwt import SGMA
from sgwt import DELAY_WECC as L

def get_signal(fname, t_range):
    Vdata = pd.read_parquet(fname).to_numpy()
    nbus = Vdata.shape[1] // 2
    V = (Vdata[:, :nbus] * np.exp(1j * Vdata[:, nbus:] / 180 * np.pi)).T
    signal = np.cumsum(np.diff(V, axis=1), axis=1)
    time = np.linspace(t_range[0], t_range[1], signal.shape[1])
    return signal, time

FILEPATH = f"{DIR}\signal.parquet"

TIME_TARGET = 2.0    # Time (s) to center the temporal wavelet
T_RANGE = (0, 60)    # Time range (s) of the signal
N_RANDOM_BUSES = 140 # Number of random buses to analyze
ORDER = 10           # Order of the spatial bandpass filter
TOP_N = 5            # Number of peaks to extract per bus
F0 = 1
W0 = 2 * np.pi  *F0     # Central frequency of the temporal wavelet

# NOTE IT WORKS!

spatial_scales = np.geomspace(1e-3, 1e1, 150)
temporal_freqs = np.linspace(0.02, 2.0, 100)
sgma = SGMA(L, s=spatial_scales, freqs=temporal_freqs, time_target=TIME_TARGET, order=ORDER, w0=W0)

V, t = get_signal(FILEPATH, T_RANGE)

print(f"\n--- Finding peaks for {N_RANDOM_BUSES} random buses ---")
subset_bus_indices = np.random.choice(L.shape[0], N_RANDOM_BUSES, replace=False)
subset_peaks, cluster_peaks = sgma.find_system_wide_peaks(V, t, subset_bus_indices, top_n=TOP_N)

print("Cluster Peaks:")
print(cluster_peaks)
# DOC_END_CODE_EXCLUDE_PLOT

splt.plot_peak_heatmap(subset_peaks, sgma.wavlen, sgma.freqs, dpi=600)

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_sgma_2.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
