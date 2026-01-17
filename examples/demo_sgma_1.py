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

BUS_TARGET = 40      # Target bus index for analysis
TIME_TARGET = 2.0    # Time (s) to center the temporal wavelet
T_RANGE = (0, 60)    # Time range (s) of the signal
ORDER = 10           # Order of the spatial bandpass filter
TOP_N = 5            # Number of peaks to extract
W0 = 2 * np.pi       # Central frequency of the temporal wavelet

spatial_scales = np.geomspace(1e-3, 1e1, 150)
temporal_freqs = np.linspace(0.02, 2.0, 100)
sgma = SGMA(L, s=spatial_scales, freqs=temporal_freqs, time_target=TIME_TARGET, order=ORDER, w0=W0)

V, t = get_signal(FILEPATH, T_RANGE)
Y_mag    = sgma.transform(V, t, bus_idx=BUS_TARGET)
peaks = sgma.peaks_from_spectrum(Y_mag, top_n=TOP_N)
# DOC_END_CODE_EXCLUDE_PLOT

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#2b2b2b') # Dark gray figure bg
splt.plot_contour(ax, sgma.wavlen, sgma.freqs, Y_mag, cmap='Spectral', levels=15)
splt.overlay_peaks(ax, peaks)

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_sgma_1.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
