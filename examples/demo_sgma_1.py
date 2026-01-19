import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import demo_plot as splt

_config_path = os.path.join(os.path.dirname(__file__), ".data_dir")
DATA_DIR = open(_config_path).read().strip() if os.path.exists(_config_path) else "."
FILEPATH = os.path.join(DATA_DIR, "signal.parquet")

def get_signal(fname, t_range):
    Vdata = pd.read_parquet(fname).to_numpy()
    nbus = Vdata.shape[1] // 2
    V = (Vdata[:, :nbus] * np.exp(1j * Vdata[:, nbus:] / 180 * np.pi)).T
    signal = np.cumsum(np.diff(V, axis=1), axis=1)
    time = np.linspace(t_range[0], t_range[1], signal.shape[1])
    return signal, time

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import SGMA
from sgwt import LENGTH_WECC as L

# Signals: Real or Complex Matrix (Rows: Buses, Cols: Time)
V, t = get_signal(FILEPATH, t_range=(0, 60))

# SGMA Parameters
BUS_TARGET = 36
TIME_TARGET = 2.0
ORDER = 1
TOP_N = 3

wmin = 1
wmax = 3e3
nscales = 150
spatial_scales = np.geomspace(wmin**2, wmax**2, nscales)  

temporal_freqs = np.linspace(0.05, 2.0, 100)
sgma = SGMA(L, spatial_scales, temporal_freqs, order=ORDER, w0=2*np.pi)

# Get complex spectrum (compute once)
M = sgma.spectrum(V, t, BUS_TARGET, TIME_TARGET, return_complex=True)

# Identify modes with frequency, damping ratio, wavelength, and magnitude
modes = sgma.find_modes(M, top_n=TOP_N)
# DOC_END_CODE_EXCLUDE_PLOT

print(modes)

# For plotting, get magnitude spectrum and peak locations
Mabs = np.abs(M)
peaks = sgma.find_peaks(Mabs, top_n=TOP_N, return_indices=True)

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#2b2b2b')
splt.plot_contour(ax, sgma.wavlen, sgma.freqs, Mabs, cmap='Spectral', levels=35)
splt.overlay_peaks(ax, peaks)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_sgma_1.png'), dpi=400, bbox_inches='tight')
plt.show()
