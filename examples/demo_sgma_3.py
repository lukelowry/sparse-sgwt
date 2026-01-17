import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

DIR = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations\Modal SGWT (Journal)\DETECTION_WECC_240"
FILEPATH = f"{DIR}\signal.parquet"

TIME_TARGET = 2.0    # Time (s) to center the temporal wavelet
T_RANGE = (0, 60)    # Time range (s) of the signal
ORDER = 10           # Order of the spatial bandpass filter
TOP_N = 2            # Number of peaks to extract per bus
W0 = 2 * np.pi        # Central frequency of the temporal wavelet

spatial_scales = np.geomspace(1e-3, 1e1, 150)
temporal_freqs = np.linspace(0.02, 2.0, 100)
sgma = SGMA(L, s=spatial_scales, freqs=temporal_freqs, time_target=TIME_TARGET, order=ORDER, w0=W0)

V, t = get_signal(FILEPATH, T_RANGE)
V = np.abs(V)

all_peaks, cluster_peaks = sgma.find_system_wide_peaks(V, t, top_n=TOP_N)

splt.plot_peak_heatmap(all_peaks, sgma.wavlen, sgma.freqs, dpi=600, output_dir=f"{DIR}\\Density")
