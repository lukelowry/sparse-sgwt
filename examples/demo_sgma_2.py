import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
_config_path = os.path.join(os.path.dirname(__file__), ".data_dir")
DATA_DIR = open(_config_path).read().strip() if os.path.exists(_config_path) else "."

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

FILEPATH = os.path.join(DATA_DIR, "signal.parquet")
TIME_TARGET = 2.0
T_RANGE = (0, 60)
N_RANDOM_BUSES = 140
ORDER = 10
TOP_N = 5
W0 = 2 * np.pi * 1

spatial_scales = np.geomspace(1e-3, 1e1, 150)
temporal_freqs = np.linspace(0.02, 2.0, 100)
sgma = SGMA(L, spatial_scales, temporal_freqs, TIME_TARGET, order=ORDER, w0=W0)

V, t = get_signal(FILEPATH, T_RANGE)

subset = np.random.choice(L.shape[0], N_RANDOM_BUSES, replace=False)
local_peaks, global_peaks = sgma.find_system_wide_peaks(V, t, subset, top_n=TOP_N)
# DOC_END_CODE_EXCLUDE_PLOT

splt.plot_peak_heatmap(local_peaks, sgma.wavlen, sgma.freqs, dpi=600)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')
os.makedirs(static_images_dir, exist_ok=True)
plt.savefig(os.path.join(static_images_dir, 'demo_sgma_2.png'), dpi=400, bbox_inches='tight')
plt.show()
