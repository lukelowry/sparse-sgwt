import os
import matplotlib.pyplot as plt

# DOC_START_CODE_EXCLUDE_IMPORTS
from sgwt import Convolve
from sgwt import IMPEDANCE_WECC as L
import numpy as np

# Bus Index, Latitude (Y), Longitude (X)
MEASURMENTS = [
    [ 191    ,-122.45    ,46.719],
    [ 202    ,-101.33    ,46.5  ],
    [  17    ,-112.24    ,32.52 ],
    [ 131    ,-121.58    ,39.59 ],
    [159    ,-110.107   ,41.395],
    [33      ,-116.778   ,35.14 ],
    [187     ,-123.14    ,44.34 ]
]

nbus = L.shape[0]
X = np.zeros((nbus, 2)) # Signal, Sparse
Xh = np.zeros_like(X) # Reconstruction, Dense

# Load Sparse Signal
for idx, long, lat in MEASURMENTS:
    X[idx] = long, lat

# Sampling operator
J = np.diagflat(X[:,0]!=0)

# Scale of Recon
s = [5]

with Convolve(L) as conv:

    for i in range(7000):
        B = (X - J@Xh).copy(order='F')
        dX = conv.lowpass(B, s)
        Xh += s * dX[0]
# DOC_END_CODE_EXCLUDE_PLOT
plt.figure(figsize=(8, 6)) # Create a figure for this plot
plt.scatter(Xh[:,0], Xh[:,1] , c='k', edgecolors='none', label='Reconstructed Signal')
plt.scatter(X[:,0][X[:,0]!=0], X[:,1][X[:,1]!=0], c='r', label='Sparse Measurements')
plt.axis('scaled')
plt.title('Signal Reconstruction from Sparse Measurements')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# Save the figure for documentation
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
static_images_dir = os.path.join(project_root, 'docs', '_static', 'images')

# Ensure the directory exists
os.makedirs(static_images_dir, exist_ok=True)
save_path = os.path.join(static_images_dir, 'demo_recon_signal.png')
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()
