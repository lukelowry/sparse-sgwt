# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: examples/demo_recon.py
Description: Reconstructs geographical coordinates from sparse measurements.
"""

from sgwt import Convolve
from sgwt import IMPEDANCE_WECC as L
import numpy as np


# Bus Index, atitude (Y), Longitude (X)
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
s = 5

with Convolve(L) as conv:

    for i in range(7000):

        B = (X - J@Xh).copy(order='F')

        dX = conv.lowpass(B, [s])

        Xh += s * dX[0]


import matplotlib.pyplot as plt
plt.scatter(Xh[:,0], Xh[:,1] , c='k', edgecolors='none')
plt.scatter(X[:,0][X[:,0]!=0], X[:,1][X[:,1]!=0], c='r')
plt.axis('scaled')   
plt.show()
