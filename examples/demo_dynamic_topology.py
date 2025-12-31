# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: examples/demo_dynamic_topology.py
Description: Demonstration of dynamic graph updates using DyConvolve.
"""
from sgwt import DyConvolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C

# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Poles
scales = [0.1, 1, 10]
poles = [1/s for s in scales]

# The tradeoff for efficient graph updates is that poles cannot change
with DyConvolve(L, poles) as conv:

    # Pre-Close Convolution
    Y_before = conv.bandpass(X)
 
    # Add Branch, effectively making Bus 1200 and 600 Neighbors.
    # We should expect Bus 600 to have positive value similar to 1200.
    tau = 1e-3
    conv.addbranch(1200, 600, 1/tau**2)

    # Post-Close Convolution
    Y_after = conv.bandpass(X)
    
from demo_plot import plot_signal
plot_signal(Y_before[0][:,0], C, 'seismic')
plot_signal(Y_after[0][:,0], C, 'seismic')
