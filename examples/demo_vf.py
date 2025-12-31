# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: examples/demo_vf.py
Description: Demonstration of Vector Fitting (VF) kernel convolution.
"""
from sgwt import DyConvolve, impulse, VFKern
from sgwt import IMPEDANCE_EASTWEST as L
from sgwt import COORD_EASTWEST as C
from sgwt import MODIFIED_MORLET as Kjson

# Signal Input
X = impulse(L, n=-1000)

# TODO kernel scaling  #  g.scale_kern(...)
K = VFKern.from_dict(Kjson)
K.Q /= 2000
K.R /= 2000


with DyConvolve(L, K) as g:

    Y = g.convolve(X)
    
from demo_plot import plot_signal
plot_signal(Y[:,0,0], C, 'Spectral')
