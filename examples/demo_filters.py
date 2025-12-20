
from sgwt import Convolve
from sgwt.data import IMPEDANCE_EASTWEST as graph
import numpy as np

# Graph
L = graph.get()
ntime = 20
nscales = 5

# Signal Input
shape = (L.shape[0], ntime)
X = np.zeros(shape, order="F")
X[50] = 1

# Scales
s = np.logspace(1e-2, 1e1, nscales)

# Memory Efficient Context
with Convolve(L) as conv:

    LP = conv.lowpass(X, s)

    BP = conv.bandpass(X, s)

    HP = conv.highpass(X, s)



