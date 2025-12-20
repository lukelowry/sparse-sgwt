
from sgwt import Convolve, impulse
from sgwt.data import DELAY_TEXAS, COORD_TEXAS
import numpy as np

from plot_points import plot_signal

# Graph
L = DELAY_TEXAS.get().copy()/(2*np.pi*60)
C = COORD_TEXAS.get()

# Impulse
X = impulse(L, n=700)


# Scales
s = np.logspace(1e-5, 1e-1, num = 5)

# Memory Efficient Context
with Convolve(L) as conv:

    #LP = conv.lowpass(X, s)
    LP = conv.bandpass(X, s)
    LP = conv.bandpass(LP[0], s)
    #HP = conv.highpass(X, s)

plot_signal(LP[0][:,0], C, 'seismic')



