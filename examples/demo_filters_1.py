
from sgwt import Convolve, impulse
from sgwt.library import DELAY_TEXAS as L
from sgwt.library import COORD_TEXAS as C

# Impulse
X  = impulse(L, n=1200)
X += impulse(L, n=600)

# Scales
s = [1e-1]

# Memory Efficient Context
with Convolve(L) as conv:

    LP = conv.lowpass(X, s)
    BP = conv.bandpass(X, s)
    HP = conv.highpass(X, s)

from demo_plot import plot_signal
plot_signal(BP[0][:,0], C, 'seismic')



