from sgwt import Convolve, impulse
from sgwt import DELAY_USA as L
from sgwt import COORD_USA as C

# Impulse
X  = impulse(L, n=15000)

# Scales
s = [3e0]

# Memory Efficient Context
with Convolve(L) as conv:

    BP = conv.bandpass(X, s)[0]
    BP = conv.bandpass(BP, s)[0]
    BP = conv.bandpass(BP, s)[0]

from demo_plot import plot_signal
plot_signal(BP[:,0], C, 'coolwarm')
