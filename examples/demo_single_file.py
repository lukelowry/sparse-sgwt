import os
from numpy import abs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sgwt import Convolve, impulse
from sgwt import DELAY_TEXAS as L
from sgwt import COORD_TEXAS as C

X   = impulse(L, n=600)
X  += impulse(L, n=1800)

# Band pass filter at scale 0.1
with Convolve(L) as conv:

    Y = conv.bandpass(X, [.1])[0]
    Y = conv.bandpass(Y, [.1])[0]


mx = sorted(abs(Y))[-10]
norm = Normalize(-mx, mx)
plt.scatter(C[:,0], C[:,1] , c=Y[:,0], cmap='seismic', norm=norm)
plt.axis('scaled')   
plt.show()
