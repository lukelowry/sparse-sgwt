
from sgwt import Convolve, impulse
from sgwt.library import DELAY_TEXAS, COORD_TEXAS


# Graph
L = DELAY_TEXAS.get()

'''
Impulse Response of BP
'''

X   = impulse(L, n=600)
X  += impulse(L, n=1800)


# Band pass filter at scale 0.1
with Convolve(L) as conv:

    Y = conv.bandpass(X, [.1])[0]
    Y = conv.bandpass(Y, [.1])[0]


'''
Plotting
'''

from numpy import abs
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

C = COORD_TEXAS.get()
L1, L2 = C['longitude'], C['latitude']

mx = sorted(abs(Y))[-10]
norm = Normalize(-mx, mx)
plt.scatter(L1, L2 , c=Y[:,0], cmap=cm.get_cmap('seismic'), norm=norm)
plt.axis('scaled')   
plt.show()

