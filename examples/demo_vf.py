
import sgwt
from sgwt.data import LENGTH_EASTWEST, COORD_EASTWEST, MODIFIED_MORLET
import numpy as np

# Graph & Kernel
L = LENGTH_EASTWEST.get()
K = MODIFIED_MORLET.get()
C = COORD_EASTWEST.get()

def plot_signal(f):

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Coordinates
    L1, L2 = C['longitude'], C['latitude']

    mx = np.sort(np.abs(f))[-20] 
    norm = Normalize(-mx, mx)
    plt.scatter(L1, L2 , c=f, edgecolors='none', cmap=cm.get_cmap('Spectral'), norm=norm)
    plt.axis('scaled')   
    plt.show()

# Signal Input
ntime = 2
X = np.zeros(
    shape=(L.shape[0], ntime), 
    order="F"
)
X[-10000] = 1

# Memory Efficient Context
with sgwt.VFConvolve(L, K) as g:

    g.Q /= 20000000 # TODO kernel scaling g.scale_kern(...)

    H = g.convolve(X)

plot_signal(H[:,0,0])
