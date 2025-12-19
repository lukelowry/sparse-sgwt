
import sgwt
from sgwt.data import LENGTH_EASTWEST as graph
from sgwt.data import MODIFIED_MORLET as kern
from sgwt.data import COORD_EASTWEST as coords
import numpy as np

def plot_signal(f):

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Coordinates
    C = coords.get()
    L1, L2 = C['longitude'], C['latitude']


    mx = np.sort(np.abs(f))[-20] 
    norm = Normalize(-mx, mx)
    plt.scatter(L1, L2 , c=f, edgecolors='none', cmap=cm.get_cmap('Spectral'), norm=norm)
    plt.axis('scaled')   
    plt.show()

# Graph & Kernel
L = graph.get()
K = kern.get()

# Signal Input
ntime = 2
X = np.zeros(
    shape=(L.shape[0], ntime), 
    order="F"
)
X[-10000] = 1

# Memory Efficient Context
with sgwt.VFitDLL(L, K) as g:

    g.Q /= 2000000 # TODO kernel scaling g.scale_kern(...)

    H = g.convolve(X)

plot_signal(H[:,0,0])
