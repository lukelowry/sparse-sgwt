
from sgwt import Convolve, impulse
from sgwt.library import IMPEDANCE_EASTWEST as L
from sgwt.library import COORD_EASTWEST as C
from sgwt.library import MODIFIED_MORLET as K

# Signal Input
X = impulse(L, n=-1000)

# TODO kernel scaling
K.Q /= 2000  #  g.scale_kern(...)
K.R /= 2000

with Convolve(L) as g:

    Y = g.convolve(X, K)
    
from demo_plot import plot_signal
plot_signal(Y[:,0,0], C, 'Spectral')
