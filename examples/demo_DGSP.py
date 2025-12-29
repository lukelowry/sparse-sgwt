
from sgwt import DyConvolve, impulse
from sgwt.library import DELAY_TEXAS, COORD_TEXAS
from demo_plot import plot_signal

# Graph
L = DELAY_TEXAS.get()
C = COORD_TEXAS.get()

# Impulse
X  = impulse(L, n=1200)
#X += impulse(L, n=600)

# Pre-Determined Poles
scales = [0.1, 1, 10, 100]
poles = [1/s for s in scales]

# Should get same answer as demo_filters_1, maybe


# The tradeoff for efficient graph updates is that poles cannot change
with DyConvolve(L, poles) as conv:

    # BEFORE CLOSE
    BP = conv.bandpass(X)
    plot_signal(BP[0][:,0], C, 'seismic')

    # OH SHIT THAT WORKED!
    conv.addbranch(1200, 600, 1e6)

    # AFTER 
    BP = conv.bandpass(X)
    plot_signal(BP[0][:,0], C, 'seismic')
    

# NOTE Description:
# The above shows how we can dynamically update the graph and still
# obtain convolutions with incredible speed
# This is owed to the Cholesky Decomposition and Kernel Fitting