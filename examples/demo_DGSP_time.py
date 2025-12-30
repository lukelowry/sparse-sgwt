'''
Description
    Measure the Computation Speed betwenn Static and Dynamic Methods.
'''


from sgwt import Convolve, DyConvolve, impulse
from sgwt.library import DELAY_USA, COORD_USA
from demo_plot import plot_signal
import numpy as np
import time 

# Graph
L = DELAY_USA.get()
C = COORD_USA.get()

# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Polesp
scales = np.geomspace(1e-5, 1e2, 20)
poles = 1/scales

# TODO Measure Runtime and compare



with Convolve(L) as conv:

    start = time.time()
    for i in range(2):
        Y = conv.bandpass(X, scales)
    T1 = time.time() - start


with DyConvolve(L, poles) as conv:

    start = time.time()
    for i in range(2):
        Y = conv.bandpass(X)
    T2 = time.time() - start

print(f"Static: {T1*1000:.3f} ms")
print(f"Dynamic: {T2*1000:.3f} ms")
