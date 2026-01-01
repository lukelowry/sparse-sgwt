from sgwt import Convolve, DyConvolve, impulse
from sgwt import DELAY_USA as L
import numpy as np
import time 

# Impulse
X  = impulse(L, n=1200)

# Pre-Determined Polesp
scales = np.geomspace(1e-5, 1e2, 20)
poles = 1/scales

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
