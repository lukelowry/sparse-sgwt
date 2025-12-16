
import sgwt 
from sgwt.laplib import IMPEDANCE_TEXAS as graph

L = graph.laplacian()

# sgwt object
fsgwt = sgwt.FastSGWT2(L, scales=[2])

# Impulse at Vertex 100
b = sgwt.impulse(L,n=100)

# Wavelet Coefficients
WAVS = fsgwt.wavelet_coeffs(b)
  