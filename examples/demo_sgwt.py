import sgwt 
from sgwt.laplib import IMPEDANCE_TEXAS as graph

L = graph.laplacian()

# Impulse at Vertex 100
b = sgwt.impulse(L,n=0)

# With New Method
fsgwt = sgwt.FastSGWT2(L, [10])
WAVS = fsgwt.wavelet_coeffs(b)
  
print(WAVS)