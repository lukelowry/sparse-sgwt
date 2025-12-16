
import numpy as np
import sgwt 
from sgwt.laplib import IMPEDANCE_TEXAS as graph
import timeit

L = graph.laplacian()

# Impulse at Vertex 100
b = sgwt.impulse(L,n=0).reshape(-1,1)

scales = [10]

# With old method
fsgwt1 = sgwt.FastSGWT(L, scales)

start = timeit.timeit()
WAVS1 = fsgwt1.scaling_coeffs(b)
end = timeit.timeit()
print("scikit: ", end - start)


# With New Method
fsgwt2 = sgwt.FastSGWT2(L, scales)

start = timeit.timeit()
WAVS2 = fsgwt2.scaling_coeffs(b)
end = timeit.timeit()
print("DLL: ", end - start)

print("Max Error in Methods")

print(np.max(np.abs(WAVS1-WAVS2)))