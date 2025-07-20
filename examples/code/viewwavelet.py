
"""
viewwavelet.py

Generates a coefficient matrix which is technically
coefficients for the dirac function, which of course
results in the wavelet basis functions at one location.

Use another program to visualize the output.

Author: Luke Lowery (lukel@tamu.edu)
"""

from scipy.sparse import load_npz
from numpy import save, zeros

from sgwt import FastSGWT

KERNEL_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
SIGNAL_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\signals\TX2000\frequency.csv'

# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
f = zeros((2000, 1))

# Data and format for use # (Bus x Time)
impulse = 1200
f[impulse] = 1

# Make SGWT operator
sgwt = FastSGWT(L, KERNEL_NAME)

# Compute SGWT
W = sgwt(f)

# Output
fname = 'coefficients.npy'
save(fname, W)
print(f'Complete!\n Saved to {fname}')
