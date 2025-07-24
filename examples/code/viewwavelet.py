
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
from sgwt import FastSGWT, VFKernelData


KERNEL = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model.npz'


KERNEL_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
SIGNAL_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\signals\TX2000\frequency.csv'

# Load laplacian, old coefficients, and signal



# Data and format for use # (Bus x Time)
nbuses  = 2000
impulse = 1200
f = zeros((nbuses, 1))
f[impulse] = 1

# Kernel File & Laplacian
kern = VFKernelData.from_file(KERNEL)
L = load_npz(LAP_NAME)

# SGWT Model
sgwt = FastSGWT(L, kern)

# Compute SGWT
W = sgwt(f)

# Output
fname = 'coefficients.npy'
save(fname, W)
print(f'Complete!\n Saved to {fname}')
