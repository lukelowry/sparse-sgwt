
from scipy.sparse import load_npz
from numpy import save, sin, cos, pi, load
from pandas import read_csv

from sgwt import FastSGWT, VFKernelData

# Files
DIR = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples'
SCALES_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\scales.npy'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
YBUS_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX_Ybus.npz'
VMAG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vmag.csv'
VANG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vang.csv'

# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
Y = load_npz(YBUS_NAME)

# Load Bus Signal (Bus x Time)
Vmag = (read_csv(VMAG_NAME).set_index('Time').to_numpy()).T
Vang = (read_csv(VANG_NAME).set_index('Time').to_numpy()).T
Vang *= pi/180

# Transform to complex format
V = Vmag*(cos(Vang) + 1j*sin(Vang))
I = Y@V

# Power Flow
S = V * I.conj()

# Scales (loaded from file for consistancy here)
scales = load(SCALES_NAME)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L)

# This works so much faster.
W_P = sgwt.analytical_wavelet_coeffs(S.real, scales)
W_Q = sgwt.analytical_wavelet_coeffs(S.imag, scales)

print(f'Complete!')

save('P.npy', W_P)
save('Q.npy', W_Q)

print(f'Written.')
