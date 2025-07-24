
from scipy.sparse import load_npz
from numpy import save, sin, cos, pi
from pandas import read_csv

from sgwt import FastSGWT, VFKernelData

# Files
DIR = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples'
KERNEL = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model.npz'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
YBUS_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX_Ybus.npz'
VMAG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vmag.csv'
VANG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vang.csv'

# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
Y = load_npz(YBUS_NAME)

# Load Bus Signal (Bus x Time)
Vmag = (read_csv(VMAG_NAME).set_index('Time').to_numpy()-1).T
Vang = (read_csv(VANG_NAME).set_index('Time').to_numpy()-1).T
Vang *= pi/180

# Transform to complex format
V = Vmag*(cos(Vang) + 1j*sin(Vang))
I = Y@V

# Different from power.py, 
# We first calculate power and then take coefficients.
# Unsure which is better just yet
S = V * I.conj()


# Kernel File
kern = VFKernelData.from_file(KERNEL)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, kern)


# SGWT of Voltage
W_P = sgwt(S.real)
print('Real Power: Done')

W_Q = sgwt(S.imag)
print('Reactive Power: Done')

save('P.npy', W_P)
save('Q.npy', W_Q)
print(f'Complete!')
