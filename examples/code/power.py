
import time
from scipy.sparse import load_npz
from numpy import save, sin, cos, pi, empty_like
from pandas import read_csv

from sgwt import FastSGWT, VFKernelData

DIR = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples'
KERNEL = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model.npz'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
YBUS_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX_Ybus.npz'


# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
Y = load_npz(YBUS_NAME)

# Load Bus Signal (Bus x Time)
VMAG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vmag.csv'
VANG_NAME = f'{DIR}\signals\TX2000\\forced\\fo_bus_vang.csv'
Vmag = (read_csv(VMAG_NAME).set_index('Time').to_numpy()-1).T
Vang = (read_csv(VANG_NAME).set_index('Time').to_numpy()-1).T

# Kernel File
kern = VFKernelData.from_file(KERNEL)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, kern)

# Transform to complex format
# NOTE only convert if stored in degrees 
Vang *= pi/180
Vr = Vmag*cos(Vang)
Vi = Vmag*sin(Vang)

V = Vr + 1j*Vi

# SGWT of Voltage
W_vr = sgwt(Vr)
W_vi = sgwt(Vi)
print('V real and imag: Done')

# SGWT of Admittance 
Ynp = Y.toarray()
W_G = sgwt(Ynp.real)
print('Conductanace (G) Done')
W_B = sgwt(Ynp.imag)
print('Susceptance (B) Done')



# NOTE I want to make a function
#      that lets me call sgwt() but pick a scale, instead of all scales.

# TODO Need to consider current from impendence modeled loads, in additiona to 
# network, so that the ONLY current we observe is caused by losses.

# Compute the current induced
# By the scale-dependent voltage
W_ir = empty_like(W_vr)
W_ii = empty_like(W_vi)

for i, W in enumerate(zip(W_vr.T, W_vr.T)):
    
    # Scale Dependent Admittance
    G, B = W_G[:,:,i], W_B[:,:,i]
    Yc = G + 1j*B

    # Scale Depednent Voltage
    Wr, Wi = W
    Wc = Wr.T + 1j*Wi.T

    # Scale Dependent Current
    Ic = Yc@Wc
    W_ir[:,:,i] = Ic.real
    W_ii[:,:,i] = Ic.imag

print('Ir Done')
print('Ii Done')


save('Vr_coeff.npy', W_vr)
save('Vi_coeff.npy', W_vi)
save('Ir_coeff.npy', W_ir)
save('Ii_coeff.npy', W_ii)
print(f'Complete!')
