
import time
from scipy.sparse import load_npz
from numpy import save, sin, cos, pi, empty_like
from pandas import read_csv

from sgwt import FastSGWT

DIR = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples'
KERNEL_NAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model'
LAP_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX2000.npz'
YBUS_NAME    = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\laplacians\TX_Ybus.npz'


# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
Y = load_npz(YBUS_NAME)

# Data and format for use # (Bus x Time)
VMAG_NAME = f'{DIR}\signals\TX2000\\vmagnitude.csv'
VANG_NAME = f'{DIR}\signals\TX2000\\vangle.csv'
Vmag = (read_csv(VMAG_NAME).set_index('Time').to_numpy()-1).T
Vang = (read_csv(VANG_NAME).set_index('Time').to_numpy()-1).T

# Down Sample
samples = 400
njump = int(Vmag.shape[1]/samples)
Vmag = Vmag[:,::njump]
Vang = Vang[:,::njump]



# Transform
Vang *= pi/180
Vr = Vmag*cos(Vang)
Vi = Vmag*sin(Vang)
V = Vr + 1j*Vi

# Calculate Current
I = Y@V
Ir = I.real
Ii = I.imag

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, KERNEL_NAME)

# SGWT of Voltage
W_vr = sgwt(Vr)
print('Vr Done')
W_vi = sgwt(Vi)
print('Vi Done')

# SGWT of Admittance 
Ynp = Y.toarray()
W_G = sgwt(Ynp.real)
print('Conductanace (G) Done')
W_B = sgwt(Ynp.imag)
print('Susceptance (B) Done')



# NOTE I want to make a function
#      that lets me call sgwt() but pick a scale, instead of all scales.

# TODO need forced oscillation simulation

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
