
from scipy.sparse import load_npz
from numpy import load, save
from sgwt import FastSGWT

KERNEL_NAME = '..\kernels\kernel_model.npz'
LAP_NAME    = '..\laplacians\texas_2000.npz'

# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)
W = load(f'coefficients.npy') # (Bus, Time, Scale)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, KERNEL_NAME)

# Perform Reconstruction, measure performance
f_recon = sgwt.inv(W)

# Save reconstructed signal, if indicated
SAVE_INV = False
if SAVE_INV:
    print('Writing....', end='')
    save('sig_recon.npy', f_recon)
    print('Complete!')
