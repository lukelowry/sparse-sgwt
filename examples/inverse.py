
import time
from scipy.sparse import load_npz, csc_matrix
from numpy import load, save
from pandas import read_csv
from main import FastSGWT


# Interpreter: sgwt_sparse3

# Temp: Working Directory
dir = r'C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations'

# Load laplacian, old coefficients, and signal
L = load_npz(f'{dir}\Laplacians\LAP.npz')
W = load(f'{dir}\coefficients.npy') # (Bus, Time, Scale)


# Make SGWT Object
sgwt = FastSGWT(L, f'{dir}\kernel_model')

# Perform Reconstruction, measure performance
f_recon = sgwt.inv(W)

# Save reconstructed signal, if indicated
SAVE_INV = False
if SAVE_INV:
    print('Writing....', end='')
    save(f'{dir}\sig_recon.npy', f_recon)
    print('Complete!')
