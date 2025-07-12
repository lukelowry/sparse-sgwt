
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
data = read_csv(f'{dir}\Bus Freq.csv').set_index('Time').to_numpy()-1
f = data.T # (Bus x Time)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, f'{dir}\kernel_model')

# Compute coefficients for only one localization
W_single = sgwt.singleton(f, 1200)

SAVE_SINGLETON = False
if SAVE_SINGLETON:
    print('Writing....', end='')
    save(f'{dir}\singleton.npy', W_single)
    print('Complete!')