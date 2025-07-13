
from scipy.sparse import load_npz
from numpy import save
from pandas import read_csv
from sgwt import FastSGWT

KERNEL_NAME = '..\kernels\kernel_model.npz'
LAP_NAME    = '..\laplacians\texas_2000.npz'
SIGNAL_NAME = '..\signals\TX2000\frequnecy.csv'

# Load laplacian, old coefficients, and signal
L = load_npz(LAP_NAME)

# Data and format for use # (Bus x Time)
f = (read_csv(SIGNAL_NAME).set_index('Time').to_numpy()-1).T

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, KERNEL_NAME)

# Compute coefficients for only one localization
W_single = sgwt.singleton(f, 1200)

SAVE_SINGLETON = False
if SAVE_SINGLETON:
    print('Writing....', end='')
    save('singleton.npy', W_single)
    print('Complete!')