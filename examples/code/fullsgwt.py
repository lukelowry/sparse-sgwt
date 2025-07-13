
import time
from scipy.sparse import load_npz
from numpy import save
from pandas import read_csv

from sgwt import FastSGWT

KERNEL_NAME = '..\kernels\kernel_model.npz'
LAP_NAME    = '..\laplacians\texas_2000.npz'
SIGNAL_NAME = '..\signals\TX2000\frequnecy.csv'

# Load laplacian, old coefficients, and signal
L    = load_npz(LAP_NAME)

# Data and format for use # (Bus x Time)
f = (read_csv(SIGNAL_NAME).set_index('Time').to_numpy()-1).T

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, KERNEL_NAME)

# Time and execute
print('SGWT: ')
start = time.time()

W = sgwt(f)

end = time.time()
dt = (end - start) 
print(f'{dt:.4f} s')


SAVE_SGWT = True
if SAVE_SGWT:
    print('Writing....', end='')
    save(f'coefficients.npy', W)
    print('Complete!')
