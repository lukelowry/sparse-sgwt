
import time
from scipy.sparse import load_npz
from numpy import save
from pandas import read_csv

from sgwt import FastSGWT

KERNEL_NAME = '..\kernels\kernel_model.npz'
LAP_NAME    = '..\laplacians\texas_2000.npz'


# Load laplacian, old coefficients, and signal
L    = load_npz(LAP_NAME)
data = read_csv('Bus Freq.csv').set_index('Time').to_numpy()-1
f    = data.T # (Bus x Time)

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
