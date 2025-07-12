
import time
from scipy.sparse import load_npz
from numpy import save
from pandas import read_csv
from main import FastSGWT


# NOTE Interpreter: sgwt_sparse3

# Working Directory
dir = r'C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations'

# Load laplacian, old coefficients, and signal
L = load_npz(f'{dir}\Laplacians\LAP.npz')
data = read_csv(f'{dir}\Bus Freq.csv').set_index('Time').to_numpy()-1
f = data.T # (Bus x Time)

# Load SGWT Object from kernel file
sgwt = FastSGWT(L, f'{dir}\kernel_model')

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
    save(f'{dir}\coefficients.npy', W)
    print('Complete!')
