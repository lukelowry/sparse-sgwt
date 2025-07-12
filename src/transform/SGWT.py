from sksparse.cholmod import cholesky, analyze

import numpy as np
from scipy.sparse import load_npz, csc_matrix
import pandas as pd
import time

class FastSGWT:
    '''
    A rational-approximation approach to the SGWT
    '''

    def __init__(self, L: csc_matrix, kern: str):

        # Sparse Laplacian
        self.L = L

        # Load Residues, Poles, Scales
        npzfile = np.load(f'{kern}.npz')
        self.R, self.Q, self.scales = npzfile['R'], npzfile['Q'], npzfile['S']
        npzfile.close()

        # Wavelet Constant (scalar mult)
        ds = np.log(self.scales[1]/self.scales[0])[0]
        self.C = 1/ds

        # Number of scales
        self.nscales = len(self.scales)

        # Pre-Factor (Symbolic)
        self.factor = analyze(L)

    def allocate(self, f):
        return np.zeros((*f.shape, self.nscales))

    def __call__(self, f):
        '''
        Returns:
            W:  Array size (Bus, Time, Scale)
        '''
        
        W = self.allocate(f)
        F = self.factor
        L = self.L

        for q, r in zip(self.Q, self.R):

            F.cholesky_inplace(L, q) 
            W += F(f)[:, :, None]*r   # Almost the entire duration is occupied multiplying here

        return W
    
    def singleton(self, f, n):
        '''
        Returns:
            Coefficients of f localized at n
        '''
        
        F = self.factor
        L = self.L

        # LOCALIZATION VECTOR
        local = np.zeros((L.shape[0], 1))
        local[n] = 1

        # Singleton Matrix
        W = np.zeros((L.shape[0], self.nscales))

        # Compute
        for q, r in zip(self.Q, self.R):

            F.cholesky_inplace(L, q) 
            W += F(local)*r.T  

        return f.T@W # 
    
    def inv(self, W):
        # The inverse transformation! (For now, only 1 time point)
        # Input W: Bus x Times x Scales
        
        fact, L = self.factor, self.L
        f = np.zeros((W.shape[0], W.shape[1]))

        for q, r in zip(self.Q, self.R):

            fact.cholesky_inplace(L, q) 
            f += fact(W@r) 

        return f/self.C
    
# Interpreter: sgwt_sparse3
# TODO Abstract code to easily pick Full/Inverse/Singleton

# Load the Laplacian and data
dir = r'C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Oscillations'

# Load laplacian, old coefficients, and signal
L = load_npz(f'{dir}\Laplacians\LAP.npz')
data = pd.read_csv(f'{dir}\Bus Freq.csv').set_index('Time').to_numpy()-1
f = data.T # (Bus x Time)
W = np.load(f'{dir}\coefficients.npy') # (Bus, Time, Scale)


# Load SGWT Object from kernel file
sgwt = FastSGWT(L, f'{dir}\kernel_model')

# Select operations to do
DO_SGWT     = True
DO_INV_SGWT = False
DO_SINGLE = False

SAVE_SGWT = True
SAVE_INV = False
SAVE_SINGLETON = False


if DO_SGWT:
    print('SGWT: ')
    start = time.time()
    W_recon = sgwt(f)
    end = time.time()
    dt = (end - start) 
    print(f'{dt:.4f} s')


if DO_INV_SGWT:
    print('Inverse SGWT: ')
    start = time.time()
    f_recon = sgwt.inv(W)
    end = time.time()
    dt = (end - start) 
    print(f'{dt:.4f} s')

if DO_SINGLE:
    # Gets coefficients of all scales at just one bus!

    print('SGWT Singleton: ')
    start = time.time()
    W_single = sgwt.singleton(f, 1200)
    end = time.time()
    dt = (end - start) 
    print(f'{dt:.4f} s')

    # Temp doing this multiplication afterward
    #W_single = f.T@W_single
    #end = time.time()
    #dt = (end - start) 
    #print(f'Multipl {dt:.4f} s')

if SAVE_SGWT:
    print('Writing....', end='')
    np.save(f'{dir}\coefficients.npy', W_recon)
    print('Complete!')

if SAVE_INV:
    print('Writing....', end='')
    np.save(f'{dir}\sig_recon.npy', f_recon)
    print('Complete!')

if SAVE_SINGLETON:
    print('Writing....', end='')
    np.save(f'{dir}\singleton.npy', W_single)
    print('Complete!')

    #singleton