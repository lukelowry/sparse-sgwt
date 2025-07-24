"""
main.py

Will be obstracted eventually. Core class for now, implementing
the VF version of the SGWT.

Author: Luke Lowery (lukel@tamu.edu)
"""

from sksparse.cholmod import analyze
from scipy.sparse import csc_matrix

import numpy as np
from .kernel import VFKernelData

class FastSGWT:
    '''
    A rational-approximation approach to the SGWT
    '''

    def __init__(self, L: csc_matrix, kern: VFKernelData):

        # Sparse Laplacian
        self.L = L

        # Load Residues, Poles, Scales
        self.R, self.Q, self.S = kern.R, kern.Q, kern.S

        # Wavelet Constant (scalar mult)
        ds = np.log(self.S[1]/self.S[0])[0]
        self.C = 1/ds

        # Number of scales
        self.nscales = len(self.S)

        # Pre-Factor (Symbolic)
        self.factor = analyze(L)

    def allocate(self, f):
        return np.zeros((*f.shape, self.nscales))

    def __call__(self, f):
        '''
        Returns
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
        Returns
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

        return f.T@W 
    
    def inv(self, W):
        '''
        Description
            The inverse SGWT transformation (only one time point for now)
            And does not support scaling coefficients right now.
        Parameters
            W: ndarray of shape (Bus x Times x Scales)
        '''
        
        fact, L = self.factor, self.L
        f = np.zeros((W.shape[0], W.shape[1]))

        for q, r in zip(self.Q, self.R):

            fact.cholesky_inplace(L, q) 
            f += fact(W@r) 

        return f/self.C
    