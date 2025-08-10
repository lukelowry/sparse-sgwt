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

    def __init__(self, L: csc_matrix, kern: VFKernelData = None):

        # Sparse Laplacian
        self.L = L

        # Pre-Factor (Symbolic)
        self.factor = analyze(L)

        # Initialize kernel if passed
        if kern is not None:
            self.init_kern(kern)
            self.kern = kern

    def init_kern(self, kern: VFKernelData):

        # Load Residues, Poles, Scales
        self.R, self.Q, self.S = kern.R, kern.Q, kern.S

        # Wavelet Constant (scalar mult)
        if len(self.S) > 1:
            ds = np.log(self.S[1]/self.S[0])[0]
            self.C = 1/ds
        else:
            self.C = 1

        # Number of scales
        self.nscales = len(self.S)


    def allocate(self, f, n=None):
        if n is None:
            return np.zeros((*f.shape, self.nscales))
        else:
            return np.zeros((*f.shape, n))
    
    '''
    GENERATLIZED SGWT
    '''

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
    
    ''' GENERALIZED SCALING COEFFS '''

    def scaling_coeffs(self, f, s):
        '''
        Returns
            ALL Scaling Coefficients of f at scale S
        '''
        
        F = self.factor
        L = self.L

        # TODO determine ideal scaling of poles based on 
        # the VF form of scaling function


        # Singleton Matrix
        W = np.zeros((L.shape[0], self.nscales))

        # Compute
        for q, r in zip(self.Q, self.R):

            F.cholesky_inplace(L, q) 
            W += F(f)*r.T  

        return f.T@W 
    
    def scaling_funcs(self, anchor_indecies, scale=None):
        '''
        Returns
            Scaling functions of indicated scale
            at specified anchors (localizations)
        Parameters:
            anchor_indicies: nodes at which to return scaling functions
            scale: scale of the scaling functions
        '''
        
        F = self.factor
        L = self.L
        

        # Get the default/reference scale
        base_scale = self.S[0]

        # Use default scale if none given
        if scale is None:
            scale = base_scale

        # Create the LOCALIZATION VECTOR
        # Number of Rows = Number of true verticies
        # Number of Cols = Number of Reduced vertices
        nLocal = len(anchor_indecies)
        anchors = np.zeros((L.shape[0], nLocal))

        for i, node_idx in enumerate(anchor_indecies):
            anchors[node_idx, i] = 1

        # Scaling Function Matrix (The columns vectors are each scaling function)
        S = np.zeros_like(anchors)

        # Iterate Scaling Kernel Poles
        for q, r in zip(self.Q, self.R):

            per_unit_scale = scale/base_scale
            
            # NOTE something weird happening, scales are inverse of what
            # they should be!
            # Dilate each pole to the new scale
            qscaled = q/per_unit_scale

            # Solve
            F.cholesky_inplace(L, qscaled) 
            S += F(anchors)*r.T  

        return S
    
    '''
    The methods below are for ANALYTICAL representations of the kernel
    functions. If a custom vector fitted function is desired,
    the above must be used.
    '''
    
    def analytical_scaling_funcs(self, anchor_indecies, scale):
        '''
        Returns
            Scaling functions of indicated scale using the analytical form.
        Parameters:
            anchor_indicies: nodes at which to return scaling functions
            scale: scale of the scaling functions
        '''
        
        F = self.factor
        L = self.L

        # Create the LOCALIZATION VECTOR
        # Number of Rows = Number of true verticies
        # Number of Cols = Number of Reduced vertices
        nLocal = len(anchor_indecies)
        anchors = np.zeros((L.shape[0], nLocal))

        for i, node_idx in enumerate(anchor_indecies):
            anchors[node_idx, i] = 1

        # Analytical solution to scaling function
        F.cholesky_inplace(L, 1/scale)
        S = F(anchors)

        # Sovle again, temp test
        S = F(S)/scale

        return S/scale##/np.sqrt(2*np.pi*scale)
    
    def analytical_wavelet_funcs(self, anchor_indecies, scale=None):
        '''
        Returns
            Wavelet functions of indicated scale using the analytical form.
            L/(L+I/s)^2
        Parameters:
            anchor_indicies: nodes at which to return wavelets
            scale: scale of the wavelet
        '''
        
        F = self.factor
        L = self.L
        

        # Get the default/reference scale
        base_scale = self.S[0]

        # Use default scale if none given
        if scale is None:
            scale = base_scale

        # Create the LOCALIZATION VECTOR
        # Number of Rows = Number of true verticies
        # Number of Cols = Number of Reduced vertices
        nLocal = len(anchor_indecies)
        anchors = np.zeros((L.shape[0], nLocal))

        for i, node_idx in enumerate(anchor_indecies):
            anchors[node_idx, i] = 1

        # Analytical solution to scaling function
        F.cholesky_inplace(L, 1/scale)
        S = F(anchors)/scale

        # Solve again
        S = F(S)

        # multiply by laplacian
        S = L@S

        return S
    
    def analytical_wavelet_coeffs(self, f, scales=[1]):
        '''
        Returns
            Wavelet functions of indicated scale using the analytical form.
            L/(L+I/s)^2
        Parameters:
            f: Function to transform (can be 2d for multiple vectors)
            scale: scale of the wavelet
        '''
        
        W = self.allocate(f, len(scales))
        F = self.factor
        L = self.L

        # TODO Divide by scale
        # could do a yeild so the user can handle it real-time. 
        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale
            F.cholesky_inplace(L, 1/scale)

            # Step 2 -> First Sovle (Scaling coeffs!)
            S = F(f)

            # Step 3 -> Second Solve and Laplacian product
            W[:,:,i] = L@F(S) #/scale 


        return W