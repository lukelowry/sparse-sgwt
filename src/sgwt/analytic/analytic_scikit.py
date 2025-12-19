

from .analytic import AnalyticFilters


from sksparse.cholmod import analyze
from scipy.sparse import csc_matrix

import numpy as np

class FiltersScikit(AnalyticFilters):
    '''
    Description: 
        A class that of analytical versions filters for SGWT and GSP
    Parameters:
        L: sparse csc_matrix form of Graph Laplacian (real valued)
        scales: optional, default scales used
    '''

    def __init__(self, L: csc_matrix, scales=[1]):

        # Sparse Laplacian
        self.L = L

        # Discrete Scales
        self.setscales(scales)

        # Pre-Factor (Symbolic)
        self.factor = analyze(L)

    def __call__(self, f):
        '''
        Description
            Conveniece call function that computes SGWT coefficients
        Returns
            W:  Array size (Bus, Time, Scale)
        '''
        
        return self.wavelet_coeffs(f)

    def setscales(self, scales):
        self.scales = scales
        self.nscales = len(scales)

    def allocate(self, f, n=None):
        if n is None:
            return np.zeros((*f.shape, self.nscales))
        else:
            return np.zeros((*f.shape, n))

    
    '''
    Local Impulse Responses
    '''

    def scaling_funcs(self, anchor_indecies, scale):
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
        S = F(anchors)/scale

        return S
    
    def wavelet_funcs(self, anchor_indecies, scale):
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

        # Create the LOCALIZATION VECTOR
        # Number of Rows = Number of true verticies
        # Number of Cols = Number of Reduced vertices
        nLocal = len(anchor_indecies)
        anchors = np.zeros((L.shape[0], nLocal))

        for i, node_idx in enumerate(anchor_indecies):
            anchors[node_idx, i] = 1

        # Analytical solution to scaling function
        F.cholesky_inplace(L, 1/scale)

        # Solve
        S = F(anchors)
        S = L@F(S)/scale

        return S
    
    '''
    Convolutions
    '''

    def scaling_coeffs(self, f, fset=None, scales=None):
        '''
        Description
            Scaling coefficnets at indicated scales using the analytical form
            I/(aL+I)
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate scaling coeffs.
            scales: list (numScales) of scales to compute scaling coefficents for.
        Returns
            Scaling coefficients for each scale (numVerticies x numScales)
        '''
        
        scales = self.scales if scales is None else scales
        W = self.allocate(f, len(scales))
        F = self.factor
        L = self.L

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale
            F.cholesky_inplace(L, 1/scale)

            # Step 2 -> Solve and Divide by squared scale for normalization
            W[:,:,i] = F(f)/scale 

        return W
    
    
    def _allocate_results(self):
        pass
    
    def _format_rhs(self, b, bset):
        return b, bset
    
    def _save_to_results(self, x, index):
        pass
    
    def _numeric_factorization(self, beta):
        pass

    def _solve(self, b, bset):
        pass

    def _solve_twice(self, b, bset):
        pass

    def _mult(self, x, scalar):
        pass

    def _mult_lap(self, x, scalar):
        pass

    
    
    def wavelet_coeffs(self, f, fset=None, scales=None):
        '''
        Returns
            Wavelet functions of indicated scale using the analytical form.
            (1/s)  L/(L+I/s)^2
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate wavelet coeffs.
            scales: list (numScales) of scales to compute wavelet coefficents for.
        Returns
            Wavelet coefficients for each scale (numVerticies x numScales)
        '''
        
        scales = self.scales if scales is None else scales
        W = self.allocate(f,len(scales))
        F = self.factor
        L = self.L

        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale
            F.cholesky_inplace(L, 1/scale)

            # Step 2 -> First Sovle (Scaling coeffs!)
            S = F(f)

            # Step 3 -> Second Solve and Laplacian product
            W[:,:,i] = L@F(S)*(4/scale)

        return W
    
    def highpass_coeffs(self, f, fset=None, scales=None):
        '''
        Description
            Scaling coefficnets at indicated scales using the analytical form
            aL/(aL+I)
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate HP coeffs.
            scales: list (numScales) of scales to compute  HP coefficents for.
        Returns
            High-pass coefficients for each scale (numVerticies x numScales)
        '''
        
        scales = self.scales if scales is None else scales
        W = self.allocate(f, len(scales))
        F = self.factor
        L = self.L

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale
            F.cholesky_inplace(L, 1/scale)

            # Step 2 -> Solve and Divide by squared scale for normalization
            W[:,:,i] = self.L@F(f)


        return W
    
    '''
    Inverse Transformations
    '''
    
    def wavelet_inv(self, W):
        '''
        Description
            The inverse SGWT transformation (only one time point for now)
            And does not support scaling coefficients right now.
        Parameters
            W: ndarray of shape (Bus x Times x Scales)
        Return
            f: reconstructed signal

        WARNING: TODO the reconstructed signal is not normalized
        '''

        F = self.factor
        L = self.L

        # Allocate reconstructed vector (nBus x nFeature)
        f = np.zeros((W.shape[0],W.shape[1]))

        for i, scale in enumerate(self.scales):

            # Coefficients of this scale
            WS = W[:,:,i]

            # Step 1 -> Set Scale
            F.cholesky_inplace(L, 1/scale)

            # Step 2 -> First Sovle (Scaling coeffs!)
            S = F(WS)

            # Step 3 -> Second Solve and Laplacian product
            f += L@F(S)/scale 

        return f