

from .cholesky import CholWrapper

from scipy.sparse import csc_matrix
import numpy as np

class FastSGWT2:
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


        # NOTE improve how this is used so I don't have to do this
        self.chol = CholWrapper(L)

        # Pre-Factor (Symbolic)
        self.chol.start()

        # Factor Symbolically
        self.chol.sym_factor()

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

    def scaling_coeffs(self, f, scales=None):
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
        C = self.chol
        L = self.L

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale
            C.num_factor(1/scale)


            # Step 2 -> Solve and Divide by squared scale for normalization
            W[:,:,i] = C.solve(f)/scale 

        return W
    
    # NOTE only function with new implementation so far
    def wavelet_coeffs(self, f, scales=None):
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
        C = self.chol
        L = self.L

        for i, scale in enumerate(scales):

            # Step 1 -> Set Scale via shifted numeric factor
            C.num_factor(1/scale)

            # Step 2 -> First Sovle
            S = C.solve(f) 

            # Step 3 -> Second Solve and Laplacian product
            W[:,i] = L@C.solve(S)/scale 

            # TODO support higher dimensions
            #W[:,:,i] = L@C.solve(S)/scale 

        return W
    
    def highpass_coeffs(self, f, scales=None):
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