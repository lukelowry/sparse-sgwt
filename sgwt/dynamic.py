"""
dynamic.py

GSP Convolution designed *specifically* for dynamic graphs (i.e. line closures and opens)

For scalable implementation, this approach requires the set of scales/poles to be pre-determined

Author: Luke Lowery (lukel@tamu.edu)
"""

from .cholesky import CholWrapper
from .cholesky import cholmod_dense, cholmod_sparse

from .ration import VFKern

import numpy as np
from scipy.sparse import csc_matrix

from ctypes import byref, POINTER
from typing import Any


class DyConvolve:

    def __init__(self, L:csc_matrix, poles, K:VFKern = None) -> None:
        '''
        Description
            A variant of Convolve except the implementation
            is optimized to handle updown calls to add branches.
            The trade-off is that the poles/scales are constant.
        Parameters 
            L: Sparse Graph Laplacian
            poles: predetermined set of poles (equiv to 1/scale)
            K: Please set poles=None if passing kern.
        '''

        # Store Number of nodes
        self.nBus = L.shape[0]
        
        # Handles symb factor when entering context
        self.chol = CholWrapper(L)

        # If VF model given
        if K is not None:
            self.K = K
            self.poles = K.Q
            self.npoles = len(K.Q)
        else:
            # Number of scales
            self.poles = poles 
            self.npoles = len(poles)


    # Context Manager for using CHOLMOD
    def __enter__(self):

        # Start Cholmod
        self.chol.start()

        # Safe Symbolic Factorization
        self.chol.sym_factor()

        # Make copies of the symbolic factor object
        self.factors = [
            self.chol.copy_factor(self.chol.fact_ptr)
            for i in range(self.npoles)
        ]

        # Now perform each unique numeric factorization A + qI
        for q, fact_ptr in zip(self.poles, self.factors):
            self.chol.num_factor(byref(self.chol.A), fact_ptr, q)

        # Workspace for operations in solve2
        self.X1    = POINTER(cholmod_dense)()
        self.X2    = POINTER(cholmod_dense)()
        self.Xset  = POINTER(cholmod_sparse)()

        # Provide solve2 with re-usable workspace
        self.Y    = POINTER(cholmod_dense)()
        self.E    = POINTER(cholmod_dense)()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # Free the factored matrix object
        self.chol.free_factor(self.chol.fact_ptr)

        # Free the auxillary factor copies
        for fact_ptr in self.factors:
            self.chol.free_factor(fact_ptr)

        # Free working memory used in solve2
        self.chol.free_dense(self.X1)
        self.chol.free_dense(self.X2)
        self.chol.free_sparse(self.Xset)

        # Free Y & E (workspacce for solve2)
        self.chol.free_dense(self.Y)
        self.chol.free_dense(self.E)


        # Finish cholmod
        self.chol.finish()


    def __call__(self, B, K: VFKern) -> Any:
        return self.convolve(B, self.K)

    def convolve(self, B, K: VFKern):
        '''
        Description
            This versatile function can perform many convolutions,
            either with a single function (i.e., smoothing) or for
            a whole transformation (Compute the SGWT)
        Parameters
            X: 2D Array (nVertex, nTime) with column major ordering (F)
            K: Kernel function to generate convolution
        '''

        # List, malloc, numpy, etc.
        nDim = K.R.shape[1]
        X1, Xset = self.X1, self.Xset
        Y, E   = self.Y, self.E

        W = np.zeros((*B.shape, nDim))
        B  = byref(self.chol.numpy_to_chol_dense(B))

        
        A_ptr = byref(self.chol.A)
   

        
        for fact_ptr, q, r in zip(self.factors, K.Q, K.R):

            # The benefit now is we never have to factor, just solve
            self.chol.solve2(fact_ptr, B,  None, X1, Xset, Y, E) 

            # Before Residue
            Z = self.chol.chol_dense_to_numpy(X1)

            # Cross multiply with residual (SLOW)
            W += Z[:, :, None]*r  

        # TODO add K.D per dimension

        return W
    
    
    def lowpass(self, B, Bset=None):
        '''
        Description
            Scaling coefficnets at indicated scales using the analytical form
            I/(aL+I) = qI/(L+qI)
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate scaling coeffs.
            fset: Used to solve for a sparse subset of coeffs. ncol must be 1
            scales: list (numScales) of scales to compute scaling coefficents for.
        Returns
            Scaling coefficients for each scale (numVerticies x numScales)
        '''

        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(B))

        # Using this requires the number of columns in f to be 1
        if Bset is not None:
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(Bset))

        # Calculate Scaling Coefficients of 'f' for each scale
        for q, fact_ptr in zip(self.poles, self.factors):

            # Step 1 -> Solve Linear System (A + beta*I) X1 = B
            self.chol.solve2(fact_ptr, B,  Bset, X1, Xset, Y, E) 

            # Step 2 ->  Multiply by pole  X1 = X1 * q
            self.chol.sdmult(X1,  X1, 0.0,  q)

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X1)
            )

        return W
    
    def bandpass(self, B):
        '''
        Description
            Wavelet  coeffs of indicated scale using the analytical form.
            4qL/(L+qI)^2  located only at a subset of buses
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate wavelet coeffs.
        Returns
            Wavelet coefficients for each scale (numVerticies x numScales)
            Solved accurately only for buses indicated by fset
        '''

        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(B))
        fact_ptr = self.chol.fact_ptr

        # Calculate Scaling Coefficients of 'f' for each scale
        for q, fact_ptr in zip(self.poles, self.factors):

            # Step 1 -> Solve Linear System (A + beta*I)^2 x = B
            self.chol.solve2(fact_ptr, B, None, X2, Xset, Y, E) 
            self.chol.solve2(fact_ptr, X2, None, X1, Xset, Y, E) 

            # Step 2 ->  Divide by scale for normalization
            self.chol.sdmult(
                matrix_ptr = X1, 
                out_ptr =X2,  
                alpha = 4*q, 
                beta  = 0.0
            )

            W.append(
                self.chol.chol_dense_to_numpy(X2)
            )


        return W

    def highpass(self, B):
        '''
        Description
            High-pass coefficnets at indicated scales using the analytical form
            L/(L+qI). Bset parameter not defined for HP filter
        Parameters
            f: Signal array (numVerticies x numFeatures) to calculate HP coeffs.
        Returns
            High-pass coefficients for each scale (numVerticies x numScales)
        '''
      
        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(B))

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, fact_ptr in enumerate(self.factors):

            # Need to ensure X2 Initialized
            if i==0:
                self.chol.solve2(fact_ptr, B, None, X2, Xset, Y, E) 

            # Step 2 -> Solve Linear System (L + I/scale) x = B
            self.chol.solve2(fact_ptr, B, None, X1, Xset, Y, E) 

            # Step 3 ->  X2 = L@X1
            self.chol.sdmult(
                matrix_ptr = X1, 
                out_ptr = X2,  
                alpha = 1.0, 
                beta  = 0.0
            )

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X2)
            )

        return W
    
    def addbranch(self, i, j, w):
        '''
        Description
            Adds a branch via cholmod_updown
        Parameters
            i: Index of Vertex A
            j: Index of Vertex B
            w: Edge Weight
        '''

        ok = True

        # Make sparse version of the single line lap
        ws = np.sqrt(w)
        data    = [ws, -ws]
        bus_ind = [i ,  j ] # Row Indicies
        br_ind  = [0 ,  0 ] # Col Indicies

        # Creates Sparse Incidence Matrix of added branch, must free later
        Cptr = self.chol.triplet_to_chol_sparse(
            nrow=self.nBus,
            ncol=1,
            rows=bus_ind,
            cols=br_ind,
            vals=data
        )

        # TODO we can optize performance eventually by 
        # splitting updown into symbolic and numeric, since symbolic same for all
        
        # Update all factors
        for fact_ptr in self.factors:
            ok = ok and self.chol.update(Cptr, fact_ptr)

        # Free Cptr now that it has been used
        self.chol.free_sparse(Cptr)

        # Add to the factorized graph
        return ok
    
