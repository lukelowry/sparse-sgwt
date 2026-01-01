# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/static.py
Description: Analytical and Vector Fitting methods for GSP & SGWT Convolution 
             on static graphs (constant topology).
"""

from .cholesky import CholWrapper, cholmod_dense, cholmod_sparse
from .io import VFKern

import numpy as np
from scipy.sparse import csc_matrix

from ctypes import byref, POINTER
from typing import Any, Union


def impulse(lap, n=0, ntime=1):
    """
    Generates a Dirac impulse signal at a specified vertex.

    Parameters
    ----------
    lap : csc_matrix
        Graph Laplacian defining the node count.
    n : int
        Index of the vertex where the impulse is applied.
    ntime : int
        Number of time steps (columns) in the resulting signal.

    Returns
    -------
    np.ndarray
        A (N x ntime) array with 1.0 at index n and 0.0 elsewhere, in Fortran order.
    """
    b = np.zeros((lap.shape[0],ntime), order='F')
    b[n] = 1

    return b

class Convolve:

    def __init__(self, L:csc_matrix) -> None:
        """
        Initializes a static convolution context.
        
        Designed for high-performance GSP operations on graphs with constant topology.
        Manages CHOLMOD symbolic and numeric factorizations.

        Parameters
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        """

        # Store Number of nodes
        self.nBus = L.shape[0]
        
        # Handles symb factor when entering context
        self.chol = CholWrapper(L)

    
    def __enter__(self):
        # Start Cholmod
        self.chol.start()

        # Safe Symbolic Factorization
        self.chol.sym_factor()

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

        # Free working memory used in solve2
        self.chol.free_dense(self.X1)
        self.chol.free_dense(self.X2)
        self.chol.free_sparse(self.Xset)

        # Free Y & E (workspacce for solve2)
        self.chol.free_dense(self.Y)
        self.chol.free_dense(self.E)

        # Finish cholmod
        self.chol.finish()

    def __call__(self, B, K: Union[VFKern, dict]) -> Any:
        return self.convolve(B, K)

    def convolve(self, B, K: Union[VFKern, dict]):
        """
        Performs graph convolution using a specified kernel.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (N x T) with column-major ordering (F).
        K : VFKern or dict
            Kernel function (Vector Fitting model) to apply.

        Returns
        -------
        np.ndarray
            Convolved signal (N x T x nDim).
        """
        # 1. Input validation and conversion before heavy lifting
        if isinstance(K, dict):
            K = VFKern.from_dict(K)

        if not isinstance(K, VFKern):
            raise TypeError("Kernel K must be a VFKern object or a compatible dictionary.")

        if K.R is None or K.Q is None:
            raise ValueError("Kernel K must contain residues (R) and poles (Q).")

        # Validate B and convert to cholmod format early
        B_chol_struct = self.chol.numpy_to_chol_dense(B)
        B_chol = byref(B_chol_struct)

        # List, malloc, numpy, etc.
        nDim = K.R.shape[1]
        X1, Xset = self.X1, self.Xset
        Y, E   = self.Y, self.E

        # Initialize result with direct term if it exists
        W = np.zeros((*B.shape, nDim))
        if K.D.size > 0:
            W += K.D

        A_ptr = byref(self.chol.A)
        fact_ptr = self.chol.fact_ptr

        for q, r in zip(K.Q, K.R):

            # Step 1 -> Numeric Factorization
            self.chol.num_factor(A_ptr, fact_ptr, q)

            # Step 2 -> Solve Linear System (A + qI) X1 = B
            self.chol.solve2(fact_ptr, B_chol,  None, X1, Xset, Y, E) 

            # Before Residue
            Z = self.chol.chol_dense_to_numpy(X1)

            # Cross multiply with residual (SLOW)
            W += Z[:, :, None]*r  

        return W
    
    def lowpass(self, B, scales=[1], Bset=None, refactor=True):
        """
        Computes low-pass filtered scaling coefficients at specified scales.

        Uses the analytical form: I / (sL + I).

        Parameters
        ----------
        B : np.ndarray
            Input signal array (N x T).
        scales : list of float
            List of scales to compute coefficients for.
        Bset : csc_matrix, optional
            Sparse indicator vector for localized coefficient computation.
        refactor : bool
            Whether to perform numeric factorization for each scale.

        Returns
        -------
        list of np.ndarray
            Filtered signals for each scale.
        """

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

        
        A_ptr = byref(self.chol.A)
        fact_ptr = self.chol.fact_ptr


        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Numeric Factorization 
            # In some instances it will alreayd be factord at appropriate scale, so we allow option to skip
            if refactor:
                self.chol.num_factor(A_ptr, fact_ptr, 1/scale)
            
            # Step 2 -> Solve Linear System (A + beta*I) X1 = B
            self.chol.solve2(fact_ptr, B,  Bset, X1, Xset, Y, E) 

            # Step 3 ->  Divide by scale  X1 = X1/scale
            self.chol.sdmult(X1,  X1, 0.0,  1/scale)

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X1)
            )

        return W
    
    def bandpass(self, B, scales=[1]):
        """
        Computes band-pass filtered wavelet coefficients at specified scales.

        Uses the analytical form: (1/s) * L / (L + I/s)^2.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (N x T).
        scales : list of float
            List of scales to compute coefficients for.

        Returns
        -------
        list of np.ndarray
            Filtered signals for each scale.
        """

        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(B))

        
        A_ptr = byref(self.chol.A)
        fact_ptr = self.chol.fact_ptr

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Numeric Factorization
            self.chol.num_factor(A_ptr, fact_ptr, 1/scale)
            
            # Step 2 -> Solve Linear System (A + beta*I)^2 x = B
            self.chol.solve2(fact_ptr, B, None, X2, Xset, Y, E) 
            self.chol.solve2(fact_ptr, X2, None, X1, Xset, Y, E) 

            # Step 3 ->  Divide by scale for normalization
            self.chol.sdmult(
                matrix_ptr = X1, 
                out_ptr =X2,  
                alpha = 4/scale, 
                beta  = 0.0
            )

            W.append(
                self.chol.chol_dense_to_numpy(X2)
            )


        return W

    def highpass(self, B, scales=[1]):
        """
        Computes high-pass filtered coefficients at specified scales.

        Uses the analytical form: sL / (sL + I).

        Parameters
        ----------
        B : np.ndarray
            Input signal array (N x T).
        scales : list of float
            List of scales to compute coefficients for.

        Returns
        -------
        list of np.ndarray
            Filtered signals for each scale.
        """
      
        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        B    = byref(self.chol.numpy_to_chol_dense(B))

        A_ptr = byref(self.chol.A)
        fact_ptr = self.chol.fact_ptr

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Numeric Factorization
            self.chol.num_factor(A_ptr, fact_ptr, 1/scale)
            
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