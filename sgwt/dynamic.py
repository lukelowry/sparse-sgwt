# -*- coding: utf-8 -*-
"""Dynamic Graph Convolution for Sparse Spectral Graph Wavelet Transform (SGWT).

This module provides GSP convolution methods specifically designed for dynamic graphs,
such as those experiencing line closures and opens. It leverages CHOLMOD's updown
routines for efficient rank-1 updates and requires pre-determined scales/poles.

Author: Luke Lowery (lukel@tamu.edu)
"""

from .cholesky import CholWrapper
from .cholesky import cholmod_dense, cholmod_sparse
from .util import VFKernel

import numpy as np
from scipy.sparse import csc_matrix # type: ignore

from ctypes import byref, POINTER
from typing import Any, Union, Optional, Type, List
from types import TracebackType


class DyConvolve:

    def __init__(self, L:csc_matrix, poles: Union[List[float], VFKernel]) -> None:
        """
        Initializes a dynamic convolution context.
        
        Optimized for graphs with evolving topologies where poles/scales remain constant.
        Uses CHOLMOD's updown routines for efficient rank-1 updates.

        Parameters 
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        poles : list[float] | VFKernel
            Predetermined set of poles (equivalent to 1/scale for analytical filters).
        """

        # Store number of vertices
        self.n_vertices = L.shape[0]
        
        # Handles symb factor when entering context
        self.chol = CholWrapper(L)

        # If VF model given
        if isinstance(poles, VFKernel): # type: ignore
            self.poles = poles.Q
            self.R = poles.R
            self.D = poles.D
        else:
            # Number of scales
            self.poles = poles 
            self.R = None
            self.D = np.array([])
        
        self.npoles = len(self.poles)


    # Context Manager for using CHOLMOD
    def __enter__(self) -> "DyConvolve":

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

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:

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

    def __call__(self, B: np.ndarray) -> np.ndarray:
        return self.convolve(B)

    def convolve(self, B: np.ndarray) -> np.ndarray:
        """
        Performs graph convolution using the pre-defined kernel.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps) with column-major ordering (F).

        Returns
        -------
        np.ndarray
            Convolved signal (n_vertices, n_timesteps, nDim).
        """

        if self.R is None:
            raise Exception("Cannot call without VFKernel Object")

        # List, malloc, numpy, etc.
        nDim = self.R.shape[1]
        X1, Xset = self.X1, self.Xset
        Y, E   = self.Y, self.E

        # Initialize with direct term if it exists
        W = np.zeros((*B.shape, nDim))
        if self.D.size > 0:
            W += self.D

        B_chol = byref(self.chol.numpy_to_chol_dense(B))
        
        for fact_ptr, r in zip(self.factors, self.R):
            # The benefit now is we never have to factor, just solve
            self.chol.solve2(fact_ptr, B_chol,  None, X1, Xset, Y, E) 
            # Before Residue
            Z = self.chol.chol_dense_to_numpy(X1)
            # Cross multiply with residual (SLOW)
            W += Z[:, :, None]*r  
        return W
    
    
    def lowpass(self, B: np.ndarray, Bset: Optional[csc_matrix] = None) -> List[np.ndarray]:
        """
        Computes low-pass filtered scaling coefficients.
        
        Uses the analytical form: qI / (L + qI).

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        Bset : csc_matrix, optional
            Sparse indicator vector for localized coefficient computation.

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each pre-defined pole.
        """

        # List, malloc, numpy, etc.
        W = []
        X1    = self.X1
        Xset  = self.Xset
        Y, E  = self.Y, self.E

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
    
    def bandpass(self, B: np.ndarray) -> List[np.ndarray]:
        """
        Computes band-pass filtered wavelet coefficients.

        Uses the analytical form: 4qL / (L + qI)^2.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each pre-defined pole.
        """

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

    def highpass(self, B: np.ndarray) -> List[np.ndarray]:
        """
        Computes high-pass filtered coefficients.

        Uses the analytical form: L / (L + qI).

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each pre-defined pole.
        """
      
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
    
    def addbranch(self, i: int, j: int, w: float) -> bool:
        """
        Adds a branch to the graph topology and updates all factorizations.

        Uses CHOLMOD's updown routines for efficient rank-1 updates.

        Parameters
        ----------
        i : int
            Index of Vertex A.
        j : int
            Index of Vertex B.
        w : float
            Edge weight.
        """

        ok = True

        # Make sparse version of the single line lap
        ws = np.sqrt(w)
        data    = [ws, -ws]
        bus_ind = [i ,  j ] # Row Indicies
        br_ind  = [0 ,  0 ] # Col Indicies

        # Creates Sparse Incidence Matrix of added branch, must free later
        Cptr = self.chol.triplet_to_chol_sparse(
            nrow=self.n_vertices,
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
    
