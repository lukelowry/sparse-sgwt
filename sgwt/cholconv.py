# -*- coding: utf-8 -*-
"""Graph Convolution Solvers for Sparse Spectral Graph Wavelet Transform (SGWT).

This module provides analytical and Vector Fitting methods for Graph Signal Processing (GSP)
and Spectral Graph Wavelet Transform (SGWT) convolution operations. It includes:
- `Convolve`: For graphs with constant topology (static).
- `DyConvolve`: For graphs with evolving topologies, using efficient rank-1 updates.

Both are designed for high-performance operations leveraging CHOLMOD.

Author: Luke Lowery (lukel@tamu.edu)
"""

from .cholesky import CholWrapper, cholmod_dense, cholmod_sparse
from .util import VFKernel

import numpy as np
from scipy.sparse import csc_matrix # type: ignore

from ctypes import byref, POINTER
from typing import Union, Optional, Type, List
from types import TracebackType
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

        # Store number of vertices
        self.n_vertices = L.shape[0]
        
        # Handles symb factor when entering context
        self.chol = CholWrapper(L)

    
    def __enter__(self) -> "Convolve":

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

    def __call__(self, B: np.ndarray, K: Union[VFKernel, dict]) -> np.ndarray:  # pragma: no cover
        return self.convolve(B, K) 
    
    def convolve(self, B: np.ndarray, K: Union[VFKernel, dict]) -> np.ndarray:
        """
        Performs graph convolution using a specified kernel.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps) with column-major ordering (F).
        K : VFKernel | dict
            Kernel function (Vector Fitting model) to apply.

        Returns
        -------
        np.ndarray
            Convolved signal (n_vertices, n_timesteps, nDim).
        """
        # 1. Input validation and conversion before heavy lifting
        if isinstance(K, dict):
            K = VFKernel.from_dict(K)

        if not isinstance(K, VFKernel):
            raise TypeError("Kernel K must be a VFKernel object or a compatible dictionary.")

        if K.R is None or K.Q is None:
            raise ValueError("Kernel K must contain residues (R) and poles (Q).")

        # Validate B and convert to cholmod format early
        if not B.flags['F_CONTIGUOUS']:  # pragma: no cover
            B = np.asfortranarray(B)
        B_chol_struct = self.chol.numpy_to_chol_dense(B)
        B_chol = byref(B_chol_struct)

        # List, malloc, numpy, etc.
        nDim = K.R.shape[1]
        X1, Xset = self.X1, self.Xset
        Y, E   = self.Y, self.E

        # Initialize result with direct term if it exists
        W = np.zeros((*B.shape, nDim))
        if K.D.size > 0:
            W += B[..., None] * K.D

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
    
    def lowpass(self, B: np.ndarray, scales: List[float] = [1], Bset: Optional[csc_matrix] = None, refactor: bool = True, order = 1) -> List[np.ndarray]:
        """
        Computes low-pass filtered scaling coefficients at specified scales.

        Uses the analytical form: I / (sL + I).

        Warning: orders > 1 may cause issues if Bset is passed, untested

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        scales : list[float], default: [1]
            List of scales to compute coefficients for.
        Bset : csc_matrix, optional
            Sparse indicator vector for localized coefficient computation.
        refactor : bool, default: True
            Whether to perform numeric factorization for each scale.

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each scale.
        """

        # Pointer to b (The function being convolved)
        if not B.flags['F_CONTIGUOUS']:  # pragma: no cover
            B = np.asfortranarray(B)


        # Using this requires the number of columns in f to be 1
        if Bset is not None:  # pragma: no cover
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(Bset))

        # List, malloc, numpy, etc.
        W = []
        X1 = self.X1
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        B_chol_struct = self.chol.numpy_to_chol_dense(B)
        A_ptr         = byref(self.chol.A)
        fact_ptr      = self.chol.fact_ptr


        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Numeric Factorization 
            # In some instances it will alreayd be factord at appropriate scale, so we allow option to skip
            if refactor:
                self.chol.num_factor(A_ptr, fact_ptr, 1/scale)

            # Only relevant for order > 1
            in_ptr = byref(B_chol_struct)

            # Solve more than once iff order > 1
            for _ in range(order):
            
                # Step 2 -> Solve Linear System (A + beta*I) X1 = B
                self.chol.solve2(fact_ptr, in_ptr,  Bset, X1, Xset, Y, E) 

                # Step 3 ->  Divide by scale  X1 = X1/scale (A bit pointless to pass A but need to pass something)
                self.chol.sdmult(A_ptr, X1,  X1, 0.0,  1/scale)

                in_ptr = X1

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X1)
            )
        return W

    def bandpass(self, B: np.ndarray, scales: List[float] = [1], order: int = 1) -> List[np.ndarray]:
        """
        Computes band-pass filtered wavelet coefficients at specified scales.

        Uses the analytical form: ((4/s) * L / (L + I/s)^2)^order.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        scales : list[float], default: [1]
            List of scales to compute coefficients for.
        order : int, default: 1
            The order of the filter (number of times the operator is applied).

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each scale.
        """

        
        if not B.flags['F_CONTIGUOUS']:  # pragma: no cover
            B = np.asfortranarray(B)

        
        # Pointer to bB (The function being convolved)
        B_chol_struct = self.chol.numpy_to_chol_dense(B)

        # List, malloc, numpy, etc.
        W        = []
        X1, X2   = self.X1, self.X2 
        Xset     = self.Xset
        Y, E     = self.Y, self.E
        A_ptr    = byref(self.chol.A)
        fact_ptr = self.chol.fact_ptr

        # Calculate Scaling Coefficients of 'f' for each scale
        for i, scale in enumerate(scales):

            # Step 1 -> Numeric Factorization
            self.chol.num_factor(A_ptr, fact_ptr, 1/scale)
            
            # Solve more than once iff order > 1
            in_ptr = byref(B_chol_struct)
            for _ in range(order):
                
                # Step 2 -> Solve Linear System (A + beta*I)^2 x = in_ptr
                self.chol.solve2(fact_ptr, in_ptr, None, X2, Xset, Y, E) 
                self.chol.solve2(fact_ptr, X2, None, X1, Xset, Y, E) 

                # Step 3 ->  Laplacian multiply and scalar normalization 
                self.chol.sdmult(
                    A_ptr = A_ptr,
                    X_ptr = X1, 
                    Y_ptr = X2,  
                    alpha = 4/scale, 
                    beta  = 0.0
                )
                in_ptr = X2

            W.append(
                self.chol.chol_dense_to_numpy(X2)
            )


        return W

    def highpass(self, B: np.ndarray, scales: List[float] = [1]) -> List[np.ndarray]:
        """
        Computes high-pass filtered coefficients at specified scales.

        Uses the analytical form: sL / (sL + I).

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        scales : list[float], default: [1]
            List of scales to compute coefficients for.

        Returns
        -------
        list[np.ndarray]
            Filtered signals for each scale.
        """
      
        # List, malloc, numpy, etc.
        W = []
        X1, X2 = self.X1, self.X2 
        Xset   = self.Xset
        Y, E   = self.Y, self.E

        # Pointer to b (The function being convolved)
        if not B.flags['F_CONTIGUOUS']:  # pragma: no cover
            B = np.asfortranarray(B)
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
                A_ptr = byref(self.chol.A),
                X_ptr = X1, 
                Y_ptr = X2
            )

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X2)
            )

        return W


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

    def __call__(self, B: np.ndarray) -> np.ndarray:  # pragma: no cover
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

        if self.R is None:  # pragma: no cover
            raise Exception("Cannot call without VFKernel Object")

        # List, malloc, numpy, etc.
        nDim = self.R.shape[1]
        X1, Xset = self.X1, self.Xset
        Y, E   = self.Y, self.E

        # Initialize with direct term if it exists
        W = np.zeros((*B.shape, nDim))
        if self.D.size > 0:  # pragma: no cover
            W += B[..., None] * self.D

        B_chol = byref(self.chol.numpy_to_chol_dense(B))
        
        for fact_ptr, r in zip(self.factors, self.R):
            # The benefit now is we never have to factor, just solve
            self.chol.solve2(fact_ptr, B_chol,  None, X1, Xset, Y, E) 
            # Before Residue
            Z = self.chol.chol_dense_to_numpy(X1)
            # Cross multiply with residual (SLOW)
            W += Z[:, :, None]*r  
        return W
    
    
    def lowpass(self, B: np.ndarray, Bset: Optional[csc_matrix] = None, order = 1) -> List[np.ndarray]:
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

        # Using this requires the number of columns in f to be 1
        if Bset is not None:  # pragma: no cover
            Bset = byref(self.chol.numpy_to_chol_sparse_vec(Bset))

        # Pointer to b (The function being convolved)
        A_ptr         = byref(self.chol.A)
        B_chol_struct = self.chol.numpy_to_chol_dense(B)


        # Calculate Scaling Coefficients of 'f' for each scale
        for q, fact_ptr in zip(self.poles, self.factors):

            in_ptr = byref(B_chol_struct)

            for _ in range(order):

                # Step 1 -> Solve Linear System (A + beta*I) X1 = B
                self.chol.solve2(fact_ptr, in_ptr,  Bset, X1, Xset, Y, E) 

                # Step 2 ->  Multiply by pole  X1 = X1 * q
                self.chol.sdmult(A_ptr, X1,  X1, 0.0,  q)

                in_ptr = X1

            # Save
            W.append(
                self.chol.chol_dense_to_numpy(X1)
            )
            

        return W
    
    def bandpass(self, B: np.ndarray, order: int = 1) -> List[np.ndarray]:
        """
        Computes band-pass filtered wavelet coefficients.

        Uses the analytical form: (4qL / (L + qI)^2)^order.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        order : int, default: 1
            The order of the filter (number of times the operator is applied).

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
        B_chol_struct = self.chol.numpy_to_chol_dense(B)
        A_ptr = byref(self.chol.A)

        # Calculate Scaling Coefficients of 'f' for each scale
        for q, fact_ptr in zip(self.poles, self.factors):
            
            in_ptr = byref(B_chol_struct)
            for _ in range(order):
                # Step 1 -> Solve Linear System (A + beta*I)^2 x = in_ptr
                self.chol.solve2(fact_ptr, in_ptr, None, X2, Xset, Y, E) 
                self.chol.solve2(fact_ptr, X2, None, X1, Xset, Y, E) 

                # Step 2 ->  Divide by scale for normalization
                self.chol.sdmult(
                    A_ptr = A_ptr,
                    X_ptr = X1, 
                    Y_ptr = X2,  
                    alpha = 4*q, 
                    beta  = 0.0
                )
                in_ptr = X2

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
                A_ptr = byref(self.chol.A),
                X_ptr = X1, 
                Y_ptr = X2
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

        # Validate node indices to prevent C-level errors
        if not (0 <= i < self.n_vertices and 0 <= j < self.n_vertices):
            return False

        # Validate weight to prevent math domain error from sqrt
        if w < 0:
            raise ValueError("math domain error: weight w must be non-negative.")

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