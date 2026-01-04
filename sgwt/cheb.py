# -*- coding: utf-8 -*-
"""Chebyshev Graph Convolution for Sparse Spectral Graph Wavelet Transform (SGWT).

This module provides Chebyshev polynomial approximation methods for Graph Signal 
Processing (GSP) convolution operations.

Author: Luke Lowery (lukel@tamu.edu)
"""

from .cholesky import CholWrapper
from .util import ChebyKernel

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
from ctypes import byref

class ChebConvolve:
    def __init__(self, L: csc_matrix) -> None:
        """
        Initializes a Chebyshev convolution context.
        
        Parameters
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        """
        self.n_vertices = L.shape[0]

        # Estimate spectral bound (lambda_max)
        e_max = eigs(L, k=1, which='LM', return_eigenvectors=False)
        self.spectrum_bound = float(e_max[0].real) * 1.01

        self.chol = CholWrapper(L)

    def __enter__(self) -> "ChebConvolve":
        self.chol.start()
        self.chol.sym_factor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.chol.free_factor(self.chol.fact_ptr)
        self.chol.finish()

    def _get_cheby_matrices(self, spectrum_bound: float):
        """Internal helper to prepare the identity and recurrence matrices."""
        EYE = self.chol.speye(self.n_vertices, self.n_vertices)
        M_ptr = self.chol.add(
            byref(self.chol.A),
            EYE,
            alpha=2.0 / spectrum_bound,
            beta=-1.0
        )
        return M_ptr, EYE

    def convolve(self, B: np.ndarray, C: ChebyKernel) -> np.ndarray:
        """Performs graph convolution using Chebyshev polynomial approximation."""
        n_vertex, n_signals = B.shape
        n_order, n_dim = C.C.shape

        if n_order == 0 or n_dim == 0:
            return np.zeros((n_vertex, n_signals, n_dim), dtype=np.float64)

        # Initialize result array in NumPy to handle multi-dimensional kernels efficiently
        # Shape: (n_vertex, n_signals, n_dim) to match static/dynamic convolve behavior
        W = np.zeros((n_vertex, n_signals, n_dim), dtype=np.float64)

        B_chol = byref(self.chol.numpy_to_chol_dense(B))
        M_ptr, EYE = self._get_cheby_matrices(C.spectrum_bound)

        # Pre-calculate row maximums to skip negligible updates
        abs_coeffs = np.abs(C.C)
        row_max = np.max(abs_coeffs, axis=1)

        # T0 = B
        T0_ptr = self.chol.copy_dense(B_chol)
        
        # Accumulate T0 contribution
        if row_max[0] > 1e-15:
            Z = self.chol.chol_dense_to_numpy(T0_ptr)
            W += Z[:, :, np.newaxis] * C.C[0, :]

        if n_order > 1:
            # T1 = M * T0
            T1_ptr = self.chol.allocate_dense(n_vertex, n_signals)
            self.chol.sdmult(M_ptr, T0_ptr, T1_ptr, alpha=1.0, beta=0.0)
            
            # Accumulate T1 contribution
            if row_max[1] > 1e-15:
                Z = self.chol.chol_dense_to_numpy(T1_ptr)
                W += Z[:, :, np.newaxis] * C.C[1, :]

            for k in range(2, n_order):
                # T_k = 2 * M * T_{k-1} - T_{k-2}
                T0_ptr, T1_ptr = T1_ptr, T0_ptr # Swap
                self.chol.sdmult(M_ptr, T0_ptr, T1_ptr, alpha=2.0, beta=-1.0)
                
                # Accumulate T_k contribution
                if row_max[k] > 1e-15:
                    Z = self.chol.chol_dense_to_numpy(T1_ptr)
                    W += Z[:, :, np.newaxis] * C.C[k, :]

        self.chol.free_dense(T0_ptr)
        if n_order > 1: self.chol.free_dense(T1_ptr)
        self.chol.free_sparse(EYE); self.chol.free_sparse(M_ptr)
        return W