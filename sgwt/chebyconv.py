# -*- coding: utf-8 -*-
"""Chebyshev Graph Convolution for Sparse Spectral Graph Wavelet Transform (SGWT).

This module provides Chebyshev polynomial approximation methods for Graph Signal 
Processing (GSP) convolution operations.

Author: Luke Lowery (lukel@tamu.edu)
"""

from .cholesky import CholWrapper
from .util import ChebyKernel, estimate_spectral_bound

import numpy as np
from scipy.sparse import csc_matrix
from ctypes import byref

class ChebyConvolve:
    def __init__(self, L: csc_matrix) -> None:
        """
        Initializes a Chebyshev convolution context.

        This context manager is used to perform graph convolutions via
        Chebyshev polynomial approximation. It estimates the spectral bound
        of the Laplacian upon initialization.

        Parameters
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        """
        self.n_vertices = L.shape[0]

        # Estimate spectral bound (lambda_max)
        self.spectrum_bound = estimate_spectral_bound(L)

        self.chol = CholWrapper(L)

    def __enter__(self) -> "ChebyConvolve":
        self.chol.start()
        self.chol.sym_factor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.chol.free_factor(self.chol.fact_ptr)
        self.chol.finish()

    def _get_cheby_recurrence_matrix(self, spectrum_bound: float):
        """Internal helper to prepare the recurrence matrix M = (2/lmax)L - I."""
        EYE = None
        try:
            EYE = self.chol.speye(self.n_vertices, self.n_vertices)
            M_ptr = self.chol.add(
                byref(self.chol.A),
                EYE,
                alpha=2.0 / spectrum_bound,
                beta=-1.0
            )
            return M_ptr
        finally:
            if EYE:
                self.chol.free_sparse(EYE)

    def convolve(self, B: np.ndarray, C: ChebyKernel) -> np.ndarray:
        """
        Performs graph convolution using Chebyshev polynomial approximation.

        This method implements Clenshaw's algorithm for the stable evaluation of
        the Chebyshev series on the graph signal `B`.

        Parameters
        ----------
        B : np.ndarray
            Input signal array of shape (n_vertices,) or (n_vertices, n_signals).
        C : ChebyKernel
            A `ChebyKernel` object containing the Chebyshev coefficients and
            the spectral bound of the approximation.

        Returns
        -------
        np.ndarray
            The convolved signal. The shape of the output is
            (n_vertices, n_signals, n_dims) for a 2D input `B`, or
            (n_vertices, n_dims) for a 1D input `B`. `n_dims` is the
            number of filter dimensions in the kernel.
        """
        input_was_1d = False
        if B.ndim == 1:
            B = B[:, np.newaxis]
            input_was_1d = True

        n_vertex, n_signals = B.shape
        n_order, n_dim = C.C.shape

        W = np.zeros((n_vertex, n_signals, n_dim), dtype=np.float64)

        if n_order == 0 or n_dim == 0:
            return W

        if not B.flags['F_CONTIGUOUS']:
            B = np.asfortranarray(B)

        B_chol = byref(self.chol.numpy_to_chol_dense(B))

        M_ptr, T_km2_ptr, T_km1_ptr = None, None, None

        try:
            # T_0(L)B = B (the identity)
            T_km2_ptr = self.chol.copy_dense(B_chol)
            Z = self.chol.chol_dense_to_numpy(T_km2_ptr)
            W += Z[:, :, np.newaxis] * C.C[0, :]

            if n_order > 1:
                M_ptr = self._get_cheby_recurrence_matrix(C.spectrum_bound)

                # T_1(L)B = M * T_0
                T_km1_ptr = self.chol.allocate_dense(n_vertex, n_signals)
                self.chol.sdmult(M_ptr, T_km2_ptr, T_km1_ptr, alpha=1.0, beta=0.0)
                Z = self.chol.chol_dense_to_numpy(T_km1_ptr)
                W += Z[:, :, np.newaxis] * C.C[1, :]

                # Clenshaw's algorithm for k >= 2
                for k in range(2, n_order):
                    # T_k = 2 * M * T_{k-1} - T_{k-2}.
                    # At loop start: T_km2_ptr holds T_{k-2}, T_km1_ptr holds T_{k-1}.
                    # We calculate T_k and store it by overwriting T_km2_ptr.
                    self.chol.sdmult(M_ptr, T_km1_ptr, T_km2_ptr, alpha=2.0, beta=-1.0)

                    # Swap pointers for the next iteration.
                    # T_km2_ptr now holds T_{k-1}, T_km1_ptr holds T_k.
                    T_km2_ptr, T_km1_ptr = T_km1_ptr, T_km2_ptr

                    # Accumulate the contribution from T_k (now in T_km1_ptr).
                    Z = self.chol.chol_dense_to_numpy(T_km1_ptr)
                    W += Z[:, :, np.newaxis] * C.C[k, :]

        finally:
            if T_km2_ptr: self.chol.free_dense(T_km2_ptr)
            if T_km1_ptr: self.chol.free_dense(T_km1_ptr)
            if M_ptr: self.chol.free_sparse(M_ptr)

        if input_was_1d:
            return W.squeeze(axis=1)
        return W