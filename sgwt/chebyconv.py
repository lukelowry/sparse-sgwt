# -*- coding: utf-8 -*-
"""Chebyshev Graph Convolution for Sparse Spectral Graph Wavelet Transform (SGWT).
Optimized version with improved computational performance.

Author: Luke Lowery (lukel@tamu.edu)
"""
from .cholesky import CholWrapper
from .util import ChebyKernel, estimate_spectral_bound
import numpy as np
from scipy.sparse import csc_matrix
from ctypes import byref


class ChebyConvolve:
    """
    Chebyshev polynomial graph convolution context.
    
    Approximates spectral graph filters using Chebyshev polynomials of the
    first kind via the recurrence relation.
    
    Parameters
    ----------
    L : csc_matrix
        Sparse Graph Laplacian.
    """
    
    def __init__(self, L: csc_matrix) -> None:
        self.n_vertices = L.shape[0]
        self.spectrum_bound = estimate_spectral_bound(L)
        self.chol = CholWrapper(L)
        self._cached_M_ptr = None
        self._cached_spectrum_bound = None
        
    def __enter__(self) -> "ChebyConvolve":
        self.chol.start()
        self.chol.sym_factor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cached_M_ptr is not None:
            self.chol.free_sparse(self._cached_M_ptr)
        self.chol.free_factor(self.chol.fact_ptr)
        self.chol.finish()
    
    def _get_recurrence_matrix(self, spectrum_bound: float, min_lambda: float = 0.0):
        """Get cached recurrence matrix M = alpha*L + beta*I for domain [min_lambda, spectrum_bound]."""
        cache_key = (spectrum_bound, min_lambda)
        if self._cached_M_ptr is not None:
            if self._cached_spectrum_bound == cache_key:
                return self._cached_M_ptr
            self.chol.free_sparse(self._cached_M_ptr)

        range_ = spectrum_bound - min_lambda
        alpha = 2.0 / range_
        beta = -(spectrum_bound + min_lambda) / range_

        EYE = self.chol.speye(self.n_vertices, self.n_vertices)
        try:
            self._cached_M_ptr = self.chol.add(
                byref(self.chol.A), EYE,
                alpha=alpha, beta=beta
            )
            self._cached_spectrum_bound = cache_key
            return self._cached_M_ptr
        finally:
            self.chol.free_sparse(EYE)
    
    def convolve(self, B: np.ndarray, C: ChebyKernel) -> np.ndarray:
        """
        Performs graph convolution using Chebyshev polynomial approximation.
        
        Parameters
        ----------
        B : np.ndarray
            Input signal of shape (n_vertices,) or (n_vertices, n_signals).
        C : ChebyKernel
            Chebyshev kernel with coefficients and spectral bound.
        
        Returns
        -------
        np.ndarray
            Convolved signal of shape (n_vertices, n_signals, n_dims) or
            (n_vertices, n_dims) for 1D input.
        """
        # Handle complex inputs by processing real and imaginary parts
        if np.iscomplexobj(B):
            input_1d = B.ndim == 1
            B = B[:, np.newaxis] if input_1d else B
            n_signals = B.shape[1]
            # Stack real and imaginary parts: [real_cols | imag_cols]
            B_stacked = np.column_stack([B.real, B.imag])
            if not B_stacked.flags['F_CONTIGUOUS']:
                B_stacked = np.asfortranarray(B_stacked)
            W = self._convolve_real(B_stacked, C)
            # Extract: first n_signals columns are real, next n_signals are imag
            W_complex = W[:, :n_signals, :] + 1j * W[:, n_signals:, :]
            return W_complex.squeeze(axis=1) if input_1d else W_complex
        
        input_1d = B.ndim == 1
        B = B[:, np.newaxis] if input_1d else B
        if not B.flags['F_CONTIGUOUS']:
            B = np.asfortranarray(B)
        
        W = self._convolve_real(B, C)
        return W.squeeze(axis=1) if input_1d else W
    
    def _convolve_real(self, B: np.ndarray, C: ChebyKernel) -> np.ndarray:
        """Core convolution for real-valued signals."""
        n_vertex, n_signals = B.shape
        n_order, n_dim = C.C.shape
        
        if n_order == 0 or n_dim == 0:
            return np.zeros((n_vertex, n_signals, n_dim), dtype=np.float64)
        
        # Pre-allocate output and get contiguous coefficient view
        W = np.zeros((n_vertex, n_signals, n_dim), dtype=np.float64)
        coeffs = np.ascontiguousarray(C.C)
        
        B_chol = byref(self.chol.numpy_to_chol_dense(B))
        T_km2, T_km1 = None, None
        
        try:
            # T_0(L)B = B
            T_km2 = self.chol.copy_dense(B_chol)
            self._accumulate(W, self.chol.chol_dense_to_numpy(T_km2), coeffs[0])
            
            if n_order == 1:
                return W
            
            M = self._get_recurrence_matrix(C.spectrum_bound, C.min_lambda)
            T_km1 = self.chol.allocate_dense(n_vertex, n_signals)
            
            # T_1(L)B = M @ B
            self.chol.sdmult(M, T_km2, T_km1, alpha=1.0, beta=0.0)
            self._accumulate(W, self.chol.chol_dense_to_numpy(T_km1), coeffs[1])
            
            # Recurrence: T_k = 2*M*T_{k-1} - T_{k-2}
            for k in range(2, n_order):
                self.chol.sdmult(M, T_km1, T_km2, alpha=2.0, beta=-1.0)
                T_km2, T_km1 = T_km1, T_km2
                self._accumulate(W, self.chol.chol_dense_to_numpy(T_km1), coeffs[k])
        finally:
            if T_km2: self.chol.free_dense(T_km2)
            if T_km1: self.chol.free_dense(T_km1)
        
        return W
    
    @staticmethod
    def _accumulate(W: np.ndarray, Z: np.ndarray, c: np.ndarray):
        """Accumulate W += Z[:,:,None] * c using optimal method."""
        n_dim = len(c)
        if n_dim <= 4:
            for d in range(n_dim):
                if c[d] != 0:
                    W[:, :, d] += Z * c[d]
        else:
            # einsum avoids large temporaries for many dimensions
            np.add(W, np.einsum('ij,k->ijk', Z, c, optimize=True), out=W)
    
    def convolve_multi(self, B: np.ndarray, kernels: list) -> list:
        """
        Apply multiple kernels efficiently by sharing Chebyshev term computation.
        
        Parameters
        ----------
        B : np.ndarray
            Input signal.
        kernels : list of ChebyKernel
            Kernels to apply.
        
        Returns
        -------
        list of np.ndarray
            Convolved signals for each kernel.
        """
        if not kernels:
            return []
        
        # Handle complex
        if np.iscomplexobj(B):
            real_res = self.convolve_multi(B.real, kernels)
            imag_res = self.convolve_multi(B.imag, kernels)
            return [r + 1j * i for r, i in zip(real_res, imag_res)]
        
        input_1d = B.ndim == 1
        B = B[:, np.newaxis] if input_1d else B
        if not B.flags['F_CONTIGUOUS']:
            B = np.asfortranarray(B)
        
        n_vertex, n_signals = B.shape
        
        # Group by (spectrum_bound, min_lambda)
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, k in enumerate(kernels):
            groups[(k.spectrum_bound, k.min_lambda)].append((idx, k))

        results = [None] * len(kernels)

        for (bound, min_lam), group in groups.items():
            max_order = max(k.C.shape[0] for _, k in group)

            # Stack all coefficients for this group for vectorized application
            # Compute terms and apply via tensor contraction
            T_terms = self._compute_terms(B, max_order, bound, min_lam)
            T_stack = np.stack(T_terms, axis=0)  # (max_order, n_vertex, n_signals)
            
            for idx, kernel in group:
                n_order, n_dim = kernel.C.shape
                # W[v,s,d] = sum_k T[k,v,s] * C[k,d]
                W = np.einsum('kvs,kd->vsd', T_stack[:n_order], kernel.C, optimize=True)
                results[idx] = W.squeeze(axis=1) if input_1d else W
        
        return results
    
    def _compute_terms(self, B: np.ndarray, max_order: int, bound: float, min_lambda: float = 0.0) -> list:
        """Compute Chebyshev terms T_k(L)B for k = 0..max_order-1."""
        n_vertex, n_signals = B.shape
        terms = []

        B_chol = byref(self.chol.numpy_to_chol_dense(B))
        T_km2, T_km1 = None, None

        try:
            T_km2 = self.chol.copy_dense(B_chol)
            terms.append(self.chol.chol_dense_to_numpy(T_km2).copy())

            if max_order > 1:
                M = self._get_recurrence_matrix(bound, min_lambda)
                T_km1 = self.chol.allocate_dense(n_vertex, n_signals)
                
                self.chol.sdmult(M, T_km2, T_km1, alpha=1.0, beta=0.0)
                terms.append(self.chol.chol_dense_to_numpy(T_km1).copy())
                
                for _ in range(2, max_order):
                    self.chol.sdmult(M, T_km1, T_km2, alpha=2.0, beta=-1.0)
                    T_km2, T_km1 = T_km1, T_km2
                    terms.append(self.chol.chol_dense_to_numpy(T_km1).copy())
        finally:
            if T_km2: self.chol.free_dense(T_km2)
            if T_km1: self.chol.free_dense(T_km1)
        
        return terms