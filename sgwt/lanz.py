# -*- coding: utf-8 -*-
"""Lanczos Graph Convolution for Sparse Spectral Graph Wavelet Transform (SGWT).

This module provides Lanczos method approximations for Graph Signal
Processing (GSP) convolution operations. Unlike polynomial methods, Lanczos
is signal-dependent, tailoring the Krylov subspace to the input.

Author: Luke Lowery (lukel@tamu.edu)
"""

import numpy as np
from ctypes import byref, c_void_p, cast
from scipy.sparse import csc_matrix
from scipy.linalg import eigh_tridiagonal
from typing import Callable, Optional, Type
from types import TracebackType
from .cholesky import CholWrapper


class LanzConvolve:
    """
    Implements graph convolution using the Lanczos method.

    The Lanczos algorithm provides a powerful way to compute the action of a
    matrix function on a vector, `f(L)x`, without diagonalizing the full
    matrix `L`. It is particularly effective for large, sparse matrices.

    The method constructs a problem-specific orthonormal basis for the Krylov
    subspace `K(L, x)`. In this basis, the action of `L` is represented by a
    small tridiagonal matrix `T`. The original problem is then approximated
    by computing `f(T)`, which is computationally inexpensive.

    Unlike the Chebyshev method, which creates a global polynomial approximation
    for `f`, the Lanczos method is signal-dependent, tailoring the approximation
    to the specific input signal `x`.
    """

    def __init__(self, L: csc_matrix):
        """
        Initializes a Lanczos convolution context.

        Parameters
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        """
        self.L = L
        self.L.sort_indices() # Preprocess: Ensure optimal sparsity pattern for CHOLMOD
        self.n_vertices = L.shape[0]
        self.chol = CholWrapper(L)
        self._q_desc = None
        self._r_desc = None

    def __enter__(self) -> "LanzConvolve":
        self.chol.start()
        # Preprocess: Persistent descriptors to avoid struct creation in loops
        # We initialize them with dummy data; pointers are updated during decomposition
        dummy = np.zeros((self.n_vertices, 1), order='F')
        self._q_desc = self.chol.numpy_to_chol_dense(dummy)
        self._r_desc = self.chol.numpy_to_chol_dense(dummy)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        self.chol.finish()
        self._q_desc = self._r_desc = None

    def _decompose(self, b: np.ndarray, order: int, Q: np.ndarray, r_vec: np.ndarray):
        """
        Performs Lanczos decomposition to build a Krylov subspace.

        Returns
        -------
        tuple
            (alphas, betas, Q_view, norm_b)
        """
        norm_b = np.linalg.norm(b)
        if norm_b < 1e-15: return None, None, None, 0.0

        Q[:, 0] = b / norm_b
        n = self.n_vertices
        ptr_base = Q.ctypes.data
        alphas, betas = [], []
        
        A_ptr = byref(self.chol.A)
        r_desc_ptr = byref(self._r_desc)
        q_desc_ptr = byref(self._q_desc)
        r = r_vec[:, 0] 

        # Initial product: r = L @ q
        self._q_desc.x = cast(ptr_base, c_void_p)
        self._r_desc.x = cast(r_vec.ctypes.data, c_void_p)
        self.chol.sdmult(A_ptr, q_desc_ptr, r_desc_ptr, 1.0, 0.0)

        for k in range(order):
            q = Q[:, k]
            alpha = np.dot(q, r)
            alphas.append(alpha)
        
            r -= alpha * q
            
            if k == order - 1: break
            
            beta = np.linalg.norm(r)
            if beta < 1e-12: break
            betas.append(beta)
            
            Q[:, k+1] = r / beta
            q_next = Q[:, k+1]
            
            # Full Re-orthogonalization (Vectorized BLAS via NumPy)
            q_next -= Q[:, :k+1] @ (Q[:, :k+1].T @ q_next) 
            q_norm = np.linalg.norm(q_next)
            if q_norm < 1e-12: break
            q_next /= q_norm

            # Next r = L @ q_next - beta * q
            r_vec[:, 0] = -beta * q
            self._q_desc.x = cast(ptr_base + (k+1) * n * 8, c_void_p) # Direct pointer arithmetic
            self.chol.sdmult(A_ptr, q_desc_ptr, r_desc_ptr, 1.0, 1.0)

        return np.array(alphas), np.array(betas), Q[:, :len(alphas)], norm_b

    def ritz_values(self, b: np.ndarray, order: int) -> np.ndarray:
        """
        Computes the Ritz values (approximate eigenvalues) for a specific signal.

        Parameters
        ----------
        b : np.ndarray
            Input signal vector.
        order : int
            Dimension of the Krylov subspace.
        """
        Q = np.zeros((self.n_vertices, order), order='F')
        r_vec = np.zeros((self.n_vertices, 1), order='F')
        
        # Ensure context is active for Ritz calculation
        with self:
            alphas, betas, _, _ = self._decompose(b.flatten(), order, Q, r_vec)

        if alphas is None: return np.array([])
        return eigh_tridiagonal(alphas, betas, eigvals_only=True)

    def convolve(self, B: np.ndarray, f: Callable, order: int) -> np.ndarray:
        """
        Performs graph convolution f(L)B using the Lanczos method.

        Parameters
        ----------
        B : np.ndarray
            Input signal array of shape (n_vertices, n_signals).
        f : Callable[[np.ndarray], np.ndarray]
            A vectorized function to apply to the graph spectrum.
        order : int
            The number of Lanczos iterations (dimension of the Krylov subspace).

        Returns
        -------
        np.ndarray
            The convolved signal array of shape (n_vertices, n_signals, n_dims).
        """
        B = B[:, np.newaxis] if B.ndim == 1 else B
        n_signals = B.shape[1]
        
        sample_f = f(np.array([1.0]))
        n_dims = sample_f.shape[1] if sample_f.ndim > 1 else 1
        W = np.zeros((self.n_vertices, n_signals, n_dims))

        # Pre-allocate buffers to reuse across signals
        Q_buf = np.zeros((self.n_vertices, order), order='F')
        r_buf = np.zeros((self.n_vertices, 1), order='F')

        for i in range(n_signals):
            alphas, betas, Q, norm_b = self._decompose(B[:, i], order, Q_buf, r_buf)
            if alphas is None: continue

            eigvals, eigvecs = eigh_tridiagonal(alphas, betas)
            f_eigvals = f(eigvals)
            if f_eigvals.ndim == 1: f_eigvals = f_eigvals[:, np.newaxis]
            
            # Project back: Q * V * f(Lambda) * V^T * e1
            y = eigvecs @ (f_eigvals * (eigvecs[0, :] * norm_b)[:, np.newaxis])
            W[:, i, :] = Q @ y

        return W if n_dims > 1 else W.squeeze(axis=2)