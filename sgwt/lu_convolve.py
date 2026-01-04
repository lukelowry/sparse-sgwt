# -*- coding: utf-8 -*-
"""Static Graph Convolution using Sparse LU Decomposition.

This module provides an LU-based implementation for Graph Signal Processing (GSP)
operations, supporting complex shifts (poles) which are not natively supported
by Cholesky-based methods.
"""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu
from typing import Union, Optional, List, Type
from types import TracebackType
from .util import VFKernel

class LUConvolve:
    """
    Static Graph Convolution using Sparse LU Decomposition.
    
    This implementation supports complex shifts (poles), allowing for more general
    filter approximations (e.g., those obtained via Vector Fitting with complex poles).
    """

    def __init__(self, L: csc_matrix) -> None:
        """
        Initializes the LU convolution context.

        Parameters
        ----------
        L : csc_matrix
            Sparse Graph Laplacian.
        """
        if not isinstance(L, csc_matrix):
            L = csc_matrix(L)
        self.L = L
        self.n_vertices = L.shape[0]
        self.I = eye(self.n_vertices, format='csc')

    def __enter__(self) -> "LUConvolve":
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> None:
        pass

    def __call__(self, B: np.ndarray, K: Union[VFKernel, dict]) -> np.ndarray:
        return self.convolve(B, K)

    def convolve(self, B: np.ndarray, K: Union[VFKernel, dict]) -> np.ndarray:
        """
        Performs graph convolution using a specified kernel via LU decomposition.

        Parameters
        ----------
        B : np.ndarray
            Input signal array (n_vertices, n_timesteps).
        K : VFKernel | dict
            Kernel function (Vector Fitting model) to apply.

        Returns
        -------
        np.ndarray
            Convolved signal (n_vertices, n_timesteps, nDim).
        """
        if isinstance(K, dict):
            K = VFKernel.from_dict(K)

        if not isinstance(K, VFKernel):
            raise TypeError("Kernel K must be a VFKernel object or a compatible dictionary.")

        # Determine if we need complex arithmetic based on poles, residues, or input
        is_complex = np.iscomplexobj(K.Q) or np.iscomplexobj(K.R) or np.iscomplexobj(B)
        dtype = complex if is_complex else float

        nDim = K.R.shape[1]
        if B.ndim == 1:
            B = B[:, np.newaxis]
        n_vertices, n_timesteps = B.shape
        
        W = np.zeros((n_vertices, n_timesteps, nDim), dtype=dtype)
        
        if K.D is not None and K.D.size > 0:
            # Apply the direct term D, scaled by the input signal B.
            W += B[..., None] * K.D

        for q, r in zip(K.Q, K.R):
            # Solve (L + qI) X = B
            A_shifted = self.L + q * self.I
            solver = splu(A_shifted)
            X = solver.solve(B)
            
            # Accumulate: W += X * r
            W += X[..., None] * r

        return W

    def lowpass(self, B: np.ndarray, scales: List[float] = [1]) -> List[np.ndarray]:
        W = []
        for scale in scales:
            A_shifted = self.L + (1.0/scale) * self.I
            solver = splu(A_shifted)
            X = solver.solve(B)
            W.append(X / scale)
        return W

    def bandpass(self, B: np.ndarray, scales: List[float] = [1], order: int = 1) -> List[np.ndarray]:
        W = []
        for scale in scales:
            A_shifted = self.L + (1.0/scale) * self.I
            solver = splu(A_shifted)
            in_ptr = B
            for _ in range(order):
                x2 = solver.solve(in_ptr)
                x1 = solver.solve(x2)
                in_ptr = (4.0/scale) * (self.L @ x1)
            W.append(in_ptr)
        return W

    def highpass(self, B: np.ndarray, scales: List[float] = [1]) -> List[np.ndarray]:
        W = []
        for scale in scales:
            A_shifted = self.L + (1.0/scale) * self.I
            solver = splu(A_shifted)
            X1 = solver.solve(B)
            W.append(self.L @ X1)
        return W