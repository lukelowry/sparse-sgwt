# -*- coding: utf-8 -*-
"""Static Graph Convolution using Sparse LU Decomposition.

This module provides an LU-based implementation for Graph Signal Processing (GSP)
operations, supporting complex shifts (poles) which are not natively supported
by Cholesky-based methods.
"""

import numpy as np
from scipy.sparse import csc_matrix, eye, spmatrix
from scipy.sparse.linalg import splu
from typing import Union, Optional, List, Type
from types import TracebackType
from ctypes import byref
from .util import VFKernel
from .klu.wrapper import KluWrapper, KLU_OK
from .klu.structs import klu_common

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
        if not isinstance(L, spmatrix):
            L = csc_matrix(L)
        # Ensure L is in csc format as KLU works with column pointers
        if not isinstance(L, csc_matrix):
            L = L.tocsc()

        self.L = L
        self.n_vertices = L.shape[0]
        self.I = eye(self.n_vertices, format='csc')
        self.klu: Optional[KluWrapper] = None
        self.common: Optional[klu_common] = None

    def __enter__(self) -> "LUConvolve":
        self.klu = KluWrapper()
        self.common = klu_common()
        self.klu.defaults(byref(self.common))
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
        
        if self.klu is None or self.common is None:
            raise RuntimeError("LUConvolve must be used within a context manager (e.g., 'with LUConvolve(L) as conv: ...').")

        # Determine if we need complex arithmetic based on poles, residues, or input
        is_complex = np.iscomplexobj(K.Q) or np.iscomplexobj(K.R) or np.iscomplexobj(B)
        dtype = np.complex128 if is_complex else np.float64

        nDim = K.R.shape[1]
        if B.ndim == 1:
            B = B[:, np.newaxis]
        n_vertices, n_timesteps = B.shape
        
        # Ensure B is Fortran-contiguous for efficient memory access
        if not B.flags['F_CONTIGUOUS']:
            B = np.asfortranarray(B)

        W = np.zeros((n_vertices, n_timesteps, nDim), dtype=dtype)
        
        if K.D is not None and K.D.size > 0:
            # Apply the direct term D.
            W += B[..., None] * K.D

        symbolic = None
        numeric = None
        for q, r in zip(K.Q, K.R):
            try:
                # 1. Create shifted matrix
                A_shifted = self.L + q * self.I

                # 2. Symbolic Analysis
                symbolic = self.klu.analyze(A_shifted, byref(self.common))
                if not symbolic:
                    raise RuntimeError(f"KLU symbolic analysis failed with status {self.common.status}.")

                # 3. Numeric Factorization
                if is_complex:
                    numeric = self.klu.z_factor(A_shifted, symbolic, byref(self.common))
                else:
                    numeric = self.klu.factor(A_shifted, symbolic, byref(self.common))

                if not numeric:
                    raise RuntimeError(f"KLU numeric factorization failed with status {self.common.status}.")

                # 4. Solve
                if is_complex:
                    X = self.klu.z_solve(symbolic, numeric, B.astype(np.complex128, order='F', copy=False), byref(self.common))
                else:
                    # klu_solve is in-place, so we solve on a copy of B
                    X = B.copy(order='F')
                    status = self.klu.solve(symbolic, numeric, X, byref(self.common))
                    if status == 0:
                        raise RuntimeError(f"KLU solve failed with status {status}")

                # Accumulate: W += X * r
                W += X[..., None] * r
            finally:
                # 5. Free KLU objects
                if numeric:
                    self.klu.free_numeric(byref(numeric), byref(self.common))
                if symbolic:
                    self.klu.free_symbolic(byref(symbolic), byref(self.common))
                symbolic, numeric = None, None

        return W

    def _solve_system(self, B: np.ndarray, shift: Union[float, complex]) -> np.ndarray:
        """Helper to solve (L + shift*I)X = B using KLU."""
        if self.klu is None or self.common is None:
            raise RuntimeError("This method must be used within a context manager.")
        
        symbolic = None
        numeric = None
        try:
            A_shifted = self.L + shift * self.I
            is_complex = np.iscomplexobj(shift) or np.iscomplexobj(B)

            symbolic = self.klu.analyze(A_shifted, byref(self.common))
            if not symbolic:
                raise RuntimeError(f"KLU symbolic analysis failed with status {self.common.status}.")

            if is_complex:
                numeric = self.klu.z_factor(A_shifted, symbolic, byref(self.common))
            else:
                numeric = self.klu.factor(A_shifted, symbolic, byref(self.common))
            
            if not numeric:
                raise RuntimeError(f"KLU numeric factorization failed with status {self.common.status}.")

            if is_complex:
                X = self.klu.z_solve(symbolic, numeric, B.astype(np.complex128, order='F', copy=True), byref(self.common))
            else:
                X = B.copy(order='F')
                status = self.klu.solve(symbolic, numeric, X, byref(self.common))
                if status == 0:
                    raise RuntimeError(f"KLU solve failed with status {status}")
            return X
        finally:
            if numeric:
                self.klu.free_numeric(byref(numeric), byref(self.common))
            if symbolic:
                self.klu.free_symbolic(byref(symbolic), byref(self.common))

    def lowpass(self, B: np.ndarray, scales: List[float] = [1]) -> List[np.ndarray]:
        W = []
        for scale in scales:
            X = self._solve_system(B, 1.0 / scale)
            W.append(X / scale)
        return W

    def bandpass(self, B: np.ndarray, scales: List[float] = [1], order: int = 1) -> List[np.ndarray]:
        W = []
        for scale in scales:
            in_mat = B
            for _ in range(order):
                x2 = self._solve_system(in_mat, 1.0 / scale)
                x1 = self._solve_system(x2, 1.0 / scale)
                in_mat = (4.0 / scale) * (self.L @ x1)
            W.append(in_mat)
        return W

    def highpass(self, B: np.ndarray, scales: List[float] = [1]) -> List[np.ndarray]:
        W = []
        for scale in scales:
            X1 = self._solve_system(B, 1.0 / scale)
            W.append(self.L @ X1)
        return W