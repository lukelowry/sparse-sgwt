# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/klu/wrapper.py
Description: Low-level Python wrapper for the KLU C library.
"""

from ctypes import byref, POINTER, c_int32, CDLL, c_double, c_int
import numpy as np
from scipy.sparse import csc_matrix

from .structs import klu_symbolic, klu_numeric, klu_common
from ..util import get_klu_dll

# Status values
KLU_OK = 0
KLU_SINGULAR = 1
KLU_OUT_OF_MEMORY = -2
KLU_INVALID = -3
KLU_TOO_LARGE = -4

# Ordering methods
KLU_ORDERING_AMD = 0
KLU_ORDERING_COLAMD = 1
KLU_ORDERING_GIVEN = 2
KLU_ORDERING_USER = 3

class KluWrapper:
    """
    A wrapper class for interacting with the KLU DLL.

    WARNING: Should only be used indirectly through a context manager like
    LUConvolve, otherwise memory leaks may occur.
    """

    def __init__(self) -> None:
        """Initializes the KLU wrapper."""
        self.dll = get_klu_dll()
        
        # DLL Setup    
        self.config_function_args(self.dll)
        self.config_return_types(self.dll)

        # This tuple holds references to numpy arrays passed to C
        # to prevent them from being garbage collected prematurely.
        self._keep_alive = ()

    def defaults(self, common_ptr) -> int:
        """Sets default parameters in a klu_common struct."""
        return self.dll.klu_defaults(common_ptr)

    def analyze(self, A: csc_matrix, common_ptr):
        """Performs symbolic analysis on a matrix A."""
        n = A.shape[0]
        (Ap, Ai, _), self._keep_alive = self._csc_to_klu_inputs(A)
        return self.dll.klu_analyze(n, Ap, Ai, common_ptr)

    def factor(self, A: csc_matrix, symbolic_ptr, common_ptr):
        """Performs numeric factorization of a real matrix A."""
        (Ap, Ai, Ax), self._keep_alive = self._csc_to_klu_inputs(A)
        return self.dll.klu_factor(Ap, Ai, Ax, symbolic_ptr, common_ptr)

    def z_factor(self, A: csc_matrix, symbolic_ptr, common_ptr):
        """Performs numeric factorization of a complex matrix A."""
        (Ap, Ai, Ax), self._keep_alive = self._csc_to_klu_inputs(A, is_complex=True)
        return self.dll.klu_z_factor(Ap, Ai, Ax, symbolic_ptr, common_ptr)

    def solve(self, symbolic_ptr, numeric_ptr, B: np.ndarray, common_ptr) -> int:
        """Solves a real system AX=B. B is overwritten with the solution."""
        if not B.flags['F_CONTIGUOUS']:
            raise ValueError("Input array B must be Fortran-contiguous.")
        ldim, nrhs = B.shape
        B_ptr = B.ctypes.data_as(POINTER(c_double))
        return self.dll.klu_solve(symbolic_ptr, numeric_ptr, ldim, nrhs, B_ptr, common_ptr)

    def z_solve(self, symbolic_ptr, numeric_ptr, B: np.ndarray, common_ptr) -> np.ndarray:
        """Solves a complex system AX=B. Returns a new array with the solution."""
        if not B.flags['F_CONTIGUOUS']:
            raise ValueError("Input array B must be Fortran-contiguous.")
        ldim, nrhs = B.shape
        
        # Convert complex numpy array to interleaved float array for KLU
        B_interleaved = np.empty(B.size * 2, dtype=np.float64)
        B_interleaved[0::2] = B.ravel(order='F').real
        B_interleaved[1::2] = B.ravel(order='F').imag
        B_ptr = B_interleaved.ctypes.data_as(POINTER(c_double))

        status = self.dll.klu_z_solve(symbolic_ptr, numeric_ptr, ldim, nrhs, B_ptr, common_ptr)
        if status == 0:
            raise RuntimeError(f"KLU complex solve failed with status: {status}")

        # Convert interleaved solution back to complex numpy array
        X_real = B_interleaved[0::2]
        X_imag = B_interleaved[1::2]
        X_flat = X_real + 1j * X_imag
        return X_flat.reshape(B.shape, order='F')

    def free_symbolic(self, symbolic_ptr, common_ptr):
        """Frees a klu_symbolic object."""
        return self.dll.klu_free_symbolic(symbolic_ptr, common_ptr)

    def free_numeric(self, numeric_ptr, common_ptr):
        """Frees a klu_numeric object."""
        return self.dll.klu_free_numeric(numeric_ptr, common_ptr)

    def _csc_to_klu_inputs(self, A: csc_matrix, is_complex: bool = False):
        """
        Prepares a SciPy CSC matrix for KLU, returning pointers and data arrays.
        The data arrays must be kept in scope to prevent garbage collection.
        """
        # KLU expects 32-bit integers for indices and pointers
        indices = A.indices.astype(np.int32, copy=False)
        indptr = A.indptr.astype(np.int32, copy=False)

        # Auto-detect complex data if not explicitly specified to avoid casting warnings
        if not is_complex and np.iscomplexobj(A.data):
            is_complex = True

        if is_complex:
            # For complex data, KLU expects an interleaved array of doubles: [real, imag, real, imag, ...]
            if not np.iscomplexobj(A.data):
                data = A.data.astype(np.complex128, copy=False)
            else:
                data = A.data
            
            interleaved_data = np.empty(data.size * 2, dtype=np.float64)
            interleaved_data[0::2] = data.real
            interleaved_data[1::2] = data.imag
            Ax_ptr = interleaved_data.ctypes.data_as(POINTER(c_double))
            keep_alive_data = interleaved_data
        else:
            # For real data, a simple double array is expected
            data = A.data.astype(np.float64, copy=False)
            Ax_ptr = data.ctypes.data_as(POINTER(c_double))
            keep_alive_data = data

        Ai_ptr = indices.ctypes.data_as(POINTER(c_int32))
        Ap_ptr = indptr.ctypes.data_as(POINTER(c_int32))

        # Return pointers and the underlying numpy arrays to be kept alive
        return (Ap_ptr, Ai_ptr, Ax_ptr), (indices, indptr, keep_alive_data)

    def config_function_args(self, dll: CDLL) -> None:
        """Set the argument types for all wrapped KLU functions."""

        # klu_defaults
        dll.klu_defaults.argtypes = [
            POINTER(klu_common)
        ]

        # klu_analyze
        dll.klu_analyze.argtypes = [
            c_int32,                # n
            POINTER(c_int32),       # Ap
            POINTER(c_int32),       # Ai
            POINTER(klu_common)
        ]

        # klu_factor
        dll.klu_factor.argtypes = [
            POINTER(c_int32),       # Ap
            POINTER(c_int32),       # Ai
            POINTER(c_double),      # Ax
            POINTER(klu_symbolic),
            POINTER(klu_common)
        ]

        # klu_z_factor
        dll.klu_z_factor.argtypes = [
            POINTER(c_int32),       # Ap
            POINTER(c_int32),       # Ai
            POINTER(c_double),      # Ax (interleaved complex)
            POINTER(klu_symbolic),
            POINTER(klu_common)
        ]

        # klu_solve
        dll.klu_solve.argtypes = [
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            c_int32,                # ldim
            c_int32,                # nrhs
            POINTER(c_double),      # B
            POINTER(klu_common)
        ]

        # klu_z_solve
        dll.klu_z_solve.argtypes = [
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            c_int32,                # ldim
            c_int32,                # nrhs
            POINTER(c_double),      # B (interleaved complex)
            POINTER(klu_common)
        ]

        # klu_tsolve
        dll.klu_tsolve.argtypes = [
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            c_int32,                # ldim
            c_int32,                # nrhs
            POINTER(c_double),      # B
            POINTER(klu_common)
        ]

        # klu_z_tsolve
        dll.klu_z_tsolve.argtypes = [
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            c_int32,                # ldim
            c_int32,                # nrhs
            POINTER(c_double),      # B (interleaved complex)
            c_int,                  # conj_solve
            POINTER(klu_common)
        ]

        # klu_free_symbolic
        dll.klu_free_symbolic.argtypes = [
            POINTER(POINTER(klu_symbolic)),
            POINTER(klu_common)
        ]

        # klu_free_numeric
        dll.klu_free_numeric.argtypes = [
            POINTER(POINTER(klu_numeric)),
            POINTER(klu_common)
        ]
        # klu_z_free_numeric is identical to klu_free_numeric
        dll.klu_z_free_numeric.argtypes = dll.klu_free_numeric.argtypes

        # klu_refactor
        dll.klu_refactor.argtypes = [
            POINTER(c_int32),       # Ap
            POINTER(c_int32),       # Ai
            POINTER(c_double),      # Ax
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            POINTER(klu_common)
        ]

        # klu_z_refactor
        dll.klu_z_refactor.argtypes = [
            POINTER(c_int32),       # Ap
            POINTER(c_int32),       # Ai
            POINTER(c_double),      # Ax (interleaved complex)
            POINTER(klu_symbolic),
            POINTER(klu_numeric),
            POINTER(klu_common)
        ]

    def config_return_types(self, dll: CDLL) -> None:
        """Set the return types for all wrapped KLU functions."""

        dll.klu_defaults.restype = c_int

        dll.klu_analyze.restype = POINTER(klu_symbolic)

        dll.klu_factor.restype = POINTER(klu_numeric)
        dll.klu_z_factor.restype = POINTER(klu_numeric)

        dll.klu_solve.restype = c_int
        dll.klu_z_solve.restype = c_int

        dll.klu_tsolve.restype = c_int
        dll.klu_z_tsolve.restype = c_int

        dll.klu_free_symbolic.restype = c_int
        dll.klu_free_numeric.restype = c_int
        dll.klu_z_free_numeric.restype = c_int

        dll.klu_refactor.restype = c_int
        dll.klu_z_refactor.restype = c_int