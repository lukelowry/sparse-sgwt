# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/klu/structs.py
Description: Ctypes definitions for KLU C structures.
"""

from ctypes import (
    Structure, POINTER, CFUNCTYPE,
    c_double, c_int, c_int32, c_void_p, c_size_t
)

# Forward declaration for klu_common to be used in function pointer type
class klu_common(Structure):
    """The KLU 'Common' object used to control parameters and store statistics."""
    pass

# Define the function pointer type for user_order
# int32_t (*user_order) (int32_t, int32_t *, int32_t *, int32_t *, struct klu_common_struct *)
USER_ORDER_FUNC = CFUNCTYPE(c_int32, c_int32, POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(klu_common))

class klu_symbolic(Structure):
    """Symbolic object from klu_analyze."""
    _fields_ = [
        ("symmetry", c_double),
        ("est_flops", c_double),
        ("lnz", c_double),
        ("unz", c_double),
        ("Lnz", POINTER(c_double)),
        ("n", c_int32),
        ("nz", c_int32),
        ("P", POINTER(c_int32)),
        ("Q", POINTER(c_int32)),
        ("R", POINTER(c_int32)),
        ("nzoff", c_int32),
        ("nblocks", c_int32),
        ("maxblock", c_int32),
        ("ordering", c_int32),
        ("do_btf", c_int32),
        ("structural_rank", c_int32),
    ]

class klu_numeric(Structure):
    """Numeric object from klu_factor."""
    _fields_ = [
        ("n", c_int32),
        ("nblocks", c_int32),
        ("lnz", c_int32),
        ("unz", c_int32),
        ("max_lnz_block", c_int32),
        ("max_unz_block", c_int32),
        ("Pnum", POINTER(c_int32)),
        ("Pinv", POINTER(c_int32)),
        ("Lip", POINTER(c_int32)),
        ("Uip", POINTER(c_int32)),
        ("Llen", POINTER(c_int32)),
        ("Ulen", POINTER(c_int32)),
        ("LUbx", POINTER(c_void_p)),
        ("LUsize", POINTER(c_size_t)),
        ("Udiag", c_void_p),
        ("Rs", POINTER(c_double)),
        ("worksize", c_size_t),
        ("Work", c_void_p),
        ("Xwork", c_void_p),
        ("Iwork", POINTER(c_int32)),
        ("Offp", POINTER(c_int32)),
        ("Offi", POINTER(c_int32)),
        ("Offx", c_void_p),
        ("nzoff", c_int32),
    ]

# Define the fields for klu_common
klu_common._fields_ = [
    ("tol", c_double),
    ("memgrow", c_double),
    ("initmem_amd", c_double),
    ("initmem", c_double),
    ("maxwork", c_double),
    ("btf", c_int),
    ("ordering", c_int),
    ("scale", c_int),
    ("user_order", USER_ORDER_FUNC),
    ("user_data", c_void_p),
    ("halt_if_singular", c_int),
    ("status", c_int),
    ("nrealloc", c_int),
    ("structural_rank", c_int32),
    ("numerical_rank", c_int32),
    ("singular_col", c_int32),
    ("noffdiag", c_int32),
    ("flops", c_double),
    ("rcond", c_double),
    ("condest", c_double),
    ("rgrowth", c_double),
    ("work", c_double),
    ("memusage", c_size_t),
    ("mempeak", c_size_t),
]