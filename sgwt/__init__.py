# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/__init__.py
Description: Main package initialization.
"""

# Static Graphs (Typical use case)
from .static import Convolve

# Chebyshev Approximation
from .cheb import ChebConvolve

# Dynamic Graphs (Optimized performance, less versatile)
from .dynamic import DyConvolve

# Lanczos Approximation
from .lanz import LanzConvolve

# LU Decomposition (for complex poles)
from .lu_convolve import LUConvolve

from .util import (

    # Vector Fitting Dataclass
    VFKernel,
    ChebyKernel,
    impulse,

    # DLL Reader
    get_klu_dll,
    get_cholmod_dll,
    
    # Kernels
    MEXICAN_HAT,
    GAUSSIAN_WAV,
    MODIFIED_MORLET,
    SHANNON,
    
    # Laplacians
    DELAY_EASTWEST,
    DELAY_HAWAII,
    DELAY_TEXAS,
    DELAY_USA,
    DELAY_WECC,
    
    IMPEDANCE_EASTWEST,
    IMPEDANCE_HAWAII,
    IMPEDANCE_TEXAS,
    IMPEDANCE_USA,
    IMPEDANCE_WECC,
    
    LENGTH_EASTWEST,
    LENGTH_HAWAII,
    LENGTH_TEXAS,
    LENGTH_USA,
    LENGTH_WECC,
    
    # Signals
    COORD_EASTWEST,
    COORD_HAWAII,
    COORD_TEXAS,
    COORD_USA
)