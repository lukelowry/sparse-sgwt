# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/__init__.py
Description: Main package initialization.
"""

# Static and Dynamic Graphs
from .cholconv import Convolve, DyConvolve

# Chebyshev Approximation
from .chebyconv import ChebyConvolve

# Analytical function generators
from . import functions

# Import Library resources
from . import library as _library

from .util import (
    VFKernel,
    ChebyKernel,
    impulse,
    estimate_spectral_bound
)

# For convenience, expose datasets and some utils from the library subpackage
# at the top level of the sgwt package.
_LAZY_RESOURCES = _library._LAZY_RESOURCES
get_cholmod_dll = _library.get_cholmod_dll
get_klu_dll = _library.get_klu_dll

def __getattr__(name):
    if name in _LAZY_RESOURCES:
        return getattr(_library, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(globals().keys()) + _LAZY_RESOURCES

__all__ = [
    "Convolve", "ChebyConvolve", "DyConvolve", "functions",
    "VFKernel", "ChebyKernel", "impulse", "get_klu_dll", "get_cholmod_dll", "estimate_spectral_bound"
] + _LAZY_RESOURCES