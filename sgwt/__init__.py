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

# Spectral Graph Modal Analysis
from .sgma import SGMA, ModeTable

# Analytical function generators
from . import functions

# Import util to access its lazy-loading capabilities and expose key components
from . import util
from .util import (
    VFKernel,
    ChebyKernel,
    impulse,
    estimate_spectral_bound,
    get_cholmod_dll,
    get_klu_dll,
)

# Delegate lazy-loading of data resources to the util module
_LAZY_RESOURCES = list(util._LAZY_REGISTRY.keys())

def __getattr__(name):
    """Lazily loads data resources (Laplacians, signals, etc.) on first access."""
    if name in _LAZY_RESOURCES:
        return getattr(util, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")  # pragma: no cover

def __dir__():  # pragma: no cover
    """Improves tab-completion for lazy-loaded attributes."""
    return list(globals().keys()) + _LAZY_RESOURCES

__all__ = [
    "Convolve", "ChebyConvolve", "DyConvolve", "SGMA", "ModeTable", "functions",
    "VFKernel", "ChebyKernel", "impulse", "get_klu_dll", "get_cholmod_dll", "estimate_spectral_bound"
] + _LAZY_RESOURCES