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

# Fitting models
from .fitting import VFModel, ChebyModel

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
    load_ply_laplacian,
    load_ply_xyz,
    list_graphs,
)

# Static type declarations for lazy-loaded resources (IDE intellisense only)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Any
    import numpy as np
    from scipy.sparse import csc_matrix

    # Kernels
    GAUSSIAN_WAV: Dict[str, Any]
    MEXICAN_HAT: Dict[str, Any]
    MODIFIED_MORLET: Dict[str, Any]
    SHANNON: Dict[str, Any]

    # Delay Laplacians
    DELAY_EAST: csc_matrix
    DELAY_EASTWEST: csc_matrix
    DELAY_HAWAII: csc_matrix
    DELAY_NEISO: csc_matrix
    DELAY_TEXAS: csc_matrix
    DELAY_USA: csc_matrix
    DELAY_WECC: csc_matrix

    # Impedance Laplacians
    IMPEDANCE_EASTWEST: csc_matrix
    IMPEDANCE_HAWAII: csc_matrix
    IMPEDANCE_TEXAS: csc_matrix
    IMPEDANCE_USA: csc_matrix
    IMPEDANCE_WECC: csc_matrix

    # Length Laplacians
    LENGTH_EASTWEST: csc_matrix
    LENGTH_HAWAII: csc_matrix
    LENGTH_NEISO: csc_matrix
    LENGTH_TEXAS: csc_matrix
    LENGTH_USA: csc_matrix
    LENGTH_WECC: csc_matrix

    # Coordinate Signals
    COORD_EASTWEST: np.ndarray
    COORD_HAWAII: np.ndarray
    COORD_TEXAS: np.ndarray
    COORD_USA: np.ndarray

    # Mesh Laplacians
    MESH_BUNNY: csc_matrix
    MESH_HORSE: csc_matrix
    MESH_LBRAIN: csc_matrix
    MESH_RBRAIN: csc_matrix

    # Mesh Coordinates
    BUNNY_XYZ: np.ndarray
    HORSE_XYZ: np.ndarray
    LBRAIN_XYZ: np.ndarray
    RBRAIN_XYZ: np.ndarray

# Delegate lazy-loading of data resources to the util module
_LAZY_RESOURCES = list(util._ensure_registry().keys())

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
    "VFKernel", "ChebyKernel", "VFModel", "ChebyModel",
    "impulse", "get_klu_dll", "get_cholmod_dll", "estimate_spectral_bound",
    "load_ply_laplacian", "load_ply_xyz", "list_graphs"
] + _LAZY_RESOURCES