# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/library/__init__.py
Description: Library module initialization for built-in datasets.
"""

from .. import util

# Define the list of lazy resources by introspecting the implementation in util
_LAZY_RESOURCES = list(util._LAZY_REGISTRY.keys())

# Expose DLL loaders from the library module
get_cholmod_dll = util.get_cholmod_dll
get_klu_dll = util.get_klu_dll

def __getattr__(name):
    if name in _LAZY_RESOURCES:
        # getattr will trigger the lazy loading mechanism in util.py
        return getattr(util, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(globals().keys()) + _LAZY_RESOURCES

__all__ = ["get_cholmod_dll", "get_klu_dll"] + _LAZY_RESOURCES
