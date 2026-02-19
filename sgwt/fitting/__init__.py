# -*- coding: utf-8 -*-
"""
Fitting Subpackage
------------------
Model fitting tools for constructing rational and polynomial
approximations from frequency response data.
"""
from .vfit import VFModel
from .chebyshev import ChebyModel

__all__ = ["VFModel", "ChebyModel"]
