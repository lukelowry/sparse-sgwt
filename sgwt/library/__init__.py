# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/library/__init__.py
Description: Library module initialization for built-in datasets.
"""
from ..io import (
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
