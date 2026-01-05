# -*- coding: utf-8 -*-
"""
Analytical Filter Functions
---------------------------
This module provides scalar implementations of common analytical filter functions
used in Spectral Graph Signal Processing. These are useful for generating target
functions for polynomial or rational approximations.

Author: Luke Lowery (lukel@tamu.edu)
"""

import numpy as np


def lowpass(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Computes the spectral response of a low-pass filter.

    Analytical form: 1 / (s*x + 1)

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues (or frequencies).
    scale : float, default: 1.0
        The scale of the filter.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    return 1.0 / (scale * x + 1.0)


def highpass(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Computes the spectral response of a high-pass filter.

    Analytical form: s*x / (s*x + 1)

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues (or frequencies).
    scale : float, default: 1.0
        The scale of the filter.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    return (scale * x) / (scale * x + 1.0)


def bandpass(x: np.ndarray, scale: float = 1.0, order: int = 1) -> np.ndarray:
    """
    Computes the spectral response of a band-pass filter.

    This is based on the SGWT wavelet design.
    Analytical form: ((4/s * x) / (x + 1/s)^2)^order

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues (or frequencies).
    scale : float, default: 1.0
        The scale of the filter.
    order : int, default: 1
        The order of the filter.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    q = 1.0 / scale
    base = (4.0 * q * x) / (x + q) ** 2
    return base**order