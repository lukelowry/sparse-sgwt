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

    .. math::

        \\phi_s(\\lambda) = \\frac{1}{s\\lambda + 1}

    This filter attenuates high-frequency (large eigenvalue) components
    while preserving low-frequency structure.

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues :math:`\\lambda`.
    scale : float, default: 1.0
        The scale parameter :math:`s`. Larger values shift the cutoff
        to lower frequencies.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    return 1.0 / (scale * x + 1.0)


def highpass(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Computes the spectral response of a high-pass filter.

    .. math::

        \\mu_s(\\lambda) = \\frac{s\\lambda}{s\\lambda + 1}

    This filter attenuates low-frequency (small eigenvalue) components
    while preserving high-frequency structure.

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues :math:`\\lambda`.
    scale : float, default: 1.0
        The scale parameter :math:`s`. Larger values shift the cutoff
        to lower frequencies.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    return (scale * x) / (scale * x + 1.0)


def bandpass(x: np.ndarray, scale: float = 1.0, order: int = 1) -> np.ndarray:
    """
    Computes the spectral response of a band-pass filter.

    This wavelet generating kernel is based on the SGWT design:

    .. math::

        \\Psi_s(\\lambda) = \\left( \\frac{4\\lambda/s}{(\\lambda + 1/s)^2} \\right)^n

    where :math:`n` is the filter order. The filter peaks at :math:`\\lambda = 1/s`
    with maximum gain of 1, and satisfies the admissibility condition
    :math:`\\Psi(0) = 0`.

    Parameters
    ----------
    x : np.ndarray
        Input array of eigenvalues :math:`\\lambda`.
    scale : float, default: 1.0
        The scale parameter :math:`s`. The filter peaks at :math:`\\lambda = 1/s`.
    order : int, default: 1
        The filter order :math:`n`. Higher orders produce narrower bandwidths.

    Returns
    -------
    np.ndarray
        The filter's gain at each point in `x`.
    """
    q = 1.0 / scale
    base = (4.0 * q * x) / (x + q) ** 2
    return base**order