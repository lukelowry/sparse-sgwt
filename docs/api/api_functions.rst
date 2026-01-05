Math Functions
===========================

This module provides scalar implementations of common analytical filter functions
used in Spectral Graph Signal Processing. These are useful for generating target
functions for polynomial or rational approximations.

Low-Pass
--------

.. math::

   \phi_s(\lambda) = \frac{1}{s\lambda + 1}

.. autofunction:: sgwt.functions.lowpass

High-Pass
---------

.. math::

   \mu_s(\lambda) = \frac{s\lambda}{s\lambda + 1}

.. autofunction:: sgwt.functions.highpass

Band-Pass
---------

.. math::

   \psi_s(\lambda) = \frac{4s\lambda}{(s\lambda+1)^2}

.. autofunction:: sgwt.functions.bandpass