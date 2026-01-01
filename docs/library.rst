SGWT Data Library
=================

This directory contains built-in graph Laplacians, signals, and spectral kernels used for testing and demonstration of the Sparse SGWT library.


Contents
--------

Laplacians
~~~~~~~~~~

The library includes Laplacians for various synthetic power grid networks (Texas, Western Interconnection, Eastern Interconnection, etc.).

*   **DELAY**: Edge weights are based on phase distance (:math:`\theta^{-2}`).
*   **LENGTH**: Edge weights are based on physical transmission line length (:math:`\ell^{-2}`).
*   **IMPEDANCE**: Edge weights are based on electrical impedance (:math:`|Z|`).

Signals
~~~~~~~

*   **SIGNALS**: Contains vertex-domain signals. ``COORDS`` are :math:`N \times 2` arrays containing the longitude and latitude of each node/bus.

Kernels
~~~~~~~

*   **KERNELS**: JSON files defining rational approximations (Vector Fitting) for spectral graph wavelets.

    *   Mexican Hat
    *   Modified Morlet
    *   Shannon
    *   Gaussian