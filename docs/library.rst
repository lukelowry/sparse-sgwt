Data Library
============

This directory contains built-in graph Laplacians, signals, and spectral kernels used for testing and demonstration of the Sparse SGWT library.

Available Data
--------------

The library includes Laplacians and signals for various synthetic power grid networks. The following table lists the available combinations of graph topologies and data types.

.. list-table::
   :widths: 25 15 15 15 15
   :header-rows: 1

   * - Graph Name
     - DELAY
     - IMPEDANCE
     - LENGTH
     - COORDS
   * - **TEXAS**
     - Yes
     - Yes
     - Yes
     - Yes
   * - **USA**
     - Yes
     - Yes
     - Yes
     - Yes
   * - **EASTWEST**
     - Yes
     - Yes
     - Yes
     - Yes
   * - **HAWAII**
     - Yes
     - Yes
     - Yes
     - Yes
   * - **WECC**
     - Yes
     - Yes
     - Yes
     - No

Laplacians
----------

The library includes Laplacians for various synthetic power grid networks (Texas, Western Interconnection, Eastern Interconnection, etc.). These are provided as compressed sparse column (CSC) matrices.

*   **DELAY**: Edge weights are based on phase distance (:math:`\theta^{-2}`).
*   **LENGTH**: Edge weights are based on physical transmission line length (:math:`\ell^{-2}`).
*   **IMPEDANCE**: Edge weights are based on electrical impedance (:math:`|Z|`).

Signals
-------

*   **COORDS**: Vertex-domain signals representing geographic locations. These are :math:`N \times 2` arrays containing the longitude and latitude of each node/bus.

Kernels
-------

*   **KERNELS**: JSON files defining rational approximations (Vector Fitting) for spectral graph wavelets. These are loaded as :class:`sgwt.io.VFKern` objects.

    *   **MEXICAN_HAT**: Mexican Hat wavelet.
    *   **MODIFIED_MORLET**: Modified Morlet wavelet.
    *   **SHANNON**: Shannon (ideal band-pass) wavelet.
    *   **GAUSSIAN_WAV**: Gaussian wavelet.

    See :doc:`library_json` for details on the file format.