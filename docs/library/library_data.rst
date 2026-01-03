Data Library
============

The ``sgwt`` library includes a repository of built-in graph Laplacians, signals, and spectral kernels for testing and demonstration. These resources can be imported directly from the top-level package.

Laplacians
----------

The library includes Laplacians for various synthetic power grid networks. These are provided as ``scipy.sparse.csc_matrix`` objects. The naming convention is ``METRIC_REGION``.

*   **DELAY**: Edge weights are based on phase distance (:math:`\theta^{-2}`).
*   **LENGTH**: Edge weights are based on physical transmission line length (:math:`\ell^{-2}`).
*   **IMPEDANCE**: Edge weights are based on electrical impedance (:math:`|Z|`).

**Usage**

.. code-block:: python

   import sgwt

   # Load the Laplacian for the synthetic Texas grid
   # where edge weights are based on phase delay.
   L_texas = sgwt.DELAY_TEXAS

   print(type(L_texas))
   # <class 'scipy.sparse.csc.csc_matrix'>
   print(L_texas.shape)
   # (2000, 2000)

Signals
-------

Vertex-domain signals are provided for some graphs, most commonly geographic coordinates.

*   **COORDS**: An ``(n_vertices,2)`` NumPy array containing the longitude and latitude of each node.

**Usage**

.. code-block:: python

   import sgwt

   # Load the geographic coordinates for the Texas grid
   coords_texas = sgwt.COORD_TEXAS

   print(type(coords_texas))
   # <class 'numpy.ndarray'>
   print(coords_texas.shape)
   # (2000, 2)

Kernels
-------

Pre-computed rational approximations for common spectral graph wavelets are available as dictionaries. These can be loaded into :class:`~sgwt.util.VFKernel` objects for use with :meth:`~sgwt.static.Convolve.convolve`.

*   **MEXICAN_HAT**: Mexican Hat wavelet.
*   **MODIFIED_MORLET**: Modified Morlet wavelet.
*   **SHANNON**: Shannon (ideal band-pass) wavelet.
*   **GAUSSIAN_WAV**: Gaussian wavelet.

**Usage**

.. code-block:: python

   from sgwt import VFKernel, MODIFIED_MORLET

   # The built-in kernel is a dictionary
   print(type(MODIFIED_MORLET))
   # <class 'dict'>

   # Load it into a VFKernel object for use in convolution
   kernel = VFKernel.from_dict(MODIFIED_MORLET)

   print(kernel.R.shape)
   # (14, 1)

.. seealso::
   See :doc:`library_json` for details on the JSON file format for custom kernels.

Available Data Summary
----------------------

The following table summarizes the available built-in datasets.

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