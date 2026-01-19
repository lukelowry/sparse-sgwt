Laplacians
----------

The library includes Laplacians for various synthetic power grid networks. These are provided as ``scipy.sparse.csc_matrix`` objects. The naming convention is ``METRIC_REGION``.

.. seealso::
   :doc:`../theory/theory_graph`
      For the mathematical derivation and physical interpretation of the **DELAY**, **LENGTH**, and **IMPEDANCE** weighting schemes.

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
Pre-computed rational approximations for common spectral graph wavelets are available as dictionaries. These can be loaded into :class:`~sgwt.VFKernel` objects for use with :meth:`~sgwt.Convolve.convolve`.

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
   See :doc:`json` for details on the JSON file format for custom kernels.

Available Data Summary
----------------------

The following table summarizes all available built-in datasets, including both power grid networks and 3D mesh models. The original mesh datasets can be found at the `Laplacian Library <https://github.com/lukelowry/laplib>`_.

.. list-table::
   :widths: 20 12 18 12 25
   :header-rows: 1

   * - Dataset
     - Vertices
     - Edges
     - Vertex Signals
     - Edge Weights
   * - **BUNNY**
     - 35947
     - 104288
     - ``x``, ``y``, ``z``
     -
   * - **HORSE**
     - 48485
     - 145449
     - ``x``, ``y``, ``z``
     -
   * - **LBRAIN**
     - 149351
     - 448041
     - ``x``, ``y``, ``z``
     -
   * - **HAWAII**
     - 37
     - 98
     - ``long``, ``lat``
     - ``length``, ``impedance``, ``delay``
   * - **TEXAS**
     - 2000
     - 2998
     - ``long``, ``lat``
     - ``length``, ``impedance``, ``delay``
   * - **EASTWEST**
     - 78000
     - 100923
     - ``long``, ``lat``
     - ``length``, ``impedance``, ``delay``
   * - **USA**
     - 82000
     - 104123
     - ``long``, ``lat``
     - ``length``, ``impedance``, ``delay``
   * - **WECC**
     - 240
     - 451
     -
     - ``length``, ``impedance``, ``delay``
