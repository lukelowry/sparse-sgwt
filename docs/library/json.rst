Kernel JSON Structure
==================

The library supports loading spectral kernels defined by rational approximations (Vector Fitting) stored in JSON format. These files are parsed into :class:`~sgwt.VFKernel` objects for use in convolution.

The JSON file represents a rational expansion of the filter :math:`\mathbf{g}(\lambda)` in the spectral domain, where :math:`\lambda` represents a graph Laplacian eigenvalue. The expansion takes the form:

.. math::

   \mathbf{g}(\lambda) \approx \mathbf{d} + \mathbf{e}\lambda + \sum_{k=1}^{M} \frac{\mathbf{r}_k}{\lambda + q_k}

This structure is derived from a Vector Fitting procedure and allows for efficient computation of the graph convolution.

Fields
~~~~~~

- ``description`` (string, optional): A human-readable description of the kernel.
- ``nfuncs`` (int): The number of separate functions or dimensions in the kernel.
- ``npoles`` (int): The number of poles (:math:`M`) in the rational approximation.
- ``d`` (float or list): The constant term :math:`\mathbf{d}` (direct feedthrough).
- ``e`` (float or list, optional): The linear term :math:`\mathbf{e}`. This is often zero.
- ``poles`` (list): A list of pole objects, where each object contains:
    - ``q`` (float): A pole location :math:`q_k`.
    - ``r`` (list): A list of residues :math:`\mathbf{r}_k` corresponding to the pole. The length of this list must match ``nfuncs``.


Example
-------

Below is a truncated example from the built-in ``MODIFIED_MORLET.json`` file.

.. code-block:: json

    {
        "description": "Modified Morlet Wavelet With Central Frequnecy of 2pi",
        "nfuncs": 1,
        "npoles": 14,
        "d": 0.0027966205394028575,
        "e": 0,
        "poles": [
            {
                "q": -19219.857112413505,
                "r": [-307.5212280632877]
            },
            {
                "q": -2935.9392937251964,
                "r": [132.53827518359017]
            },
            {
                "q": -56.43582644772509,
                "r": [800722.1859990739]
            }
        ]
    }

.. seealso::
   :doc:`../usage/kernel_functions`
      For an example of how to load and use a custom kernel in a convolution.