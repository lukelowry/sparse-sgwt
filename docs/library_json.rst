Kernel File Format
==================

The library supports loading spectral kernels defined by rational approximations (Vector Fitting) stored in JSON format. These files are parsed into :class:`sgwt.io.VFKern` objects.

JSON Structure
--------------

The JSON file represents a rational expansion of the form:

.. math::

   \mathbf{g}(\lambda) \approx \mathbf{d} + \mathbf{e}\lambda + \sum_{k=1}^{M} \frac{\mathbf{r}_k}{\lambda - q_k}

Fields
~~~~~~

- ``description`` (string, optional): A description of the kernel.
- ``nfuncs`` (int): Number of functions in the kernel.
- ``npoles`` (int): Number of poles (:math:`M`) in the approximation.
- ``d`` (float or list): The constant term :math:`d`.
- ``e`` (float or list, optional): The linear term :math:`e`.
- ``poles`` (list): A list of objects, each containing:
    - ``q`` (float): The pole location.
    - ``r`` (list): The residues corresponding to this pole.


Example
-------

An example structure based on the ``MODIFIED_MORLET.json`` file:

.. code-block:: json

    {
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
            }
        ],
        "description": "Modified Morlet Wavelet With Central Frequnecy of 2pi"
    }