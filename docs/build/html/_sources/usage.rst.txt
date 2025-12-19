Basic Usage
===========

A typical workflow consists of:

1. Constructing a graph Laplacian
2. Defining a spectral kernel
3. Applying the wavelet transform

Example:

.. code-block:: python

   import numpy as np
   from sgwt import wavelet_transform

   W = wavelet_transform(L, f, scales)

Further details are provided in the API reference.
