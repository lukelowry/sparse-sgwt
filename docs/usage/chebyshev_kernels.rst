Chebyshev Approximation
=====================================

A Chebyshev approximation represents a spectral filter :math:`g(\lambda)` as a sum of Chebyshev polynomials :math:`T_k`:

.. math::

   g(\lambda) \approx \sum_{k=0}^{N} c_k T_k\left(\frac{2\lambda}{\lambda_{\text{max}}} - 1\right)

where :math:`\lambda_{\text{max}}` is the largest eigenvalue of the graph Laplacian and :math:`c_k` are the fitted coefficients. This allows for convolution via an efficient recurrence relation (Clenshaw's algorithm).

While ``sgwt`` is optimized for sparse direct solvers (rational filters), it provides a ``ChebyKernel`` implementation as an alternative for specific use cases and performance benchmarking.

The ``ChebyKernel`` class allows you to approximate arbitrary continuous functions on the graph spectrum. This is primarily used for:

1. **Performance Benchmarking**: Comparing the efficiency of polynomial recurrence against sparse direct solves.
2. **Non-Rational Kernels**: Approximating spectral shapes that are difficult to represent with rational functions.
3. **Avoiding Factorization**: In scenarios where the overhead of a sparse Cholesky factorization is undesirable.

Basic Usage
-----------

.. code-block:: python

    import sgwt
    import numpy as np

    # Define a custom filter function
    def my_filter(x):
        return np.exp(-x) * np.sin(5 * x)

    def f(x):
        return np.array([my_filter(x)]).T

    # Create the kernel
    kernel = sgwt.ChebyModel.kernel(L, f, order=50)

    # Apply convolution
    with sgwt.ChebyConvolve(L) as conv:
        result = conv.convolve(signal, kernel)

.. seealso::
   :doc:`../theory/theory_kernel_fitting`
      For a comparison between polynomial and rational approximations.
