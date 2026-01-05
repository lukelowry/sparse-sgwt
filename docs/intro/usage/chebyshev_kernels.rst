Chebyshev Approximation
=====================================

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

    # Initialize convolution context to get spectrum bound
    conv = sgwt.ChebConvolve(L)
    ubnd = conv.spectrum_bound

    # Create the kernel
    kernel = sgwt.ChebyKernel.from_function(f, order=50, spectrum_bound=ubnd)

    # Apply convolution
    with conv:
        result = conv.convolve(signal, kernel)