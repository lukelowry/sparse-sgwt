Validation Tests
================

The test suite ensures the correctness of the CHOLMOD integration and the SGWT implementation.

Test Categories
---------------

- **Functionality** (``test_functionality.py``): Validates core features including all filter types (Low-pass, Band-pass, High-pass), Vector Fitting (VF) kernels, and dynamic topology updates via ``DyConvolve``.
- **I/O Utilities** (``test_io.py``): Ensures built-in Laplacians, signals, and kernels are loaded correctly and the CHOLMOD DLL is accessible.
- **Chebyshev** (``test_chebyshev.py``): Verifies the accuracy and stability of Chebyshev polynomial approximations for graph filters.
- **LU Decomposition** (``test_lu.py``): Tests the LU-based convolution method, specifically for handling complex poles which are not supported by the Cholesky solver.

Running Tests
-------------

The preferred way to run tests is using the master test runner, which provides a clean, color-coded summary of all test cases:

.. code-block:: bash

    python tests/run_tests.py
