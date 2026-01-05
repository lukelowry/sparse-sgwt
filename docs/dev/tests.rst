Validation Tests
================

The test suite ensures the correctness of the CHOLMOD integration and the SGWT implementation.

Test Categories
---------------

- **Functionality** (``test_cholesky.py``): Validates core features including all filter types (Low-pass, Band-pass, High-pass), Vector Fitting (VF) kernels, and dynamic topology updates via ``DyConvolve``.
- **Utilities** (``test_util.py``): Ensures built-in Laplacians, signals, and kernels are loaded correctly and the CHOLMOD DLL is accessible.
- **Chebyshev** (``test_chebyshev.py``): Verifies the accuracy and stability of Chebyshev polynomial approximations for graph filters.

Running Tests
-------------

The preferred way to run tests is using the master test runner, which provides a clean, color-coded summary of all test cases:

.. code-block:: bash

    python tests/run_tests.py
