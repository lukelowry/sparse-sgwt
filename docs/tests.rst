Validation Tests
================

The test suite ensures the correctness of the CHOLMOD integration and the SGWT implementation.

Test Categories
---------------

- **Functionality** (``test_functionality.py``): Validates core features including all filter types (Low-pass, Band-pass, High-pass), Vector Fitting (VF) kernels, and dynamic topology updates via ``DyConvolve``.
- **I/O Utilities** (``test_io.py``): Ensures built-in Laplacians, signals, and kernels are loaded correctly and the CHOLMOD DLL is accessible.

Running Tests
-------------

The preferred way to run tests is using the master test runner, which provides a clean, color-coded summary of all test cases:

.. code-block:: bash

    python tests/run_tests.py

.. note::
   Validation tests currently verify the ``cholmod`` implementation against ``sksparse`` which provides a subset of similar functionality.