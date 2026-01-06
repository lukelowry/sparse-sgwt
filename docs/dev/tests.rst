Validation Tests
================

The test suite is included with the package and can be run to verify the correctness of an ``sgwt`` installation. It ensures the proper functioning of the CHOLMOD integration, filter implementations, and utility functions.

Test Categories
---------------

- **Functionality** (``sgwt/tests/test_cholesky.py``): Validates core features including all filter types (Low-pass, Band-pass, High-pass), Vector Fitting (VF) kernels, and dynamic topology updates via ``DyConvolve``.
- **Utilities** (``sgwt/tests/test_util.py``): Ensures built-in Laplacians, signals, and kernels are loaded correctly and the CHOLMOD/KLU DLLs are accessible.
- **Chebyshev** (``sgwt/tests/test_chebyshev.py``): Verifies the accuracy and stability of Chebyshev polynomial approximations for graph filters.

Running Tests
-------------

The test suite is built on `pytest`. Before running, install the necessary test dependencies.

From a source checkout:

.. code-block:: bash

    pip install .[test]

On an installed package:

.. code-block:: bash

    pip install sgwt[test]

There are two primary ways to run the tests:

From a Source Checkout
~~~~~~~~~~~~~~~~~~~~~~

If you have cloned the repository, you can run tests directly on the source code. This is useful for development.

1.  **Using pytest directly (recommended)**:
    Navigate to the project root directory and run:

.. code-block:: bash

    pytest

    Pytest will automatically discover the ``pytest.ini`` configuration and run all tests in the ``sgwt/tests`` directory.

2.  **Using the convenience script**:
    The package includes a script that ensures the local `sgwt` module is used, even without installation. This can be run from the project root as:

.. code-block:: bash

    python -m sgwt.tests.run_tests

On an Installed Package
~~~~~~~~~~~~~~~~~~~~~~~

To verify that an installed version of ``sgwt`` is working correctly in your environment, you can run the tests that were included with the package. Use the ``--pyargs`` flag to tell pytest to find the tests inside the installed ``sgwt`` module:

.. code-block:: bash

    pytest --pyargs sgwt.tests

Test Report
~~~~~~~~~~~

The output is configured to be verbose and has been cleaned to show only the test's class and function name (e.g., ``TestCholesky::test_...``), providing a more readable report.
