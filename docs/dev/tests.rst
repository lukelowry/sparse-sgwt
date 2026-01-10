Validation Tests
================

The test suite verifies the correctness of ``sgwt`` installations. All tests achieve 100% code coverage.

Test Modules
------------

- **test_cholconv.py**: Static and dynamic graph convolution (``Convolve``, ``DyConvolve``), VF kernels, topology updates.
- **test_chebyshev.py**: Chebyshev polynomial approximation and ``ChebyConvolve``.
- **test_functions.py**: Analytical filter functions (low-pass, band-pass, high-pass).
- **test_util.py**: Resource loading, DLL utilities, ``ChebyKernel``, ``VFKernel``.

Running Tests
-------------

Install test dependencies:

.. code-block:: bash

    pip install sgwt[test]

Run from source checkout:

.. code-block:: bash

    pytest

Run on installed package:

.. code-block:: bash

    pytest --pyargs sgwt.tests
