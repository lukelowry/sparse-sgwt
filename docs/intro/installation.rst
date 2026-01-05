Installation
============

Prerequisites
-------------

The ``sgwt`` package requires Python 3.7 or newer.

Installation via Pip
--------------------

The easiest way to install ``sgwt`` is via pip. This will automatically install the necessary dependencies (NumPy, SciPy).

.. code-block:: bash

    pip install sgwt

Dependencies
------------

The following dependencies are automatically installed:

*   **NumPy**
*   **SciPy**
*   **importlib-resources** (for Python < 3.9)

CHOLMOD
-------------------

This package leverages the ``CHOLMOD`` library from SuiteSparse for high-performance sparse matrix operations. The package currently includes a pre-compiled shared library only for **Windows**, so it is only compatible with Windows operating systems.