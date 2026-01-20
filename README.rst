Sparse Graph Convolution
====================================

.. |pypi| image:: https://img.shields.io/pypi/v/sgwt.svg
   :target: https://pypi.org/project/sgwt/
   :alt: PyPI Version

.. |python| image:: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg
   :target: https://pypi.org/project/sgwt/
   :alt: Python Version

.. |license| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: ./LICENSE.md
   :alt: License

.. |coverage| image:: https://img.shields.io/badge/coverage-100%25-brightgreen.svg
   :alt: Coverage

.. only:: not latex

   |pypi| |python| |license| |coverage|

A high-performance Python library for sparse Graph Signal Processing (GSP) and Spectral Graph Wavelet Transforms (SGWT). This package leverages the ``CHOLMOD`` library for efficient sparse direct solvers, providing significant speedups over traditional dense or iterative methods for large-scale graph convolution.

Some of the key features include:


- **High-Performance Sparse Solvers**: Direct integration with the ``CHOLMOD`` library for optimized sparse Cholesky factorizations and linear system solves.
- **Generalized Graph Convolution**: Support for arbitrary spectral kernels via rational approximation (Kernel Fitting), polynomial approximation (Chebyshev), and standard analytical filters (low-pass, band-pass, high-pass).
- **Dynamic Topology Support**: Specialized routines for graphs with evolving structures, utilizing efficient rank-1 updates for real-time topology changes.
- **Resource-Aware Execution**: Context-managed memory allocation and workspace reuse to minimize overhead in high-throughput applications.
- **Integrated Graph Repository**: Built-in access to standardized graph Laplacians and signals from power systems and infrastructure networks.

For detailed usage, API reference, and theoretical background, please visit the `documentation website <https://sgwt.readthedocs.io/>`_.

Installation
------------

The ``sgwt`` package requires Python 3.7+ and is currently only compatible with **Windows** operating systems due to its reliance on a pre-compiled ``CHOLMOD`` library.

Install the latest stable release from `PyPI <https://pypi.org/project/sgwt/>`_:

.. code-block:: bash

    pip install sgwt

This command will also install the necessary dependencies (e.g., NumPy, SciPy).

Basic Example
-------------

Here is a quick example using a band-pass filter to an impulse signal on the synthetic Texas grid to get the wavelet function at three different scales.


.. code-block:: python

    import sgwt

    # Graph Laplacian
    L = sgwt.DELAY_TEXAS

    # Impulse at 600th Vertex
    X = sgwt.impulse(L, n=600)

    with sgwt.Convolve(L) as conv:
        
        # Wavelet at 3 scales
        Y = conv.bandpass(X, scales=[0.1, 1, 10])


The `examples/ <https://github.com/lukelowry/sgwt/tree/main/examples>`_ directory contains a comprehensive suite of demonstrations, also rendered in the `Examples <https://sgwt.readthedocs.io/en/stable/examples/static.html>`_ section of the documentation. Key applications include:

- **Static Filtering**: Basic low-pass, band-pass, and high-pass filtering on various graph sizes.
- **Dynamic Graphs**: Real-time topology updates, performance comparisons, and online stream processing.


Citation & Acknowledgements
---------------------------

If you use this library in your research, please cite it. The `GitHub repository <https://github.com/lukelowry/sgwt>`_ includes a ``CITATION.cff`` file that provides citation metadata. On GitHub, you can use the "Cite this repository" button on the sidebar to get the citation in your preferred format (including BibTeX).

For convenience, the BibTeX entry for the associated paper is:

.. code-block:: bibtex

    @inproceedings{lowery-sgwt-2026,
      title={Using Spectral Graph Wavelets to Analyze Large Power System Oscillation Modes},
      author={Lowery, Luke and Baek, Jongoh and Birchfield, Adam},
      year={2026}
    }

Luke Lowery developed this module during his PhD studies at Texas A&M University. You can learn more on his `research page <https://lukelowry.github.io/>`_ or view his publications on `Google Scholar <https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en>`_.

An alternative implementation in `Julia <https://github.com/lukelowry/SpectralGraphWavelet.jl>`_ is also available and leverages native SuiteSparse support.

- The core performance of this library relies on the ``CHOLMOD`` library from `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_, developed by Dr. Tim Davis at Texas A&M University.
- The graph laplacians used in the examples are derived from the `synthetic grid repository <https://electricgrids.engr.tamu.edu/electric-grid-test-cases/>`_, made available by Dr. Adam Birchfield at Texas A&M University.
