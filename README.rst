Sparse SGWT
====================================

.. image:: https://badge.fury.io/py/sgwt.svg
    :target: https://badge.fury.io/py/sgwt
    :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/sgwt.svg
    :target: https://pypi.org/project/sgwt/
    :alt: Python Version

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: ./LICENSE.md
    :alt: License

.. image:: https://img.shields.io/badge/Changelog-Keep%20a%20Changelog-blue.svg
    :target: ./CHANGELOG.rst
    :alt: Changelog


A high-performance Python library for sparse Graph Signal Processing (GSP) and Spectral Graph Wavelet Transforms (SGWT). This package leverages the ``CHOLMOD`` library for efficient sparse direct solvers, providing significant speedups over traditional dense or iterative methods for large-scale graph convolution.

Key Features
------------

- **High-Performance Sparse Solvers**: Direct integration with the ``CHOLMOD`` library for optimized sparse Cholesky factorizations and linear system solves.
- **Generalized Graph Convolution**: Support for arbitrary spectral kernels via rational approximation (*Kernel Fitting*) and standard analytical filters (low-pass, band-pass, high-pass).
- **Dynamic Topology Support**: Specialized routines for graphs with evolving structures, utilizing efficient rank-1 updates for real-time topology changes.
- **Resource-Aware Execution**: Context-managed memory allocation and workspace reuse to minimize overhead in high-throughput applications.
- **Integrated Graph Repository**: Built-in access to standardized graph Laplacians and signals from power systems and infrastructure networks.

Installation
------------

You can install ``sgwt`` from PyPI:

.. code-block:: bash

    pip install sgwt

Usage Example
-------------

Here is a quick example of applying a band-pass filter to an impulse signal on the built-in Texas grid Laplacian.

.. code-block:: python

    import sgwt

    # 1. Load a built-in graph Laplacian, which defines the graph's topology.
    L = sgwt.DELAY_TEXAS

    # 2. Create a vertex-domain signal. Here, a Dirac impulse on the 600th node.
    #    The `impulse` helper function ensures the required column-major memory order.
    signal = sgwt.impulse(L, n=600)

    # 3. Use the static convolution context manager. This performs a one-time
    #    symbolic factorization of the Laplacian for efficient repeated solves.
    with sgwt.Convolve(L) as conv:
        # 4. Apply an analytical band-pass filter. The scale parameter controls
        #    the filter's center frequency; larger scales target lower frequencies.
        filtered_signals = conv.bandpass(signal, scales=[10.0])

    # 5. The result is a list of filtered signals, one for each input scale.
    result = filtered_signals[0]

    print(f"Graph has {L.shape[0]} nodes.")
    print(f"Signal on node 600, shape: {signal.shape}")
    print(f"Filtered signal shape: {result.shape}")

Documentation
-------------

For detailed usage, API reference, and theoretical background, please visit the `documentation website <https://sgwt.readthedocs.io/en/stable/>`_.

Citation
--------

If you use this library in your research, please cite it. The `GitHub repository <https://github.com/lukelowry/sparse-sgwt>`_ includes a ``CITATION.cff`` file that provides citation metadata. On GitHub, you can use the "Cite this repository" button on the sidebar to get the citation in your preferred format (including BibTeX).

For convenience, the BibTeX entry for the associated paper is:

.. code-block:: bibtex

    @inproceedings{lowery-sgwt-2026,
      title={Using Spectral Graph Wavelets to Analyze Large Power System Oscillation Modes},
      author={Lowery, Luke and Baek, Jongoh and Birchfield, Adam},
      year={2026}
    }

Author
------

This module was developed by Luke Lowery during his PhD studies at Texas A&M University. You can learn more on his `research page <https://lukelowry.github.io/>`_ or view his publications on `Google Scholar <https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en>`_.

An alternative implementation in `Julia <https://github.com/lukelowry/SpectralGraphWavelet.jl>`_ is also available, which takes advantage of native SuiteSparse support.

Acknowledgements
----------------

- The core performance of this library relies on the ``CHOLMOD`` library from `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_, developed by Dr. Tim Davis at Texas A&M University.
- The graph laplacians used in the examples are derived from the `synthetic grid repository <https://electricgrids.engr.tamu.edu/electric-grid-test-cases/>`_, made available by Dr. Adam Birchfield at Texas A&M University.