Sparse Graph Signal Processing (GSP)
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

A high-performance Python library for sparse Graph Signal Processing (GSP) and Spectral Graph Wavelet Transforms (SGWT). This package leverages the ``CHOLMOD`` library for efficient sparse direct solvers, providing significant speedups over traditional dense or iterative methods for large-scale graph convolution.

Key Features
------------

- **High-Performance Sparse Solvers**: Direct integration with the ``CHOLMOD`` library for optimized sparse Cholesky factorizations and linear system solves.
- **Generalized Graph Convolution**: Support for arbitrary spectral kernels via rational approximation (*Kernel Fitting*) and standard analytical filters (low-pass, band-pass, high-pass).
- **Dynamic Topology Support**: Specialized routines for graphs with evolving structures, utilizing efficient rank-1 updates for real-time topology changes.
- **Resource-Aware Execution**: Context-managed memory allocation and workspace reuse to minimize overhead in high-throughput applications.
- **Integrated Graph Repository**: Built-in access to standardized graph Laplacians and signals from power systems and infrastructure networks.

Other
-----

This module is also implemented 
in `Julia <https://github.com/lukelowry/SpectralGraphWavelet.jl>`_ which takes advantage of 
the native SuiteSparse support. Although, it is not as well documented or tested.

References
~~~~~~~~~~

The ``CHOLMOD`` library of `SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_ was developed by Dr. Tim Davis at Texas A&M University.

The graph laplacians used in the examples are derived from the `synthetic grid repository <https://electricgrids.engr.tamu.edu/electric-grid-test-cases/>`_, available thanks to the research of Dr. Adam Birchfield at Texas A&M University.
- Birchfield, Adam B. et al. “Grid Structural Characteristics as Validation Criteria for Synthetic Networks”. In: IEEE Trans. on Power Sys. 32.4 (2017)

The theoretical work of this module is derived in part from this `paper <https://scholarspace.manoa.hawaii.edu/items/3f08d29d-db06-41d5-b235-2ee549bd198b>`_ nominated for best paper at HICSS-59.
- Lowery, Luke, Jongoh Baek, and Adam Birchfield. "Using Spectral Graph Wavelets to Analyze Large Power System Oscillation Modes." (2026)