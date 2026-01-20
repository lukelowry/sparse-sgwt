API Reference
=============

This page provides a detailed reference for the classes and functions within the ``sgwt`` package.

Graph Convolution
-----------------

The core of the library consists of three specialized convolution engines designed for different use cases: static graphs, dynamic graphs with evolving topologies, and polynomial approximations.

.. toctree::
   :maxdepth: 2

   api_static
   api_dynamic
   api_cheb

Applications
------------

High-level application modules that leverage the underlying convolution engines for specific analysis tasks, such as joint spatial-temporal modal identification.

.. toctree::
   :maxdepth: 2

   api_sgma

Utility
-------

Helper functions for signal generation, spectral estimation, and the definition of spectral kernels (rational and polynomial).

.. toctree::
   :maxdepth: 2

   api_util
   api_functions