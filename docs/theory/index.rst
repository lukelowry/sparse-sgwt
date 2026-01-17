Theory
===========

This section provides the mathematical and algorithmic foundations of the Sparse SGWT library. 
It covers the definition of the graph Laplacian as a discrete operator, the analytical forms 
of spectral filters, the rational approximation of custom kernels via Vector Fitting, and 
the implementation details of the high-performance Cholesky-based solvers that enable 
scalable graph convolutions.

.. toctree::
   :maxdepth: 2

   theory_graph
   theory_analytical
   theory_kernel_fitting