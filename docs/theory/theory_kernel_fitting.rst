Kernel Fitting
==============



From Polynomials to Linear Solves
---------------------------------

In classical Graph Signal Processing, filters are often approximated using **Chebyshev polynomials**. To apply a filter :math:`g(\mathbf{L})`, one computes a polynomial of the matrix :math:`\mathbf{L}`. While effective for small graphs, this method has a major drawback: it requires repeated matrix-vector multiplications. For sharp filters (like an ideal band-pass), the polynomial degree must be very high (often >50), making the calculation slow.

``sgwt`` utilizes **Rational Approximation (Kernel Fitting)** to bypass this bottleneck.

Instead of a polynomial, we approximate the filter kernel :math:`g(\lambda)` as a sum of partial fractions (poles and residues):

.. math::

    g(\lambda) \approx d + \sum_{k=1}^{K} \frac{r_k}{\lambda + q_k}

**The Computational Advantage**
This mathematical transformation changes the nature of the computation. Applying the filter no longer relies on high-degree matrix multiplication. Instead, it becomes a series of **linear system solves**:

.. math::

    (\mathbf{L} + q_k \mathbf{I}) \mathbf{y}_k = \mathbf{x}

Because power grid graphs are physically sparse (buses are only connected to a few neighbors), these linear systems are extremely sparse. We can solve them efficiently using **Sparse Cholesky Factorization**. This is better the polynomial approaches, due to:

1.  **Speed:** For large, sparse matrices, a Cholesky solve is orders of magnitude faster than the dense operations required by high-degree polynomials.
2.  **Precision:** Rational functions can approximate sharp transitions (like step functions) with far fewer terms than polynomials.

This architecture allows ``sgwt`` to perform complex filtering operations on grids with 80,000+ nodes in seconds.

For the context of large-scale power systems, this **direct solve approach** is preferred to iterative approximations.  Time-varying graph signals must be processed with maximum memory efficiency to ensure scalability, especially if the process is running online.

The ``cholmod_solve2`` and ``updown`` functions serve as the primary engine, alongside various design choices that accelerate graph convolutions.

.. seealso::
    The ``updown`` functionality is exposed through the dynamic convolution context,
    allowing for efficient updates to the graph topology.

    * :meth:`~sgwt.DyConvolve.addbranch`

    For more on the high-level concepts of graph filtering:

    * :doc:`../overview/usage/kernel_functions`

The kernel fitting representation is more generally a vector fitted function, a simple pole expansion of the form:

.. math:: 

   \mathbf{g}(\lambda) \approx \mathbf{d} + \mathbf{e}\lambda + \sum_{k=1}^{M} \frac{\mathbf{r}_k}{\lambda + q_k}

An iterative pole reallocation procedure is used to converge to a reduced order model. The convolution of some function :math:`\mathbf{f}*g_a` is computed using the Cholesky decomposition and memory efficient re-factors.

.. seealso::
   :doc:`../library/json`
      For details on the file format used to store these kernels.