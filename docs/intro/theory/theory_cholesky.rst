Cholesky Implementation
=======================



From Polynomials to Linear Solves
---------------------------------

In classical Graph Signal Processing, filters are often approximated using **Chebyshev polynomials**. To apply a filter :math:`g(\mathbf{L})`, one computes a polynomial of the matrix :math:`\mathbf{L}`. While effective for small graphs, this method has a major drawback: it requires repeated matrix-vector multiplications. For sharp filters (like an ideal band-pass), the polynomial degree must be very high (often >50), making the calculation slow.

``sgwt`` utilizes **Rational Approximation (Kernel Fitting)** to bypass this bottleneck.

Instead of a polynomial, we approximate the filter kernel :math:`g(\lambda)` as a sum of partial fractions (poles and residues):

.. math::

    g(\lambda) \approx d + \sum_{k=1}^{K} \frac{r_k}{\lambda - p_k}

**The Computational Advantage**
This mathematical transformation changes the nature of the computation. Applying the filter no longer relies on high-degree matrix multiplication. Instead, it becomes a series of **linear system solves**:

.. math::

    (\mathbf{L} - p_k \mathbf{I}) \mathbf{z}_k = \mathbf{x}

Because power grid graphs are physically sparse (buses are only connected to a few neighbors), these linear systems are extremely sparse. We can solve them efficiently using **Sparse Cholesky Factorization**.

**Why this is better:**

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

    * :doc:`../usage/kernel_functions`