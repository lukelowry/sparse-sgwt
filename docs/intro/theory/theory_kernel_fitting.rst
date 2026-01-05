Kernel Fitting
==============

The kernel fitting representation is more generally a vector fitted function, a simple pole expansion of the form:

.. math::

   g_a(\mathbf{\Lambda})\approx 
        d_aI + e_a\mathbf{\Lambda}
        + \sum_{q\in Q}\dfrac{r_{q,a}}{\mathbf{\Lambda}+qI} 

An iterative pole reallocation procedure is used to converge to a reduced order model. The convolution of some function :math:`\mathbf{f}*g_a` is computed using the Cholesky decomposition and memory efficient re-factors.

.. seealso::
   :doc:`../library/library_json`
      For details on the file format used to store these kernels.
   :doc:`theory_cholesky`
      For information on the underlying Cholesky-based solver implementation.