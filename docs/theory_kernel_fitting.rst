Kernel Fitting
==============

The kernel fitting representation is more generally a vector fitted function, a simple pole expansion of the form:

.. math::

   g_a(\mathbf{\Lambda})\approx 
        d_aI + e_a\mathbf{\Lambda}
        + \sum_{q\in Q}\dfrac{r_{q,a}}{\mathbf{\Lambda}+qI} 

An iterative pole reallocation procedure is used to converge to a reduced order model. The convolution of some function :math:`\mathbf{f}*g_a` is computed using the Cholesky decomposition and memory efficient re-factors.

An example of an appropriate format of the rational expansion:

.. code-block:: json

    {
        "nfunc": "N",
        "d": ["d0", "d1", "...", "dN"],
        "npoles": "M",
        "poles": [
            {
                "q": "q0", 
                "r": ["r0", "r1", "...", "rN"]
            }
        ]
    }