Chebyshev Approximation Reference
=================================

This example demonstrates how to use ``ChebConvolve`` to approximate a standard band-pass filter. It serves as a reference to verify that the polynomial approximation converges to the same spatial result as the analytical solver.

.. literalinclude:: ../../examples/demo_cheby_convolve.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Chebyshev Approximation vs. Analytical Solver

.. image:: /_static/images/demo_cheby_convolve.png
   :alt: Chebyshev Convolution Comparison
   :align: center