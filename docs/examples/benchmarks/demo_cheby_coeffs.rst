Spectral Approximation Quality
==============================

This demo explores the relationship between polynomial order and approximation accuracy. It highlights the high order often required to match the precision of the direct rational solvers used in the core library.

.. literalinclude:: ../../../examples/demo_cheby_coeffs.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Chebyshev Order Comparison

.. image:: /_static/images/demo_cheby_coeffs.png
   :alt: Chebyshev Approximation Error
   :align: center