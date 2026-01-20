Benchmark: Chebyshev vs. Direct Solvers
=======================================

This benchmark demonstrates the performance trade-offs between polynomial approximation and sparse direct solvers. In most cases, the optimized ``DyConvolve`` solver outperforms high-order Chebyshev approximations on large sparse graphs.

.. literalinclude:: ../../../examples/demo_cheby_time.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Chebyshev vs. Analytical Timing

.. image:: /_static/images/demo_cheby_time.png
   :alt: Chebyshev Timing Benchmark
   :align: center