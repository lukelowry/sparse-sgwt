Benchmark: Graph Size Scaling
=============================

This benchmark shows how execution time scales with graph size (number of edges) for both ``Convolve`` (Static) and ``DyConvolve`` (Dynamic) solvers across lowpass, bandpass, and highpass filters.

.. literalinclude:: ../../../examples/demo_benchmark_a.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Graph Size Scaling Benchmark

.. image:: /_static/images/demo_benchmark_a.png
   :alt: Graph Size Scaling Benchmark
   :align: center
   :width: 80%
