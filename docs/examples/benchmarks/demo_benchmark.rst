Benchmark: Analytical Solver Performance
========================================

This benchmark visualizes the performance characteristics of the analytical graph wavelet solvers across various dimensions: graph size, signal count, scale count, and filter order.

The four panels show:

- **Graph Size Scaling**: Execution time vs. number of edges for both ``Convolve`` (Static) and ``DyConvolve`` (Dynamic) solvers across lowpass, bandpass, and highpass filters.
- **Signal Count Scaling**: How execution time scales with the number of input signals.
- **Scale Count Scaling**: How execution time scales with the number of wavelet scales.
- **Filter Order Scaling**: How bandpass filter order affects execution time.

.. image:: /_static/images/demo_benchmark.png
   :alt: Analytical Solver Performance Benchmarks
   :align: center
