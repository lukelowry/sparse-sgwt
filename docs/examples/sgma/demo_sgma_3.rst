.. _demo_sgma_3:

Full Network Analysis
=====================

Demonstrates using ``analyze_many()`` to perform a comprehensive modal analysis across all buses in a network.

This example shows how to:

- Use ``analyze_many()`` to compute modes for every bus in the network.
- Generate a peak density heatmap to visualize system-wide modal patterns.
- Interpret the resulting wavelength-frequency plot.

.. literalinclude:: ../../../examples/demo_sgma_3.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Full Network SGMA Analysis

.. image:: /_static/images/demo_sgma_3.png
   :alt: Complete Network Modal Density Map
   :align: center

By calling ``analyze_many()`` without the ``buses`` argument, the analysis runs on all buses in the graph. The resulting heatmap visualizes the density of all detected peaks. Hotspots on the map indicate dominant modes that are present across many buses in the system.

- **Wavelength (y-axis)**: Larger wavelengths correspond to wide-area inter-area modes, while smaller wavelengths indicate local modes.
- **Frequency (x-axis)**: Identifies the temporal frequency of the oscillation in Hz.
