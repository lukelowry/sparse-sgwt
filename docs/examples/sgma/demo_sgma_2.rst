.. _demo_sgma_2:

System-Wide Peak Detection
=============================

Demonstrates system-wide peak detection across multiple buses to identify common oscillatory modes throughout the network.

This example shows how to:

- Analyze SGMA transforms across a subset of randomly selected buses
- Identify common peaks using clustering algorithms
- Visualize the density of detected modes in the wavelength-frequency plane

.. literalinclude:: ../../../examples/demo_sgma_2.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: System-Wide Peak Detection

.. image:: /_static/images/demo_sgma_2.png
   :alt: Peak Density Heatmap Across Multiple Buses
   :align: center

**Key Features:**

- ``find_system_wide_peaks()``: Efficiently computes transforms across multiple buses by pre-computing the temporal wavelet matrix
- ``cluster_peaks``: Identifies dominant modes that appear consistently across the system
- ``subset_bus_indices``: Random sampling allows scalable analysis on large networks

The resulting peak heatmap shows the distribution of detected modes across all analyzed buses, revealing system-wide oscillatory patterns.

**Performance Note:**

The method pre-computes ``V @ B`` once and reuses it for all buses, significantly reducing computation time compared to individual transforms.
