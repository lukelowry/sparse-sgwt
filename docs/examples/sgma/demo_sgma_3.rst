.. _demo_sgma_3:

Full Network Analysis
========================

Demonstrates SGMA analysis across all buses in the network to create a comprehensive modal identification map.

This example shows how to:

- Compute SGMA transforms for every bus in the network
- Identify system-wide oscillatory modes
- Generate density plots showing modal participation across the entire system

.. literalinclude:: ../../../examples/demo_sgma_3.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Full Network SGMA Analysis

.. image:: /_static/images/demo_sgma_3.png
   :alt: Complete Network Modal Density Map
   :align: center

**Analysis Workflow:**

1. Load voltage magnitude signals (``np.abs(V)``) from the network
2. Initialize SGMA with appropriate spatial and temporal resolution
3. Compute transforms across all buses using ``find_system_wide_peaks()``
4. Generate peak density heatmaps for visualization

**Interpreting Results:**

- **Wavelength axis**: Larger wavelengths correspond to inter-area oscillation modes spanning the continent
- **Frequency axis**: Identifies the temporal oscillation frequency (Hz)
- **Density/Magnitude**: Indicates how many buses exhibit each mode

This comprehensive analysis reveals the complete modal structure of the power system, enabling identification of both local and inter-area oscillations.

**Computational Considerations:**

For very large networks (>80k buses), consider using a representative subset of buses rather than analyzing all buses to reduce computation time.
