.. _demo_sgma_2:

Modal Analysis on a Bus Subset
==============================

Demonstrates using ``analyze_many()`` to efficiently identify oscillatory modes across a subset of buses.

This example shows how to:

- Analyze a random subset of buses to get a representative view of network-wide modes.
- Use the ``analyze_many()`` method for efficient batch processing.
- Visualize the combined peaks from all analyzed buses on a heatmap.

.. literalinclude:: ../../../examples/demo_sgma_2.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Analysis on a Subset of Buses

.. image:: /_static/images/demo_sgma_2.png
   :alt: Peak Density Heatmap Across Multiple Buses
   :align: center

The ``analyze_many()`` method is highly efficient because it computes the temporal transform ``V @ B`` only once and reuses it for each bus. The resulting heatmap shows all detected peaks, providing a picture of the dominant modes present in the selected subset of the network.
