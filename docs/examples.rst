Examples
========

The library comes with several examples demonstrating static and dynamic graph convolutions.

Static Graph Filtering
----------------------

Basic filtering (Low-pass, Band-pass, High-pass) on the Texas grid.

.. literalinclude:: ../examples/demo_filters_1.py
   :language: python
   :caption: Basic Filtering Demo

Dynamic Topology
----------------

Demonstration of dynamic graph updates using ``DyConvolve``. This allows adding branches to the graph topology efficiently.

.. literalinclude:: ../examples/demo_dynamic_topology.py
   :language: python
   :caption: Dynamic Topology Update

Signal Inpainting
-----------------

Reconstructs a smooth signal across the USA grid using only a small fraction of known data points.

.. literalinclude:: ../examples/demo_inpainting.py
   :language: python
   :caption: Signal Inpainting