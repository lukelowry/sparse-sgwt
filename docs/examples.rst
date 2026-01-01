Examples
========

The library comes with several examples demonstrating static and dynamic graph convolutions.

Basic Usage
-----------

These examples demonstrate basic filtering operations (Low-pass, Band-pass, High-pass) on various synthetic power grid networks.

**Basic Filtering (Texas Grid)**

.. literalinclude:: ../examples/demo_filters_1.py
   :language: python
   :caption: Basic Filtering on Texas Grid

**Filtering on East-West Grid**

.. literalinclude:: ../examples/demo_filters_2.py
   :language: python
   :caption: Filtering on East-West Grid

**Filtering on USA Grid**

.. literalinclude:: ../examples/demo_filters_3.py
   :language: python
   :caption: Filtering on USA Grid

**Self-Contained Example**

A standalone example of band-pass filtering.

.. literalinclude:: ../examples/demo_single_file.py
   :language: python
   :caption: Single File Demo

Advanced Graph Convolution
--------------------------

Examples demonstrating more advanced features like custom kernels and signal reconstruction.

**Vector Fitting Kernels**

Using Vector Fitting (VF) kernels for custom filter shapes (e.g., Modified Morlet).

.. literalinclude:: ../examples/demo_vf.py
   :language: python
   :caption: Vector Fitting Demo

**Signal Reconstruction**

Recovering signal values (e.g., coordinates) from sparse measurements.

.. literalinclude:: ../examples/demo_recon.py
   :language: python
   :caption: Signal Reconstruction

**Signal Inpainting**

Reconstructs a smooth signal across the USA grid using only a small fraction of known data points.

.. literalinclude:: ../examples/demo_inpainting.py
   :language: python
   :caption: Signal Inpainting

Dynamic Graphs
--------------

Demonstrations of dynamic graph updates using ``DyConvolve``.

**Dynamic Topology Update**

Updating graph topology (adding branches) on-the-fly.

.. literalinclude:: ../examples/demo_dynamic_topology.py
   :language: python
   :caption: Dynamic Topology Update

**Performance Comparison**

Comparing performance between static (``Convolve``) and dynamic (``DyConvolve``) convolution methods.

.. literalinclude:: ../examples/demo_dynamic_time.py
   :language: python
   :caption: Static vs Dynamic Performance

**Online Stream Processing**

Simulating an online processor handling a stream of data and topology events.

.. literalinclude:: ../examples/demo_dynamic_stream.py
   :language: python
   :caption: Dynamic Stream Processing

Utilities
---------

Helper functions used in the examples.

.. literalinclude:: ../examples/demo_plot.py
   :language: python
   :caption: Plotting Utilities