Examples
========

The library comes with several examples demonstrating static and dynamic graph convolutions.

Static Graph Filtering
----------------------

These examples demonstrate basic filtering operations (Low-pass, Band-pass, High-pass) on various synthetic power grid networks.

**Basic Filtering (Texas Grid)**

Demonstrates low-pass, band-pass, and high-pass filtering on the synthetic Texas grid (~2k nodes).

.. literalinclude:: ../examples/demo_filters_1.py
   :language: python
   :caption: Basic Filtering on Texas Grid

**Filtering on East-West Grid**

Performs filtering operations on the larger synthetic East-West US grid (~65k nodes).

.. literalinclude:: ../examples/demo_filters_2.py
   :language: python
   :caption: Filtering on East-West Grid

**Filtering on USA Grid**

Scales up to the synthetic USA grid (~82k nodes) to demonstrate performance on larger networks.

.. literalinclude:: ../examples/demo_filters_3.py
   :language: python
   :caption: Filtering on USA Grid

**Self-Contained Example**

A standalone example of band-pass filtering that includes all necessary imports and setup in a single file.

.. literalinclude:: ../examples/demo_single_file.py
   :language: python
   :caption: Single File Demo

Advanced Usage
--------------------------

Examples demonstrating more advanced features like custom kernels and signal reconstruction.

**Vector Fitting Kernels**

Shows how to use Vector Fitting (VF) kernels to implement custom filter shapes, such as the Modified Morlet wavelet.

.. literalinclude:: ../examples/demo_vf.py
   :language: python
   :caption: Vector Fitting Demo

**Signal Reconstruction**

Demonstrates recovering signal values (e.g., geographic coordinates) from sparse measurements using graph convolution.

.. literalinclude:: ../examples/demo_recon.py
   :language: python
   :caption: Signal Reconstruction

**Signal Inpainting**

Reconstructs a smooth signal across the USA grid using only a small fraction (e.g., 0.1%) of known data points via iterative low-pass filtering.

.. literalinclude:: ../examples/demo_inpainting.py
   :language: python
   :caption: Signal Inpainting

Dynamic Graphs
--------------

Demonstrations of dynamic graph updates using ``DyConvolve``.

**Dynamic Topology Update**

Illustrates updating the graph topology (adding branches) on-the-fly without recomputing the entire decomposition.

.. literalinclude:: ../examples/demo_dynamic_topology.py
   :language: python
   :caption: Dynamic Topology Update

**Performance Comparison**

Compares the execution time and performance between static (``Convolve``) and dynamic (``DyConvolve``) convolution methods.

.. literalinclude:: ../examples/demo_dynamic_time.py
   :language: python
   :caption: Static vs Dynamic Performance

**Online Stream Processing**

Simulates an online processor handling a continuous stream of data and topology change events.

.. literalinclude:: ../examples/demo_dynamic_stream.py
   :language: python
   :caption: Dynamic Stream Processing