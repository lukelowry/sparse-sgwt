3D Meshes for Visual Understanding
====================

This example demonstrates filtering on non-traditional graph structures, such as 3D meshes. We load the Stanford Bunny and a horse model, apply a band-pass filter to an impulse signal, and visualize how the wavelet propagates across the mesh surface in 3D.

The color of each point in the scatter plot corresponds to the magnitude of the wavelet coefficient at that vertex, showing the filter's spatial localization. This is a powerful way to analyze signals on complex, irregular domains.

.. literalinclude:: ../../examples/demo_mesh_wavelet.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Mesh Wavelet Analysis

.. image:: /_static/images/demo_mesh_wavelet_1.png
   :alt: Wavelet on Stanford Bunny
   :align: center

.. image:: /_static/images/demo_mesh_wavelet_2.png
   :alt: Wavelet on Horse Mesh
   :align: center