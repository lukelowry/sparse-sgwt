3D Meshes for Visual Understanding
===================================

Spectral graph wavelets extend naturally to 3D mesh surfaces, where the graph Laplacian captures the intrinsic geometry of the manifold. This example demonstrates wavelet localization on three mesh models of varying complexity: the Stanford Bunny, a horse, and a cortical brain surface.

For each mesh, we place an impulse signal at a single vertex and apply a band-pass filter. The ``scale`` parameter controls the center frequency of the filter (larger scales target lower frequencies), while the ``order`` parameter determines filter sharpness by repeated application of the operator. The resulting wavelet coefficients reveal how the filter's frequency response translates to spatial localization on the surface. Vertices are colored by coefficient magnitude, with the diverging colormap distinguishing positive and negative values.


Stanford Bunny
--------------

The Stanford Bunny (35,947 vertices) serves as a standard benchmark in computer graphics. With a low filter order (4) at scale 200, the wavelet produces a broad, smooth response centered on vertex 15,000. The relatively gentle spectral rolloff results in gradual spatial decay across the mesh surface.

.. image:: /_static/images/demo_mesh_wavelet_1.png
   :alt: Wavelet on Stanford Bunny
   :align: center

Horse Model
-----------

The horse mesh (48,485 vertices) presents an elongated geometry with thin limbs and a complex torso. Using a higher filter order (50) at scale 60 yields a more sharply localized wavelet. The filter response, centered at vertex 28,000, demonstrates how wavelets adapt to the local surface geometry while respecting the mesh connectivity.

.. image:: /_static/images/demo_mesh_wavelet_2.png
   :alt: Wavelet on Horse Mesh
   :align: center

Cortical Surface
----------------

The left hemisphere cortical surface (~150k vertices) represents a neuroimaging application where spectral methods have direct relevance. The highly folded geometry of the brain creates a graph Laplacian that encodes both spatial proximity and cortical folding patterns. The wavelet at scale 200 with order 30 shows localization that follows the cortical ribbon rather than Euclidean distance, which could feasible be useful for multi-scale analysis of brain activity or morphometry.

.. image:: /_static/images/demo_mesh_wavelet_3.png
   :alt: Wavelet on Cortical Surface
   :align: center

Engine
----------------

This engine model has ~1.4 million verticies.


.. image:: /_static/images/demo_mesh_wavelet_4.png
   :alt: Wavelet on Engine
   :align: center