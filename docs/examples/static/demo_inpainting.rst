.. _demo_inpainting:

Signal Inpainting
=================

Reconstructs a smooth signal across the USA grid using only a small fraction (e.g., 0.1%) of known data points via iterative low-pass filtering.

.. literalinclude:: ../../../examples/demo_inpainting.py
   :language: python
   :caption: Signal Inpainting
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT

.. image:: /_static/images/inpainting_reconstruction.png
   :alt: Signal Inpainting Reconstruction Example