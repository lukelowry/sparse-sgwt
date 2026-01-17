.. _demo_sgma_1:

Single Bus SGMA Transform
============================

Demonstrates computing the SGMA transform at a single bus location to identify dominant oscillatory modes in the wavenumber-frequency domain.

This example shows how to:

- Initialize the SGMA engine with spatial scales and temporal frequencies
- Compute the joint spatial-temporal wavelet transform for a specific bus
- Extract and visualize dominant peaks in the transform spectrum

.. literalinclude:: ../../../examples/demo_sgma_1.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Single Bus SGMA Analysis

.. image:: /_static/images/demo_sgma_1.png
   :alt: SGMA Transform Spectrum at Single Bus
   :align: center

**Key Parameters:**

- ``spatial_scales``: Logarithmically spaced values covering the range of expected wavelengths
- ``temporal_freqs``: Frequency range of interest (e.g., 0.02-2.0 Hz for power system oscillations)
- ``time_target``: Time instant to center the temporal wavelet
- ``order``: Order of the spatial bandpass filter (higher values provide sharper frequency localization)
- ``w0``: Central frequency of the temporal wavelet (typically 2π)

The transform produces a 2D spectrum in the wavelength-frequency domain, where peaks indicate dominant oscillatory modes.
