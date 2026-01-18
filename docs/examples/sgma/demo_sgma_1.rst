.. _demo_sgma_1:

Local Mode Identification
=========================

Demonstrates computing the SGMA spectrum at a single bus to identify dominant oscillatory modes in the wavelength-frequency domain.

This example shows how to:

- Initialize the ``SGMA`` engine with spatial scales and temporal frequencies.
- Compute the spectrum for a specific bus and time using ``sgma.spectrum()``.
- Extract and visualize dominant peaks using ``sgma.find_peaks()``.

.. literalinclude:: ../../../examples/demo_sgma_1.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Single Bus SGMA Analysis

.. image:: /_static/images/demo_sgma_1.png
   :alt: SGMA Spectrum at a Single Bus
   :align: center

The resulting contour plot shows the spectrum in the wavelength-frequency domain. The overlaid markers indicate the ``top_n`` most dominant oscillatory modes (peaks) identified at the target bus and time instant.
