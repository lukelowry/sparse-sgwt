.. _demo_sgma_1:

Local Mode Identification
=========================

Demonstrates computing the SGMA spectrum at a single bus to identify dominant
oscillatory modes in the wavelength-frequency domain.

The Joint Spectrum
------------------

For a target bus :math:`n` and time :math:`\tau`, the SGMA computes the joint
wavelet transform:

.. math::

    m_{n,\tau}(\Lambda \times S) \approx L_n X R_\tau

The resulting spectrum :math:`m_{n,\tau}` reveals the energy distribution across
spatial wavelengths (related to :math:`\sqrt{s}`) and temporal frequencies.
Peaks in this 2D spectrum correspond to dominant oscillatory modes.

This example shows how to:

- Initialize the ``SGMA`` engine with spatial scales and temporal frequencies.
- Compute the spectrum for a specific bus and time using ``sgma.spectrum()``.
- Extract modes with ``sgma.find_modes()`` to obtain frequency, damping, and wavelength.

.. literalinclude:: ../../../examples/demo_sgma_1.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: Single Bus SGMA Analysis

.. image:: /_static/images/demo_sgma_1.png
   :alt: SGMA Spectrum at a Single Bus
   :align: center

The contour plot shows the spectrum magnitude in the wavelength-frequency domain.
The overlaid markers indicate the ``top_n`` most dominant oscillatory modes (peaks)
identified at the target bus and time instant. Each peak provides:

- **Wavelength** :math:`r = \sqrt{s}`: spatial extent of the mode
- **Frequency** :math:`f_0`: oscillation rate in Hz
- **Damping** :math:`\zeta`: decay rate estimated from phase slope
