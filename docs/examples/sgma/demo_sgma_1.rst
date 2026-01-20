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


IEEE 39 Bus New England Case
----------------------------

The IEEE 39 bus New England ISO case demonstrates SGMA on a real-world power system topology. The signal is the complex voltage response to a balanced three-phase fault at Bus 16, occurring at :math:`t=0.5` seconds and cleared at :math:`t=0.7` seconds. The simulation uses a time step of 1 millisecond from :math:`t=0` to :math:`t=10` seconds.

SGMA Parameters
^^^^^^^^^^^^^^^

The analysis uses the following configuration:

**Graph and Analysis Point:**
  - Laplacian: ``LENGTH_NEISO`` (39 buses)
  - Target bus: 12
  - Target time: :math:`\tau = 1.5` seconds
  - Bandpass order: :math:`K = 1`

**Spatial Sampling:**
  - Number of scales: 150
  - Scale range: :math:`s \in [(0.1)^2, (100)^2]` (geometrically spaced)
  - Wavelength range: :math:`r = \sqrt{s} \in [0.1, 100.0]`

**Temporal Sampling:**
  - Number of frequencies: 200
  - Frequency range: :math:`f \in [0.01, 3.0]` Hz (linearly spaced)
  - Wavelet parameter: :math:`\omega_0 = 2\pi`

.. literalinclude:: ../../../examples/demo_sgma_1b.py
   :language: python
   :start-after: # DOC_START_CODE_EXCLUDE_IMPORTS
   :end-before: # DOC_END_CODE_EXCLUDE_PLOT
   :caption: IEEE 39 Bus SGMA Analysis

.. image:: /_static/images/demo_sgma_1b.png
   :alt: SGMA Spectrum for IEEE 39 Bus System
   :align: center

Identified Modes
^^^^^^^^^^^^^^^^

The ``find_modes`` function extracts the top 3 peaks from the spectrum:

.. code-block:: text

   ------------------------------------------------------------
   #   Freq (Hz)     Damping    Wavelength   Magnitude
   ------------------------------------------------------------
   1      0.4908      0.0000          4.91      0.0209
   2      1.0768      0.3920          4.48      0.0201
   3      1.9182      0.1638          4.27      0.0084
   ------------------------------------------------------------

Each mode is characterized by:
  - **Frequency (Hz)**: Oscillation rate extracted from the temporal frequency axis
  - **Damping**: Decay rate estimated from phase variation in the complex spectrum
  - **Wavelength**: :math:`r = \sqrt{s}` at the peak location in the spatial domain
  - **Magnitude**: Transform magnitude at the peak location