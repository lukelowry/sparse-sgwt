Kernel Functions
================

This library supports two primary ways to define spectral graph filters: pre-defined analytical filters and fully custom filters using Vector Fitting.

Analytical Filters
------------------

For common signal processing tasks, three analytical filters are provided as methods on the convolution context.

- ``lowpass(X, scales)``: Smooths the signal by attenuating high-frequency components.
- ``bandpass(X, scales)``: Isolates components within a specific frequency band.
- ``highpass(X, scales)``: Emphasizes sharp transitions by attenuating low-frequency components.

The ``scales`` parameter is a list of floating-point values that control the cutoff frequencies of the filters. Larger scales correspond to lower frequencies.

.. code-block:: python

    from sgwt import Convolve, impulse
    from sgwt import DELAY_TEXAS as L

    X = impulse(L, n=600)
    scales = [0.1, 1.0, 10.0]

    with Convolve(L) as conv:
        # Returns a list of filtered signals, one for each scale
        Y_lp = conv.lowpass(X, scales)
        Y_bp = conv.bandpass(X, scales)
        Y_hp = conv.highpass(X, scales)

Custom Kernels via Vector Fitting
---------------------------------

For advanced use cases, you can define arbitrary filter shapes using the :class:`~sgwt.util.VFKernel` class. This approach uses a rational approximation (poles and residues) to model the desired frequency response, a technique known as Vector Fitting.

The library includes several pre-computed kernels. You can use them by passing the kernel object to ``convolve()``.

.. code-block:: python

    from sgwt import Convolve, impulse, VFKernel
    from sgwt import DELAY_USA as L
    from sgwt import MODIFIED_MORLET

    # Load a pre-computed Vector Fitting kernel
    K = VFKern.from_dict(MODIFIED_MORLET)
    X = impulse(L, n=35000)

    with Convolve(L) as conv:
        # Apply the custom kernel
        Y = conv.convolve(X, K)

This powerful feature allows you to implement specialized filters, such as the Modified Morlet wavelet used in the :doc:`/static/demo_vf` example.