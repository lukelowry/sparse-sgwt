Kernel Functions
----------------

There are three convenience analytical filters available.

.. code-block:: python

    with Convolve(L) as conv:
        Y = conv.lowpass(X, s)
        Y = conv.bandpass(X, s)
        Y = conv.highpass(X, s)

For more advanced functionality, the convolution is generalized using kernel fitting. Single Function kernels include ``MEXICAN_HAT``, ``MODIFIED_MORLET``, ``SHANNON``, and more.

The convolutional kernel ``F`` can be a vector function, meaning multiple filters can be applied concurrently (i.e., an orthogonal kernel to generate the wavelet coefficients ``SGWT``) This kernel will be available soon.

.. code-block:: python

    with Convolve(L) as conv:
        Y = conv(X, F)

Same as before, the convolution is simply performed on our signal ``X`` by first defining ``L`` as the convolution context.