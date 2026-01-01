Quickstart
==========

Here is a simple example of performing a band-pass filter on a graph signal using the Texas power grid topology.

.. code-block:: python

    from sgwt import Convolve, impulse
    from sgwt import DELAY_TEXAS as L
    from sgwt import COORD_TEXAS as C

    # Create an impulse signal at 600-th vertex
    X = impulse(L, n=600)

    # Define scales for the filter
    scales = [0.1, 1, 10]

    # Initialize the convolution context
    with Convolve(L) as conv:

        # Apply band-pass filter
        Y = conv.bandpass(X, scales)
        
    # Y is the filtered signal coefficients