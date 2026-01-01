Quickstart
==========

Installation
------------

To install the package, clone the repository and install via pip:

.. code-block:: bash

   git clone https://github.com/yourusername/sparse-sgwt.git
   cd sparse-sgwt
   pip install .

Basic Usage
-----------

Here is a simple example of performing a band-pass filter on a graph signal using the Texas power grid topology.

.. code-block:: python

    from sgwt import Convolve, impulse
    from sgwt import DELAY_TEXAS as L
    from sgwt import COORD_TEXAS as C

    # Create an impulse signal on the graph
    X = impulse(L, n=600)

    # Define scales for the filter
    scales = [0.1]

    # Initialize the convolution context
    with Convolve(L) as conv:
        # Apply band-pass filter
        Y = conv.bandpass(X, scales)
        
    # Y is now the filtered signal coefficients

See the :doc:`examples` page for more advanced usage scenarios.