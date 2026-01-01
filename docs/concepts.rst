Basic Concepts
==============

Underlying Graph
----------------

The module has a small repository of built in graph laplacians that are useful for quick start examples.

.. code-block:: python

    L = sgwt.LENGTH_TX
    L = sgwt.IMPEDANCE_HAWAII
    L = sgwt.DELAY_USA

The user can also load any graph Laplacian so long it is in the ``csc_matrix`` format.

Input Signals
-------------

A real-valued time-vertex function :math:`X\in\mathbb{R}^{|N|\times|T|}` stored as a 2D numpy array in column-major ordering (i.e., fortran style) can be used. For example, an empty array meeting these specifications:

.. code-block:: python

    X = np.empty(
        shape=(nVert, nTime),
        order = 'F'
    )

Although, a ``(N,1)`` array can also be used.

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