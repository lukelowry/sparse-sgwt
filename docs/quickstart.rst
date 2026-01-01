Quickstart
==========

Installation
------------

The package can be installed via pip:

.. code-block:: bash

    pip install sgwt

The package uses a compiled version of ``CHOLMOD``.

Basic Usage
-----------

Quick Start
~~~~~~~~~~~

For the quick-start example, we will find the response of a low-pass filter :math:`\phi` scaled by ``s`` to impulse :math:`\delta` at node :math:`n` over the graph ``L``. This is mathematically denoted by :math:`\phi_{n,s}=\delta_n*\phi_s`.

.. code-block:: python

    import sgwt

    # CSC Graph Laplacian
    L = sgwt.DELAY_TX

    # Impulse at Vertex n
    X = sgwt.impulse(L, n=...)

    # Discrete Scales
    s = np.logspace(...)

    # L -> Context of Convolution
    with Convolve(L) as conv:

        # Apply Low-Pass Filters
        Y = conv.lowpass(X, s)

The numpy arrays ``Y[i]`` correspond to a filtered signal ``X`` at the ``i-th`` scale.

The purpose of the context manager is to provide safe re-use of ``CHOLMOD`` workspace. While inside the context, the convolution procedure optimizes memory usage.

Underlying Graph
~~~~~~~~~~~~~~~~

The module has a small repository of built in graph laplacians that are useful for quick start examples.

.. code-block:: python

    L = sgwt.LENGTH_TX
    L = sgwt.IMPEDANCE_HAWAII
    L = sgwt.DELAY_USA

The user can also load any graph Laplacian so long it is in the ``csc_matrix`` format.

Input Signals
~~~~~~~~~~~~~

A real-valued time-vertex function :math:`X\in\mathbb{R}^{|N|\times|T|}` stored as a 2D numpy array in column-major ordering (i.e., fortran style) can be used. For example, an empty array meeting these specifications:

.. code-block:: python

    X = np.empty(
        shape=(nVert, nTime),
        order = 'F'
    )

Although, a ``(N,1)`` array can also be used.

Kernel Functions
~~~~~~~~~~~~~~~~

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

Dynamic Graphs
~~~~~~~~~~~~~~

In many real-world applications, such as power systems or sensor networks, the underlying graph topology is dynamic. Re-initializing the entire convolution context for every edge update is computationally prohibitive. This example demonstrates the use of ``DyConvolve`` to perform efficient, real-time signal filtering on an evolving graph by leveraging rank-1 updates to adapt existing factorizations on-the-fly.

.. code-block:: python

    from sgwt.dynamic import DyConvolve
    from sgwt import DELAY_USA as L

    poles = [10.0, 1.0, 0.1]

    with DyConvolve(L, poles) as conv:
        for f_t, event in stream:
            if event:
                conv.addbranch(*event) # Update topology
            W = conv.bandpass(f_t)     # Filter signal

At each iteration, the matrix ``W`` contains the column vectors which are the filtered version of ``f`` at the 'spatial' scale associated with each pole. So in this example ``W`` be a 3-column matrix representing the signal at three different scales.

See the :doc:`examples` page for more advanced usage scenarios.