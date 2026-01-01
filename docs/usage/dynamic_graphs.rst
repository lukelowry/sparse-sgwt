Dynamic Graphs
--------------

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