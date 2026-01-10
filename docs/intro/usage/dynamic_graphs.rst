Dynamic Graphs
==============

In many real-world applications, such as power systems or communication networks, the underlying graph topology is not static. Edges (e.g., transmission lines or communication links) can be added or removed over time. Re-calculating the entire graph factorization for every small change is computationally expensive and not feasible for real-time applications.

The :class:`~sgwt.DyConvolve` context is specifically designed for this scenario. It provides high-performance filtering on evolving graphs by leveraging efficient rank-1 updates to the factorization.

The Key Trade-off
-----------------

To achieve this performance, ``DyConvolve`` requires that the filter poles (or scales) are defined upfront and remain constant. The context pre-factors the graph for each pole, allowing subsequent topology updates and convolutions to be extremely fast.

Usage Example
-------------

Here is a simple example demonstrating how to initialize the context, update the topology, and see the effect on a filtered signal.

.. code-block:: python

    from sgwt import DyConvolve, impulse
    from sgwt import DELAY_TEXAS as L

    # 1. Define poles upfront (e.g., for a band-pass filter at scale=0.1)
    poles = [1 / 0.1]

    # 2. Create an impulse signal at a specific vertex
    X  = impulse(L, n=1200)

    # 3. Use the DyConvolve context
    with DyConvolve(L, poles) as conv:
        # Filter the signal on the original graph
        Y_before = conv.bandpass(X)

        # 4. Introduce a topology change: add a new edge
        # This connects vertex 1200 and 600 with a high weight.
        conv.addbranch(1200, 600, w=100.0)

        # 5. Filter the same signal again on the modified graph
        Y_after = conv.bandpass(X)

In this example, ``Y_after`` will show the impulse signal having propagated from vertex 1200 to vertex 600, which would not have occurred in ``Y_before``.

For a more advanced simulation of a real-time data stream with topology events, see the :doc:`/dynamic/demo_dynamic_stream` example.