Signals
==============

The primary input for all filtering operations is a signal defined on the vertices of the graph. This signal should be provided as a NumPy array.

Format and Memory Layout
------------------------

The signal array ``X`` can be either:

- **1D**: Shape ``(n_vertices,)`` for a single signal
- **2D**: Shape ``(n_vertices, n_timesteps)`` for multiple signals

where ``n_vertices`` is the number of vertices (nodes) in the graph, matching ``L.shape[0]``, and ``n_timesteps`` is the number of independent signals or time samples to be processed concurrently.

When a 1D signal is passed, the output will also be 1D (squeezed accordingly).

For optimal performance with 2D signals, it is highly recommended to create the signal array with **column-major (Fortran) ordering** by specifying ``order='F'``. This memory layout aligns with the underlying C-based ``CHOLMOD`` library, avoiding costly data re-ordering during computation.

.. code-block:: python

    import numpy as np
    from sgwt import DELAY_USA as L

    n_vertices = L.shape[0]

    # 1D signal (single signal)
    X_1d = np.random.randn(n_vertices)

    # 2D signal (multiple signals)
    n_signals = 10
    X_2d = np.random.randn(n_vertices, n_signals).astype(np.float64)
    X_2d = np.asfortranarray(X_2d)

    # Verify the memory order for 2D
    assert X_2d.flags['F_CONTIGUOUS']

.. seealso::
   :doc:`../theory/theory_graph`
      For the definition of vertex-domain functions.

Creating Test Signals with ``impulse``
--------------------------------------

For testing and examples, the library provides the :func:`~sgwt.util.impulse` helper function to quickly generate a Dirac impulse (a value of `1` at one vertex and `0` everywhere else).

.. code-block:: python

    from sgwt import impulse
    from sgwt import DELAY_TEXAS as L

    # Create a signal with a single impulse at vertex 600
    X_impulse = impulse(L, n=600)

    print(X_impulse.shape)
    # Output: (2000, 1)