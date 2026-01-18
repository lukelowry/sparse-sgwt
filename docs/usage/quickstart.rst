Quickstart
==========

This guide demonstrates the fundamental workflow for filtering a signal on a graph using ``sgwt``. We will load a built-in graph, create a test signal, and apply a multi-scale band-pass filter.

.. code-block:: python

    from sgwt import Convolve, impulse
    from sgwt import DELAY_TEXAS as L

    # Load the graph Laplacian for the synthetic Texas power grid.
    # Create a test signal: a single impulse on vertex 600.
    signal = impulse(L, n=600)

    # Define the scales for the band-pass filter.
    scales = [0.1, 1.0, 10.0]

    # Use the Convolve context for efficient filtering.
    with Convolve(L) as conv:
        # Apply the filter.
        filtered_signals = conv.bandpass(signal, scales)

    # The output is a list of signals, one for each scale.
    print(f"Graph has {L.shape[0]} vertices.")
    print(f"Filtered signal for scale={scales[0]}: shape={filtered_signals[0].shape}")
    print(f"Filtered signal for scale={scales[1]}: shape={filtered_signals[1].shape}")
    print(f"Filtered signal for scale={scales[2]}: shape={filtered_signals[2].shape}")

What's Happening in This Example?
---------------------------------

1.  **Load the Graph and Signal**
    - ``from sgwt import DELAY_TEXAS as L``: We import a pre-built graph Laplacian for the **synthetic Texas** power grid. This matrix ``L`` defines the graph's structure.
    - ``signal = impulse(L, n=600)``: We create a simple test signal. It's a Dirac impulse, with a value of ``1.0`` on vertex 600 and ``0.0`` everywhere else. This is useful for seeing how a filter "spreads" energy across the graph.

2.  **Define Filter Parameters**
    - ``scales = [0.1, 1.0, 10.0]``: We specify the scales for our band-pass filter. In spectral graph filtering, the scale is inversely related to frequency. Smaller scales target high-frequency components (details, sharp changes), while larger scales target low-frequency components (smooth variations).

3.  **Perform the Convolution**
    - ``with Convolve(L) as conv:``: This is the core of the library. We create a :class:`~sgwt.Convolve` context. Upon entry, it performs an efficient one-time symbolic factorization of the graph Laplacian ``L``. This pre-computation makes all subsequent filtering operations extremely fast. The context also manages all the low-level memory required by the CHOLMOD backend.
    - ``filtered_signals = conv.bandpass(signal, scales)``: Inside the context, we call the :meth:`~sgwt.Convolve.bandpass` method. It solves the underlying linear systems to apply the filter kernel at each of the specified scales.

4.  **Inspect the Output**
    - The result, ``filtered_signals``, is a Python list where each element is a NumPy array corresponding to the filtered signal at one of the input scales. All arrays have the same shape as the input ``signal``.

.. seealso::
   * :doc:`input_signals` for signal formatting requirements.
   * :doc:`kernel_functions` for details on filter types.
   * :doc:`../theory/index` for the mathematical background.