Underlying Graph
================

The foundational element for all operations in this library is the **Graph Laplacian**. It is a matrix that mathematically represents the structure and connectivity of your graph.

Format
------

The Laplacian matrix ``L`` must be provided as a ``scipy.sparse.csc_matrix``. This sparse format is essential for the high-performance memory and computation management provided by the underlying ``CHOLMOD`` library. The matrix must be square and symmetric.

Built-in Graph Repository
-------------------------

For convenience and reproducibility, ``sgwt`` includes a repository of pre-built graph Laplacians from common power system test cases. These can be imported directly.

The naming convention is ``METRIC_REGION``:

- **METRIC**: The physical property used for edge weights (e.g., ``DELAY``, ``IMPEDANCE``, ``LENGTH``).
- **REGION**: The geographical or network name (e.g., ``TEXAS``, ``USA``, ``WECC``).

.. code-block:: python

    import sgwt

    # Load the Laplacian for the synthetic USA grid
    # where edge weights are based on phase delay.
    L_usa = sgwt.DELAY_USA

    print(type(L_usa))
    # Output: <class 'scipy.sparse.csc.csc_matrix'>

    print(L_usa.shape)
    # Output: (82223, 82223)
