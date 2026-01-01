Underlying Graph
==============

The module has a small repository of built in graph laplacians that are useful for quick start examples.

.. code-block:: python

    L = sgwt.LENGTH_TX
    L = sgwt.IMPEDANCE_HAWAII
    L = sgwt.DELAY_USA

The user can also load any graph Laplacian so long it is in the ``csc_matrix`` format.