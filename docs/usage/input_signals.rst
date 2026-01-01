Input Signals
-------------

A real-valued time-vertex function :math:`X\in\mathbb{R}^{|N|\times|T|}` stored as a 2D numpy array in column-major ordering (i.e., fortran style) can be used. For example, an empty array meeting these specifications:

.. code-block:: python

    X = np.empty(
        shape=(nVert, nTime),
        order = 'F'
    )

Although, a ``(N,1)`` array can also be used.