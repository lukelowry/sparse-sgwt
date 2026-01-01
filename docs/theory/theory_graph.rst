Graph Laplacian
==============

This section outlines the mathematical foundations of the Sparse SGWT library.

Let an undirected graph :math:`\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathbf{A}, \mathbf{w}\}` be defined by a set of verticies :math:`|\mathcal{V}|=N` and a set of edges :math:`\mathcal{E}` which are related by the arc-node incident matrix :math:`\mathbf{A}\in\mathbb{R}^{|\mathcal{E}|\times |\mathcal{V}|}` and the vector of branch weights :math:`\mathbf{w}\in\mathbb{R}^{|\mathcal{E}|}`. A *vertex domain* function on the graph :math:`f:\mathcal{V}\to\mathbb {R}` can be written as a vector :math:`\mathbf{f}\in \mathbb{R}^N`, whose :math:`i^{th}` element corresonds to the evaluation of :math:`f` at the :math:`i^{th}` vertex. The *graph Laplacian* is denoted by :math:`\mathbf{L}\in\mathbb{R}^{N\times N}`, a discrete analogue of the continuous Laplace-Beltrami operator.

.. math::

   \mathbf{L} := \mathbf{A}^\top \text{diag}(\mathbf{w}) \mathbf{A}

When working with a physical distance as a distance metric: for each branch of length :math:`\ell`, the corresponding branch weight is assigned :math:`\ell^{-2}` so that the eigenvalues of the Laplacian represent the squared spatial frequency :math:`\lambda = k^2`, where :math:`k` is the wavenumber. When we apply the SGWT to this graph, each scale :math:`a\in\mathcal{A}` is physically meaningful, corresponding to a squared *pseudo-wavelength*, :math:`r\in \mathbb{R}`, where :math:`a=r^2` defines this mapping.
