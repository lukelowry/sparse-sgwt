Graph Laplacian
===============

This section outlines the mathematical foundations of the Sparse SGWT library.

Let an undirected graph :math:`\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathbf{A}, \mathbf{w}\}` be defined by a set of vertices :math:`|\mathcal{V}|=N` and a set of edges :math:`\mathcal{E}` which are related by the arc-node incidence matrix :math:`\mathbf{A}\in\mathbb{R}^{|\mathcal{E}|\times |\mathcal{V}|}` and the vector of branch weights :math:`\mathbf{w}\in\mathbb{R}^{|\mathcal{E}|}`.

A *vertex domain* function on the graph :math:`f:\mathcal{V}\to\mathbb {R}` can be written as a vector :math:`\mathbf{f}\in \mathbb{R}^N`, whose :math:`i^{th}` element corresponds to the evaluation of :math:`f` at the :math:`i^{th}` vertex.

The **Graph Laplacian** is denoted by :math:`\mathbf{L}\in\mathbb{R}^{N\times N}`, a discrete analogue of the continuous Laplace-Beltrami operator:

.. math::

   \mathbf{L} := \mathbf{A}^\top \text{diag}(\mathbf{w}) \mathbf{A}

Alternatively, this can be viewed element-wise as :math:`\mathbf{L} = \mathbf{D} - \mathbf{W}`, where :math:`\mathbf{D}` is the degree matrix and :math:`\mathbf{W}` is the weighted adjacency matrix.

The Physics of the Weights
--------------------------

While any symmetric matrix can serve as a Laplacian, ``sgwt`` utilizes specific weighting schemes to ensure the graph spectral domain aligns with the physical behavior of power grids. 

[Image of standing wave harmonics]


When working with a physical distance metric, the library utilizes **Inverse Squared Length Weighting**. For each branch of length :math:`\ell_{ij}`, the corresponding branch weight is assigned as:

.. math::

    w_{ij} = \frac{1}{\ell_{ij}^2}

**Why this matters:**
This weighting is not arbitrary. By defining the weights as :math:`\ell^{-2}`, the eigenvalues of the graph Laplacian (:math:`\lambda`) correspond directly to the **squared wavenumber** (:math:`k^2`) of traveling waves on the grid.

* **Low Eigenvalues (:math:`\lambda \approx 0`):** Correspond to long-wavelength, inter-area oscillations that span the entire continent.
* **High Eigenvalues:** Correspond to short-wavelength, local disturbances.

When we apply the SGWT to this graph, each scale :math:`a\in\mathcal{A}` becomes physically meaningful, corresponding to a squared *pseudo-wavelength*, :math:`r\in \mathbb{R}`, where :math:`a=r^2` defines this mapping. This allows for filtering based on physical spread rather than just temporal frequency.