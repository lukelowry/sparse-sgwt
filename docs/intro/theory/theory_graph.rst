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

When working with a physical distance metric, the library utilizes **Inverse Squared Length Weighting**. For each branch of length :math:`\ell_{ij}`, the corresponding branch weight is assigned as:

.. math::

    w_{ij} = \frac{1}{\ell_{ij}^2}

**Why this matters:**
This weighting is not arbitrary. By defining the weights as :math:`\ell^{-2}`, the eigenvalues of the graph Laplacian (:math:`\lambda`) correspond directly to the **squared wavenumber** (:math:`k^2`) of traveling waves on the grid.

* **Low Eigenvalues (:math:`\lambda \approx 0`):** Correspond to long-wavelength, inter-area oscillations that span the entire continent.
* **High Eigenvalues:** Correspond to short-wavelength, local disturbances.

When we apply the SGWT to this graph, each scale :math:`a\in\mathcal{A}` becomes physically meaningful, corresponding to a squared *pseudo-wavelength*, :math:`r\in \mathbb{R}`, where :math:`a=r^2` defines this mapping. This allows for filtering based on physical spread rather than just temporal frequency.

Derivation of the Delay Weight
------------------------------

To derive the precise branch weights for the **Delay Graph Laplacian**, we must calculate the effective parameters of transformers and transmission lines, accounting for the fact that real-world grid data often aggregates these values.

**Effective Branch Parameters**

1. **Inductance:**
   We derive the branch inductance :math:`L_{ij}` from the imaginary part of the series impedance :math:`Z^{br}_{ij}`:

   .. math::

      \omega L_{ij}= \text{Im}\left(Z^{br}_{ij}\right)

2. **Capacitance:**
   Shunt admittance is often stored as a nodal aggregate. We first calculate the net nodal capacitance :math:`C_n` from the shunt admittance :math:`Y^{sh}_{n}`:

   .. math::

       \omega C_{n}= \text{Im}\left(Y^{sh}_{n}\right)

   To apply this to the edges (consistent with :math:`\pi`-model assumptions), we define the **effective branch capacitance** :math:`C_{ij}` as the average of the nodal capacitances at the branch terminals:

   .. math::

       C_{ij}=\dfrac{1}{2}(C_{i}+C_{j})

**Lossless Propagation Delay**

Using these parameters, we estimate the lossless propagation delay :math:`\tau_{ij}` for every branch. This represents the "electrical distance" a wave must travel:

.. math::

    \omega_{base}\tau_{ij} = \text{Re}\left( \sqrt{Y_{ij}Z_{ij}} \right)

We define :math:`\mathbf{T}` as the diagonal matrix of these propagation delays. This forms the basis of the **Delay Graph Laplacian**:

.. math::

    \mathbf{L}_{\tau} = \mathbf{A}^\top \mathbf{T}^{-2} \mathbf{A}

In this formulation, the Laplacian operator :math:`\mathbf{L}` is mathematically equivalent to the continuous wave operator :math:`c^2\nabla^2`.

Alternative Weighting Schemes
-----------------------------

Depending on the physical phenomenon being analyzed, different weighting schemes may be appropriate.

* **Reciprocal Squared Delay (Preferred):**
    
    .. math::
       w_{ij} = \frac{1}{\tau_{ij}^2}
       
    Used for **wave propagation** analysis. Aligns the graph spectrum with the squared wavenumber (:math:`k^2`).

* **Reciprocal Squared Distance:**
    
    .. math::
       w_{ij} = \frac{1}{d_{ij}^2}
       
    Used for **geographic analysis** when electrical parameters are unavailable but GPS coordinates are known.

* **Admittance Magnitude:**
    
    .. math::
       w_{ij} = \left| \frac{1}{Z_{ij}} \right|
       
    Used for **electrical coupling** analysis. This is standard for spectral clustering, as it groups tightly coupled buses regardless of the line length.