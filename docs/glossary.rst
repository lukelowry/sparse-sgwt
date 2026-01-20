Glossary
========

.. glossary::
   :sorted:

   Adjacency Matrix
      Matrix :math:`\mathbf{W} \in \mathbb{R}^{N \times N}` where :math:`W_{ij}`
      is the edge weight between vertices :math:`i` and :math:`j`. Zero entries
      indicate no edge.

   Degree Matrix
      Diagonal matrix :math:`\mathbf{D} \in \mathbb{R}^{N \times N}` where
      :math:`D_{ii} = \sum_j W_{ij}` is the sum of weights connected to
      vertex :math:`i`.

   Graph Laplacian
      The matrix :math:`\mathbf{L} = \mathbf{D} - \mathbf{W}` encoding graph
      structure. Its eigenvalues provide a notion of graph frequency, with
      physical interpretation depending on the branch weighting scheme.

   Poles
      Shift values :math:`q` in :math:`(\mathbf{L} + q\mathbf{I})^{-1}`. For
      analytical filters, pole = 1/scale. In :class:`VFKernel`, stored as
      :attr:`Q` with shape ``(n_poles,)``.

   Rational Approximation
      Representing a filter function as a ratio of polynomials. Vector
      Fitting produces pole-residue form :math:`g(\lambda) \approx d +
      \sum_k \frac{r_k}{\lambda + q_k}`, enabling efficient linear solves.

   Residues
      Complex coefficients :math:`r_k` in a rational approximation. In
      :class:`VFKernel`, stored as :attr:`R` with shape ``(n_poles, n_dims)``
      where ``n_dims`` is the output dimension of the kernel.

   Scale
      Dilation parameter :math:`s` controlling filter bandwidth. Related to
      poles by :math:`s = 1/q` and to wavelength by :math:`\lambda \approx
      \sqrt{s}`. Larger scales correspond to lower graph frequencies.

   Spectral Domain
      The eigenspace of the Graph Laplacian. Signals are decomposed into
      eigenvector components, analogous to Fourier analysis on regular domains.

   Vector Fitting
      Algorithm for fitting a rational function to frequency-domain samples.
      Produces poles and residues stored in :class:`VFKernel` for use with
      :class:`Convolve` and :class:`DyConvolve`.

   Vertex Domain
      Signals defined on graph nodes as vectors :math:`\mathbf{f} \in
      \mathbb{R}^N`, where the :math:`i`-th element is the signal value at
      vertex :math:`i`. Contrast with :term:`Spectral Domain`.

   VFKernel
      Data structure holding Vector Fitting results:

      - :attr:`Q`: poles, shape ``(n_poles,)``
      - :attr:`R`: residues, shape ``(n_poles, n_dims)``
      - :attr:`D`: direct term, shape ``(n_dims,)``

   Wavelength
      Spatial extent of oscillation modes, approximately :math:`\sqrt{s}`
      where :math:`s` is the scale. Large wavelengths indicate spatially
      extended (inter-area) modes; small wavelengths indicate localized modes.

   Wavenumber
      Spatial frequency :math:`k`. For appropriately weighted graphs (e.g.,
      inverse squared distance), Laplacian eigenvalues correspond to
      :math:`k^2`, linking spectral and physical domains.
