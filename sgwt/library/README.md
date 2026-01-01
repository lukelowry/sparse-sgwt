# SGWT Data Library

This directory contains built-in graph Laplacians, signals, and spectral kernels used for testing and demonstration of the Sparse SGWT library.

## Mathematical Definition

Let an undirected graph $\mathcal{G}=\{\mathcal{V}, \mathcal{E}, \mathbf{A}, \mathbf{w}\}$ be defined by a set of verticies $|\mathcal{V}|=N$ and a set of edges $\mathcal{E}$ which are related by the arc-node incident matrix $\mathbf{A}\in\mathbb{R}^{|\mathcal{E}|\times |\mathcal{V}|}$ and the vector of branch weights $\mathbf{w}\in\mathbb{R}^{|\mathcal{E}|}$. A *vertex domain* function on the graph $f:\mathcal{V}\to\mathbb {R}$ can be written as a vector $\mathbf{f}\in \mathbb{R}^N$, whose $i^{th}$ element corresonds to the evaluation of $f$ at the $i^{th}$ vertex. The *graph Laplacian* is denoted by $\mathbf{L}\in\mathbb{R}^{N\times N}$, a discrete analogue of the continuous Laplace-Beltrami operator.

$$
\mathbf{L} := \mathbf{A}^\top \text{diag}(\mathbf{w}) \mathbf{A}
$$

When working with a physical distance as a distance metric: for each branch of length $\ell$, the corresponding branch weight is assigned $\ell^{-2}$ so that the eigenvalues of the Laplacian represent the squared spatial frequency $\lambda = k^2$, where $k$ is the wavenumber. When we apply the SGWT to this graph, each scale $a\in\mathcal{A}$ is physically meaningful, corresponding to a squared *pseudo-wavelength*, $r\in \mathbb{R}$, where $a=r^2$ defines this mapping.

## Contents

### Laplacians

The library includes Laplacians for various synthetic power grid networks (Texas, Western Interconnection, Eastern Interconnection, etc.).

*   **DELAY**: Edge weights are based on phase distance ($\theta^{-2}$).
*   **LENGTH**: Edge weights are based on physical transmission line length ($\ell^{-2}$).
*   **IMPEDANCE**: Edge weights are based on electrical impedance ($|Z|$).

### Signals

*   **SIGNALS**: Contains vertex-domain signals. `COORDS` are $N \times 2$ arrays containing the longitude and latitude of each node/bus.

### Kernels

*   **KERNELS**: JSON files defining rational approximations (Vector Fitting) for spectral graph wavelets.
    *   Mexican Hat
    *   Modified Morlet
    *   Shannon
    *   Gaussian