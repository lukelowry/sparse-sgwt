# Sparse GSP & SGWT Tools

A Collection of Graph signal processing Functions for Large Sparse Networks

## Introduction

A highly customizable, sparse-friendly SGWT/GSP module. Existing GSP tools for the SGWT over sparse networks is limited. This package provides tools to design, approximate, and implement a custom SGWT kernel for use over sparse networks.



## Installation Notes

The package can be installed using:

```
pip install sgwt
```

The CHOLMOD library can be used by installing `scikit-sparse` or using the a compiled CHOLMOD `.dll` file.

## Quick-Start

```python
import sgwt

L = sgwt.data.DELAY_TEXAS.laplacian()

scales = np.logspace(1e-2, 1e1, nscales)

with sgwt.FiltersDLL(L, scales) as gsp:

    LP = gsp.scaling_coeffs(b)
```

## Motivation

Given a rational approximation of some kernel function, we are able to implement graph convolutions using the Cholesky Decomposition.

Kernel Fitted Functions
------

The kernel fitting representation (1) is more generally a vector fitted function
```math
g_a(\mathbf{\Lambda})\approx 
        d_aI + e_a\mathbf{\Lambda}
        + \sum_{q\in Q}\dfrac{r_{q,a}}{\mathbf{\Lambda}+qI} 
```

An iterative pole realocation procedure is used to converge to a reduced order model. The convolution of some function $\mathbf{f}*g_a$ is computed using the cholesky decomposition and memory efficient re-factors.
## Analytical Filters

Low-Pass Spectral Graph Filter
------

The low-pass filter (2) is *refinable*, as it is a self-similar rational function. The refinability of (2) makes it useful for signal smoothing across a range of spatial scales.

```math
\phi(\mathbf{\Lambda}) = \dfrac{I}{\mathbf{\Lambda}+I} 
```


High-Pass Spectral Graph Filter
------

The proposed high-pass filter \eqref{eq:highpass} acts as a container for variations over the graph below a given spatial scale.

```math
\mu(\mathbf{\Lambda}) = \dfrac{\mathbf{\Lambda}}{\mathbf{\Lambda}+I}
```


Band-Pass Spectral Graph Filter
------

A convenient closed-form wavelet generating kernel was found to be a useful kernel as an alternative to the vector-fitting procedure if a particular filter does not need to be designed. 

```math
\Psi(\mathbf{\Lambda}) = \dfrac{4\mathbf{\Lambda}}{(\mathbf{\Lambda}+I)^2} 
```

This filter qualifies as a wavelet generating kernel for the SGWT, since $\Psi(0)=0$ and the admissibility condition is satisfied. The admissibility constant of this band-pass filter is $C_f=8/3$.

```math
\Psi(0)=0  \qquad\text{and}\quad \int_0^{\infty}\dfrac{\Psi^2(x)}{x}\mathrm{d}x <\infty
```


