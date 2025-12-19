# Sparse GSP & SGWT Tools

A highly customizable, sparse-friendly SGWT/GSP module. This package provides tools to design, approximate, and implement a custom SGWT kernel for use over sparse networks.

Intended for GSP of time-vertex signals over static and dynamic sparse graphs.

## Installation

The package can be installed using:

```
pip install sgwt
```

The CHOLMOD library can be used by installing `scikit-sparse` or using the a compiled CHOLMOD `.dll` file.

## Quick Start

The module has a small repository of built in graph laplacians that are useful for quick start examples. The user can load any graph Laplacian in `csc_matrix` format.


Then, we create or import a time-vertex function $X\in\mathbb{R}^{|N|\times|T|}$ stored as a 2D numpy array in column-major ordering (i.e., fortran style). The convolution of `X` with various graph filters can be computed efficiently as follows.
```python
import sgwt

# Graph Laplacian
L = sgwt.data.DELAY_TX

# Signal (e.g., Dirac Delta)
X = sgwt.impulse(L, node=...)

# Discrete Scales
s = np.logspace(...)

# Assign L as the convolution context
with sgwt.Filters(L) as filt:

    # Apply Low-Pass Filters
    Y = filt.lowpass(X, s)
```

The numpy arrays `Y[i]` correspond to a filtered signal `X` at the `i-th` scale.

The purpose of the context manager is to provide safe re-use of `cholmod` workspace. While inside the context, the convolution procedure optimizes memory usage.


## Kernel Fitting


The kernel fitting representation is more generally a vector fitted function, a simple pole expansion of the form:
```math
g_a(\mathbf{\Lambda})\approx 
        d_aI + e_a\mathbf{\Lambda}
        + \sum_{q\in Q}\dfrac{r_{q,a}}{\mathbf{\Lambda}+qI} 
```

An iterative pole realocation procedure is used to converge to a reduced order model. The convolution of some function $\mathbf{f}*g_a$ is computed using the cholesky decomposition and memory efficient re-factors.

### Example Use

For more advanced functionality, the convolution is generalized using kernel fitting. Same as before, we 
```python
import sgwt

# Underlying Graph
L = sgwt.data.LENGTH_TEXAS

# Kernel Function
F = sgwt.data.MEXICAN_HAT

# Signal (nVert x nTime)
X = np.random.random(
    shape=(L.shape[0], 100),
    order = 'F'
)

```

Then the convolution is simply performed on our signal `X` as follows:

```python
with sgwt.Convolve(L) as conv:

    Y = conv(X, F)

```

The convolutional kernel `f` can be a vector function, meaning multiple filters can be applied concurrently (i.e., you have have an SGWT kernel that compactly calculates all wavelet coefficients)

### Rational Kernel JSON Format

```
{
    "nfunc": n,
    "d": [d_0, d_1, ..., d_n],
    "npoles": m
    "poles": [
        {
            'q': q_0, 
            'r':[r_0, r_1, ..., r_n]
        },
        {
            'q': q_1, 
            'r':[r_0, r_1, ..., r_n]
        },
        ...
        {
            'q': q_m, 
            'r':[r_0, r_1, ..., r_n]
        }
    ]
}
```

## Analytical Filters

### Low-Pass Spectral Graph Filter

The low-pass filter (2) is *refinable*, as it is a self-similar rational function. The refinability of (2) makes it useful for signal smoothing across a range of spatial scales.

```math
\phi(\mathbf{\Lambda}) = \dfrac{I}{\mathbf{\Lambda}+I} 
```


### High-Pass Spectral Graph Filter

The proposed high-pass filter \eqref{eq:highpass} acts as a container for variations over the graph below a given spatial scale.

```math
\mu(\mathbf{\Lambda}) = \dfrac{\mathbf{\Lambda}}{\mathbf{\Lambda}+I}
```


### Band-Pass Spectral Graph Filter


A convenient closed-form wavelet generating kernel was found to be a useful kernel as an alternative to the vector-fitting procedure if a particular filter does not need to be designed. 

```math
\Psi(\mathbf{\Lambda}) = \dfrac{4\mathbf{\Lambda}}{(\mathbf{\Lambda}+I)^2} 
```

This filter qualifies as a wavelet generating kernel for the SGWT, since $\Psi(0)=0$ and the admissibility condition is satisfied. The admissibility constant of this band-pass filter is $C_f=8/3$.

```math
\Psi(0)=0  \qquad\text{and}\quad \int_0^{\infty}\dfrac{\Psi^2(x)}{x}\mathrm{d}x <\infty
```


## Cholesky Implementation

Given a rational approximation of some kernel function, we are able to implement graph convolutions using the Cholesky Decomposition. To ensure scalability to signals of large sparse networks, time-varying graph signals must be as efficient as possible with memory.

The `cholmod_solve2` function is the primary engine behind the fast reusable convolution environment. Access to the `cholmod` functions also means that this module is ideal for GSP of signals on dynamic graphs, using low-rank updates to change the factorization of the graph Laplacian.

