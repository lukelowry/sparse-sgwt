# Sparse Spectral Graph Wavelet Transform (SGWT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python library for computing Spectral Graph Wavelet Transforms (SGWT) on large-scale sparse graphs. This package leverages the CHOLMOD library for efficient sparse direct solvers, providing significant speedups over traditional dense or iterative methods.

## Key Features

- **High Performance**: Direct integration with CHOLMOD for fast sparse matrix factorizations.
- **Versatile Kernels**: Support for analytical filters (low-pass, band-pass, high-pass) and custom kernels via Vector Fitting (VF).
- **Dynamic Topology**: Optimized routines for graphs with evolving structures (e.g., power system line closures).
- **Memory Efficient**: Context-managed workspace reuse to minimize allocation overhead.
- **Graph Library**: Built-in access to common graph Laplacians (Texas, USA, WECC, etc.).

## Installation

The package can be installed via pip:

```
python -m pip install sgwt
```

The package uses a compiled CHOLMOD `.dll` file. Tests use `scikit-sparse` as a second level of vertification.

## Basic Usage

### Quick Start
For the quick-start example, we will find the response of a low-pass filter $\phi$ scaled by `s` to impulse $\delta$ at node $n$ over the graph `L`. This is mathematically denoted by $\phi_{n,s}=\delta_n*\phi_s$.
```python
import sgwt

# Graph Laplacian
L = sgwt.IMPEDANCE_TX

# Impulse at Vertex n
X = sgwt.impulse(L, n=...)

# Discrete Scales
s = np.logspace(...)

# L -> Context of Convolution
with sgwt.Convolve(L) as conv:

    # Apply Low-Pass Filters
    Y = conv.lowpass(X, s)
```

The numpy arrays `Y[i]` correspond to a filtered signal `X` at the `i-th` scale.

The purpose of the context manager is to provide safe re-use of `cholmod` workspace. While inside the context, the convolution procedure optimizes memory usage.

### Underlying Graph

The module has a small repository of built in graph laplacians that are useful for quick start examples. 

```python
L = sgwt.LENGTH_TX
L = sgwt.IMPEDANCE_HAWAII
L = sgwt.DELAY_USA
```

The user can also load any graph Laplacian so long it is in the `csc_matrix` format.


### Input Signals

A real-valued time-vertex function $X\in\mathbb{R}^{|N|\times|T|}$ stored as a 2D numpy array in column-major ordering (i.e., fortran style) can be used. For example, an empty array meeting these specifications:

```python
X = np.empty(
    shape=(nVert, nTime),
    order = 'F'
)
```

Although, a `(nVert,1)` array can also be used.

### Kernel Functions

There are three convenience analytical filters available.
```python
with Convolve(L) as conv:

    Y = conv.lowpass(X, s)
    Y = conv.bandpass(X, s)
    Y = conv.highpass(X, s)
```

For more advanced functionality, the convolution is generalized using kernel fitting. Single Function kernels include `MEXICAN_HAT`, `MODIFIED_MORLET`, `SHANNON`, and more.

The convolutional kernel `F` can be a vector function, meaning multiple filters can be applied concurrently (i.e., an orthoginal kernel to generate the wavaelet coefficients `SGWT`) This kernel will be available soon.
```python
with Convolve(L) as conv:

    Y = conv(X, F)
```

Same as before, the convolution is simply performed on our signal `X` by first defining L as the convolution context.