# Sparse Graph Signal Processing (GSP)

[![PyPI version](https://badge.fury.io/py/sgwt.svg)](https://badge.fury.io/py/sgwt)
[![Python Version](https://img.shields.io/pypi/pyversions/sgwt.svg)](https://pypi.org/project/sgwt/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](./LICENSE.md)

A high-performance Python library for sparse Graph Signal Processing (GSP) and Spectral Graph Wavelet Transforms (SGWT). This package leverages the `CHOLMOD` library for efficient sparse direct solvers, providing significant speedups over traditional dense or iterative methods for large-scale graph convolution.

## Key Features

- **High-Performance Sparse Solvers**: Direct integration with the `CHOLMOD` library for optimized sparse Cholesky factorizations and linear system solves.
- **Generalized Graph Convolution**: Support for arbitrary spectral kernels via rational approximation (*Kernel Fitting*) and standard analytical filters (low-pass, band-pass, high-pass).
- **Dynamic Topology Support**: Specialized routines for graphs with evolving structures, utilizing efficient rank-1 updates for real-time topology changes.
- **Resource-Aware Execution**: Context-managed memory allocation and workspace reuse to minimize overhead in high-throughput applications.
- **Integrated Graph Repository**: Built-in access to standardized graph Laplacians and signals from power systems and infrastructure networks.

## Installation

The package can be installed via pip:

```bash
python -m pip install sgwt
```

The package uses a compiled version of `CHOLMOD`.

## Basic Usage

### Quick Start
For the quick-start example, we will find the response of a low-pass filter $\phi$ scaled by `s` to impulse $\delta$ at node $n$ over the graph `L`. This is mathematically denoted by $\phi_{n,s}=\delta_n*\phi_s$.
```python
import sgwt

# CSC Graph Laplacian
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

The purpose of the context manager is to provide safe re-use of `CHOLMOD` workspace. While inside the context, the convolution procedure optimizes memory usage.

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

Although, a `(N,1)` array can also be used.

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

Same as before, the convolution is simply performed on our signal `X` by first defining `L` as the convolution context.

### Dynamic Graphs

In many real-world applications, such as power systems or sensor networks, the underlying graph topology is dynamic. Re-initializing the entire convolution context for every edge update is computationally prohibitive. This example demonstrates the use of `DyConvolve` to perform efficient, real-time signal filtering on an evolving graph by leveraging rank-1 updates to adapt existing factorizations on-the-fly.

```python
from sgwt.dynamic import DyConvolve
from sgwt import DELAY_USA as L

poles = [10.0, 1.0, 0.1]

with DyConvolve(L, poles) as conv:
    for f_t, event in stream:
        if event:
            conv.addbranch(*event) # Update topology
        W = conv.bandpass(f_t)     # Filter signal
```

At each iteration, the matrix `W` contains the column vectors which are the filtered versionf of `f` at the 'spatial' scale associated with each pole. So in this example `W` be a 3-column matrix representing the signal at three different scales.

## Other

A brief review of the theoretical work used in this module is available in this repository in markdown format [THEORY.md](THEORY.md). Additional examples with descriptions are also available in the [/examples](/examples) directory.


This module is also implemented in [Julia](https://github.com/lukelowry/SpectralGraphWavelet.jl) which takes advantage of the native SuiteSparse support.

### References

The `CHOLMOD` library of [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) was developed by Dr. Tim Davis at Texas A&M University.

The graph laplacians used in the examples are derived from the [synthetic grid repository](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/), available thanks to the research of Dr. Adam Birchfield at Texas A&M University. 
- Birchfield, Adam B. et al. “Grid Structural Characteristics as Validation Criteria for Synthetic Networks”. In: IEEE Trans. on Power Sys. 32.4 (2017)

The theoretical work of this module is derived in part from this [paper](https://scholarspace.manoa.hawaii.edu/items/3f08d29d-db06-41d5-b235-2ee549bd198b) nominated for best paper at HICSS-59.
- Lowery, Luke, Jongoh Baek, and Adam Birchfield. "Using Spectral Graph Wavelets to Analyze Large Power System Oscillation Modes." (2026)



### Author

Luke Lowery received the B.S. degree in electrical engineering at Texas A&M University in 2023, where he is currently pursuing the Ph.D. in electrical engineering. His research interests include numerical macromodeling and graph-based signal processing for large power systems. More information can be found at [his website](https://lukelowry.github.io/).