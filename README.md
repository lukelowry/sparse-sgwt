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


## Quick-Staart

```
import sgwt

L = sgwt.data.DELAY_TEXAS.laplacian()

scales = np.logspace(1e-2, 1e1, nscales)

with sgwt.FiltersDLL(L, scales) as gsp:

    LP = gsp.scaling_coeffs(b)
```

## Motivation

Given a rational approximation of some kernel function, we are able to implement graph convolutions using the Cholesky Decomposition.