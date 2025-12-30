## Installation

The package can be installed using:

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
L = sgwt.library.IMPEDANCE_TX

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
