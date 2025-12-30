## Installation

The package can be installed using:

```
tbd
```

An advantage of the julia implementation is that it has native access to `CHOLMOD`. 

## Basic Usage

### Quick Start
For the quick-start example, we will find the response of a low-pass filter $\phi$ scaled by `s` to impulse $\delta$ at node $n$ over the graph `L`. This is mathematically denoted by $\phi_{n,s}=\delta_n*\phi_s$.

```julia
using SpectralGraphWavelet


```

The numpy arrays `Y[i]` correspond to a filtered signal `X` at the `i-th` scale.
