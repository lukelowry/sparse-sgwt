

## Kernel Fitting

The kernel fitting representation is more generally a vector fitted function, a simple pole expansion of the form:
```math
g_a(\mathbf{\Lambda})\approx 
        d_aI + e_a\mathbf{\Lambda}
        + \sum_{q\in Q}\dfrac{r_{q,a}}{\mathbf{\Lambda}+qI} 
```

An iterative pole realocation procedure is used to converge to a reduced order model. The convolution of some function $\mathbf{f}*g_a$ is computed using the cholesky decomposition and memory efficient re-factors.

An example of an approriate format of the rational expansion:

```json
{
    "nfunc": N,
    "d": [d0, d1, ..., dN],
    "npoles": M,
    "poles": [
        {
            'q': q0, 
            'r':[r0, r1, ..., rN]
        },
        {
            'q': q1, 
            'r':[r0, r1, ..., rN]
        },
        ...
        {
            'q': qM, 
            'r':[r0, r1, ..., rN]
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

For the context that this was designed for, a direct solve approach is preferred to an iterative solver like `ARMA`. Time-varying graph signals must be as efficient as possible with memory, to ensure scalability to signals of large sparse networks. Especially if the process is online.

The `cholmod_solve2` and `updown` functions are the primary engine, in addition to other various design choices that accelerate graph convolutions.
