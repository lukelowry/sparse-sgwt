# -*- coding: utf-8 -*-
"""Chebyshev Polynomial Approximation for SGWT.

Provides :class:`ChebyModel`, a fitting-oriented wrapper around
:class:`~sgwt.util.ChebyKernel` that mirrors the :class:`VFModel`
interface for a consistent fitting API.

Author: Luke Lowery (lukel@tamu.edu)
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from typing import Callable, Optional


@dataclass
class ChebyModel:
    """Chebyshev polynomial approximation model.

    Represents a spectral filter approximated by Chebyshev polynomials
    of the first kind on the interval
    ``[min_lambda, spectrum_bound]``.

    Parameters
    ----------
    C : ndarray, shape (order + 1, n_dims)
        Chebyshev coefficient matrix.
    spectrum_bound : float
        Upper bound of the approximation domain.
    min_lambda : float
        Lower bound of the approximation domain.

    See Also
    --------
    sgwt.util.ChebyKernel : Lightweight kernel used by :class:`~sgwt.ChebyConvolve`.
    VFModel : Rational (Vector Fitting) approximation model.

    Examples
    --------
    Fit a lowpass function and evaluate:

    >>> import numpy as np
    >>> model = ChebyModel.fit(lambda x: 1 / (x + 1), order=20, spectrum_bound=100.0)
    >>> model(np.array([0.0, 1.0, 50.0]))
    array(...)

    Fit and return a kernel ready for convolution:

    >>> K = ChebyModel.kernel(lambda x: 1 / (x + 1), order=20, spectrum_bound=100.0)
    """

    C: ndarray
    """Chebyshev coefficient matrix, shape ``(order + 1, n_dims)``."""

    spectrum_bound: float
    """Upper bound of the approximation domain."""

    min_lambda: float = 0.0
    """Lower bound of the approximation domain."""

    # ------------------------------------------------------------------
    # Fitting classmethods
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        order: int,
        spectrum_bound: float,
        n_samples: Optional[int] = None,
        sampling: str = 'chebyshev',
        min_lambda: float = 0.0,
        rtol: float = 1e-12,
        adaptive: bool = False,
        max_order: int = 500,
        target_error: float = 1e-10,
    ) -> 'ChebyModel':
        """Fit a Chebyshev polynomial to a vectorized function.

        Delegates to :meth:`ChebyKernel.from_function
        <sgwt.util.ChebyKernel.from_function>` and wraps the result.

        Parameters
        ----------
        f : callable
            Vectorized function ``f(x) -> y`` to approximate.
        order : int
            Polynomial order (must be >= 1).
        spectrum_bound : float
            Upper bound of the approximation domain.
        n_samples : int, optional
            Number of sample points (non-Chebyshev sampling only).
        sampling : str, default 'chebyshev'
            Sampling strategy: ``'chebyshev'``, ``'linear'``,
            ``'quadratic'``, or ``'logarithmic'``.
        min_lambda : float, default 0.0
            Lower bound of the approximation domain.
        rtol : float, default 1e-12
            Relative tolerance for truncating negligible coefficients.
        adaptive : bool, default False
            Automatically determine optimal order to achieve
            *target_error*.
        max_order : int, default 500
            Maximum order for adaptive mode.
        target_error : float, default 1e-10
            Target approximation error for adaptive mode.

        Returns
        -------
        ChebyModel
            Fitted polynomial model.
        """
        from ..util import ChebyKernel

        ck = ChebyKernel.from_function(
            f, order, spectrum_bound,
            n_samples=n_samples, sampling=sampling,
            min_lambda=min_lambda, rtol=rtol,
            adaptive=adaptive, max_order=max_order,
            target_error=target_error,
        )
        return cls(C=ck.C, spectrum_bound=ck.spectrum_bound,
                   min_lambda=ck.min_lambda)

    @classmethod
    def kernel(cls, f: Callable[[np.ndarray], np.ndarray],
               order: int, spectrum_bound: float, **kw):
        """Fit a Chebyshev polynomial and return a kernel for convolution.

        Performs the fit and returns a :class:`~sgwt.util.ChebyKernel`
        ready for use with :class:`~sgwt.ChebyConvolve`.

        Parameters
        ----------
        f : callable
            Vectorized function ``f(x) -> y`` to approximate.
        order : int
            Polynomial order (must be >= 1).
        spectrum_bound : float
            Upper bound of the approximation domain.
        **kw
            Forwarded to :meth:`fit`.

        Returns
        -------
        ChebyKernel
            Kernel ready for graph convolution.
        """
        from ..util import ChebyKernel

        ck = ChebyKernel.from_function(f, order, spectrum_bound, **kw)
        return ck

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x):
        """Evaluate the Chebyshev approximation at given points.

        Parameters
        ----------
        x : array_like
            Points in ``[min_lambda, spectrum_bound]`` at which to
            evaluate.

        Returns
        -------
        ndarray
            Approximated values.
        """
        x = np.asarray(x)
        x_scaled = (2.0 * (x - self.min_lambda)
                    / (self.spectrum_bound - self.min_lambda) - 1.0)
        y = np.polynomial.chebyshev.chebval(x_scaled, self.C)
        return y.T if y.ndim > 1 else y
