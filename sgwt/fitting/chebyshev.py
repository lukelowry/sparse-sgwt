# -*- coding: utf-8 -*-
"""Chebyshev Polynomial Approximation for SGWT.

Provides :class:`ChebyModel`, the fitting model for constructing Chebyshev
polynomial approximations of spectral filters. Mirrors the :class:`VFModel`
interface for a consistent fitting API.

Author: Luke Lowery (lukel@tamu.edu)
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix
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

    >>> from scipy.sparse import eye
    >>> L = eye(10, format="csc")
    >>> K = ChebyModel.kernel(L, lambda x: 1 / (x + 1), order=20)
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
        if order < 1:
            raise ValueError("Order must be >= 1")

        if adaptive:
            return cls._adaptive_fit(f, order, spectrum_bound, min_lambda,
                                     sampling, rtol, max_order, target_error)

        coeffs = cls._compute_coefficients(f, order, spectrum_bound, min_lambda,
                                           n_samples, sampling)
        coeffs = cls._ensure_2d(coeffs)
        coeffs = cls._truncate(coeffs, rtol)

        return cls(C=coeffs, spectrum_bound=spectrum_bound, min_lambda=min_lambda)

    @classmethod
    def kernel(
        cls,
        L: csc_matrix,
        f: Callable[[np.ndarray], np.ndarray],
        order: int,
        **kw,
    ):
        """Fit a graph Chebyshev kernel and return it for convolution.

        Estimates the spectral bound from *L*, performs the fit, and
        returns a :class:`~sgwt.util.ChebyKernel` ready for use with
        :class:`~sgwt.ChebyConvolve`.

        Parameters
        ----------
        L : csc_matrix
            Graph Laplacian.
        f : callable
            Vectorized function ``f(x) -> y`` to approximate.
        order : int
            Polynomial order (must be >= 1).
        **kw
            Forwarded to :meth:`fit`.

        Returns
        -------
        ChebyKernel
            Kernel ready for graph convolution.
        """
        from ..util import ChebyKernel, estimate_spectral_bound

        spectrum_bound = estimate_spectral_bound(L)
        model = cls.fit(f, order, spectrum_bound, **kw)
        return ChebyKernel(C=model.C, spectrum_bound=model.spectrum_bound,
                           min_lambda=model.min_lambda)

    @classmethod
    def kernel_on_graph(
        cls,
        L: csc_matrix,
        f: Callable[[np.ndarray], np.ndarray],
        order: int,
        **kw,
    ):
        """Backward-compatible alias for :meth:`kernel`.

        Parameters
        ----------
        L : csc_matrix
            Graph Laplacian.
        f : callable
            Vectorized function ``f(x) -> y`` to approximate.
        order : int
            Polynomial order (must be >= 1).
        **kw
            Forwarded to :meth:`kernel`.

        Returns
        -------
        ChebyKernel
            Kernel ready for graph convolution.
        """
        return cls.kernel(L, f, order, **kw)

    # ------------------------------------------------------------------
    # Private fitting helpers
    # ------------------------------------------------------------------

    @classmethod
    def _compute_coefficients(cls, f, order, spectrum_bound, min_lambda, n_samples, sampling):
        """Compute Chebyshev coefficients using optimal or fallback sampling.

        Uses Chebyshev-Gauss-Lobatto nodes for the 'chebyshev' strategy
        (optimal polynomial interpolation), or weighted least-squares
        fitting for other strategies.

        Parameters
        ----------
        f : callable
            Vectorized function to approximate.
        order : int
            Polynomial order.
        spectrum_bound : float
            Upper bound of the approximation domain.
        min_lambda : float
            Lower bound of the approximation domain.
        n_samples : int or None
            Number of sample points (non-Chebyshev sampling only).
        sampling : str
            Sampling strategy: 'chebyshev', 'linear', 'quadratic', 'logarithmic'.

        Returns
        -------
        np.ndarray
            Coefficient array of shape ``(order + 1,)`` or ``(order + 1, n_dims)``.
        """
        lambda_range = spectrum_bound - min_lambda
        lambda_mid = (spectrum_bound + min_lambda) / 2.0

        if sampling == 'chebyshev':
            # Chebyshev-Gauss-Lobatto nodes: optimal for polynomial interpolation
            n = order + 1
            k = np.arange(n)
            x_cheb = np.cos(np.pi * k / order)  # Nodes in [-1, 1]
            sample_x = lambda_mid + (lambda_range / 2.0) * x_cheb
            f_values = cls._ensure_2d(f(sample_x))

            # Compute coefficients via discrete orthogonality (DCT-like)
            coeffs = np.zeros((n, f_values.shape[1]))
            w = np.ones(n); w[0] = w[-1] = 0.5  # Endpoint weights

            for j in range(n):
                T_j = np.cos(j * np.pi * k / order)
                scale = 2.0 / order if 0 < j < order else 1.0 / order
                coeffs[j] = scale * np.sum(w[:, None] * f_values * T_j[:, None], axis=0)

            return coeffs

        # Fallback: least-squares fitting for other sampling strategies
        n_samples = n_samples or max(4 * (order + 1), 1000)
        t = np.linspace(0, 1, n_samples)

        if sampling == 'quadratic':
            sample_x = min_lambda + lambda_range * (t ** 2)
        elif sampling == 'logarithmic':
            eps = max(min_lambda * 0.001, 1e-10)
            sample_x = np.exp(np.log(min_lambda + eps) + t * np.log(spectrum_bound / (min_lambda + eps)))
        else:  # linear
            sample_x = min_lambda + lambda_range * t

        x_scaled = 2.0 * (sample_x - min_lambda) / lambda_range - 1.0

        # Chebyshev-weighted least squares
        weights = 1.0 / np.sqrt(1.0 - np.clip(x_scaled ** 2, 0, 0.9999))
        return np.polynomial.chebyshev.chebfit(x_scaled, f(sample_x), order, w=weights)

    @classmethod
    def _adaptive_fit(cls, f, start_order, spectrum_bound, min_lambda,
                      sampling, rtol, max_order, target_error):
        """Adaptively determine optimal polynomial order.

        Iteratively increases the polynomial order until the relative
        approximation error on a test grid falls below ``target_error``
        or ``max_order`` is reached.

        Parameters
        ----------
        f : callable
            Vectorized function to approximate.
        start_order : int
            Initial polynomial order.
        spectrum_bound : float
            Upper bound of the approximation domain.
        min_lambda : float
            Lower bound of the approximation domain.
        sampling : str
            Sampling strategy.
        rtol : float
            Relative tolerance for coefficient truncation.
        max_order : int
            Maximum polynomial order.
        target_error : float
            Target relative approximation error.

        Returns
        -------
        ChebyModel
            Fitted model with adaptively chosen order.
        """
        test_x = np.linspace(min_lambda, spectrum_bound, 1000)
        f_exact = cls._ensure_2d(f(test_x))
        order = max(start_order, 8)

        while True:
            coeffs = cls._ensure_2d(
                cls._compute_coefficients(f, order, spectrum_bound, min_lambda, None, sampling)
            )

            # Evaluate and compute relative error
            x_scaled = 2.0 * (test_x - min_lambda) / (spectrum_bound - min_lambda) - 1.0
            f_approx = np.polynomial.chebyshev.chebval(x_scaled, coeffs)
            f_approx = f_approx.T if f_approx.ndim > 1 else f_approx[:, None]

            rel_error = np.max(np.abs(f_exact - f_approx) / np.maximum(np.abs(f_exact), 1e-15))

            if rel_error <= target_error or order >= max_order:
                break
            order = min(int(order * 1.5) + 1, max_order)

        return cls(C=cls._truncate(coeffs, rtol), spectrum_bound=spectrum_bound, min_lambda=min_lambda)

    @classmethod
    def _ensure_2d(cls, arr):
        """Ensure array is 2D with shape (n, dims)."""
        arr = np.atleast_1d(arr)
        return arr[:, None] if arr.ndim == 1 else arr

    @classmethod
    def _truncate(cls, coeffs, rtol):
        """Truncate negligible higher-order coefficients."""
        threshold = rtol * np.max(np.abs(coeffs))
        row_max = np.max(np.abs(coeffs), axis=1)
        significant = np.where(row_max > threshold)[0]
        last_idx = significant[-1] + 1 if significant.size > 0 else 1
        return coeffs[:last_idx]

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
