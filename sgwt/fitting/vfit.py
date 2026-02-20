# -*- coding: utf-8 -*-
"""Vector Fitting (Rational Approximation) for SGWT.

Implements the Vector Fitting algorithm for constructing rational
approximations of frequency response data. Provides both general-purpose
fitting and a convenience constructor for graph spectral kernels.

Author: Luke Lowery (lukel@tamu.edu)
"""

from dataclasses import dataclass

from numpy import (
    abs, append, asarray, atleast_2d, concatenate,
    diag, geomspace, hstack, mean, ndarray,
    ones, outer, sort_complex, sqrt, vstack, zeros,
)
from numpy.linalg import eigvals, lstsq, qr


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cauchy(s, poles):
    """Build the Cauchy matrix.

    Parameters
    ----------
    s : ndarray, shape (K,)
        Sample points.
    poles : ndarray, shape (N,)
        Current pole locations.

    Returns
    -------
    ndarray, shape (K, N)
        Matrix where element ``[i, j] = 1 / (s[i] - poles[j])``.
    """
    return 1.0 / (s[:, None] - poles)


def _lstsq(A, b):
    """Solve an overdetermined least-squares system.

    Parameters
    ----------
    A : ndarray, shape (M, N)
        Coefficient matrix.
    b : ndarray, shape (M,) or (M, P)
        Right-hand side.

    Returns
    -------
    ndarray, shape (N,) or (N, P)
        Least-squares solution.
    """
    return lstsq(A, b, rcond=None)[0]


def _basis(s, poles, fit_d, fit_e):
    """Construct the regression matrix ``[Cauchy | 1 | s]``.

    Parameters
    ----------
    s : ndarray, shape (K,)
        Sample points.
    poles : ndarray, shape (N,)
        Current pole locations.
    fit_d : bool
        Append a column of ones for the direct term.
    fit_e : bool
        Append a column of ``s`` for the linear term.

    Returns
    -------
    ndarray, shape (K, N + fit_d + fit_e)
        Augmented regression matrix.
    """
    cols = [_cauchy(s, poles)]
    if fit_d:
        cols.append(ones((len(s), 1)))
    if fit_e:
        cols.append(s[:, None])
    return hstack(cols)


def _enforce_conjugates(poles):
    """Snap complex poles to exact conjugate pairs.

    Any pole whose imaginary part is below a tolerance is forced real.
    Remaining complex poles are paired with their conjugates.

    Parameters
    ----------
    poles : ndarray, shape (N,)
        Poles to regularize.

    Returns
    -------
    ndarray, shape (N,)
        Poles with exact conjugate symmetry.
    """
    tol = 1e-6 * (abs(poles).max() + 1)
    real = poles[abs(poles.imag) < tol].real + 0j
    upper = poles[poles.imag >= tol]
    return sort_complex(concatenate([real, upper, upper.conj()]))


def _initial_poles(s, N, real_only):
    """Generate logarithmically-spaced starting poles.

    For ``real_only=True``, produces real negative poles spanning the
    data range. Otherwise, produces complex conjugate pairs with small
    negative real parts.

    Parameters
    ----------
    s : ndarray, shape (K,)
        Sample points used to determine the frequency range.
    N : int
        Number of poles to generate.
    real_only : bool
        If True, generate only real negative poles.

    Returns
    -------
    ndarray, shape (N,)
        Initial pole locations (complex dtype).
    """
    if real_only or N < 2:
        r = abs(s[s != 0]) if (s != 0).any() else abs(s) + 1
        return -geomspace(max(r.min() * 0.1, 1e-6), r.max() * 2, N).astype(complex)
    w = abs(s.imag)
    w = w[w > 0]
    pts = geomspace(max(w.min() * 0.5, 1e-3), w.max() * 1.5, N // 2)
    cc = -pts * 0.01 + 1j * pts
    poles = concatenate([cc, cc.conj()])
    return append(poles, -pts[-1])[:N] if N % 2 else poles[:N]


def _relocate(s, f, poles, fit_d, fit_e, real_only):
    """Pole relocation step of the Vector Fitting algorithm.

    Builds an augmented least-squares system whose solution yields a
    weight function, then computes the zeros of that weight function as
    eigenvalues of a companion matrix. These zeros become the relocated
    poles, with stability enforced (negative real parts).

    Parameters
    ----------
    s : ndarray, shape (K,)
        Sample points.
    f : ndarray, shape (K, A)
        Response data (A channels).
    poles : ndarray, shape (N,)
        Current pole locations.
    fit_d : bool
        Include direct term in the basis.
    fit_e : bool
        Include linear term in the basis.
    real_only : bool
        Force relocated poles to be real.

    Returns
    -------
    ndarray, shape (N,)
        Relocated pole locations.
    """
    K, A = f.shape
    N = len(poles)

    Phi_sig = _basis(s, poles, True, False)
    Q, _ = qr(_basis(s, poles, fit_d, fit_e), mode='reduced')

    # Remove the signal subspace so we only fit the weight function
    weighted = f[:, :, None] * Phi_sig[:, None, :]
    flat = weighted.reshape(K, -1)
    rows = (flat - Q @ (Q.conj().T @ flat)).reshape(-1, N + 1)

    # Normalize so the weight function can't vanish
    w = sqrt(K * A)
    M = vstack([rows, Phi_sig.sum(0)[None, :] * w])
    rhs = concatenate([zeros(M.shape[0] - 1), [K * w]])
    z = _lstsq(M, rhs)
    c_tilde, d_tilde = z[:N], z[N] + 1e-30

    # Zeros of the weight function = new poles, forced stable
    H = diag(poles) - outer(ones(N), c_tilde / d_tilde)
    new_poles = eigvals(H)
    new_poles = -abs(new_poles.real) + 1j * new_poles.imag

    if real_only:
        return sort_complex(new_poles.real.astype(complex))
    return _enforce_conjugates(new_poles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class VFModel:
    """Rational approximation model from Vector Fitting.

    Represents a transfer function of the form:

    .. math::

        \\mathbf{g}(\\lambda) \\approx \\mathbf{d} + \\mathbf{e}\\lambda
        + \\sum_{k=1}^{K} \\frac{\\mathbf{r}_k}{\\lambda + q_k}

    where :math:`q_k` are the poles and :math:`\\mathbf{r}_k` the residues,
    both shared across all response channels.

    Parameters
    ----------
    poles : ndarray, shape (N,)
        Pole locations (complex or real).
    residues : ndarray, shape (N, A)
        Residue matrix (A = number of response channels).
    d : ndarray, shape (A,)
        Direct (constant) term.
    e : ndarray, shape (A,)
        Linear term (defaults to zeros).
    rmse : float
        Root-mean-square error of the fit.
    iters : int
        Number of pole relocation iterations performed.

    See Also
    --------
    sgwt.util.VFKernel : Lightweight kernel representation for graph convolution.

    Examples
    --------
    General-purpose fitting of frequency response data:

    >>> import numpy as np
    >>> s = 1j * np.linspace(1, 100, 200)
    >>> f = 1.0 / (s + 5) + 2.0 / (s + 20)
    >>> model = VFModel.fit(s, f, n_poles=2)
    >>> model
    VFModel(2 poles, 1 ch, rmse=..., ... iters)

    Graph kernel fitting for use with :class:`sgwt.Convolve`:

    >>> lam = np.linspace(0.01, 100, 500)
    >>> g = 1.0 / (lam + 1)
    >>> K = VFModel.kernel(lam, g, n_poles=4)
    """

    poles: ndarray
    """Pole locations, shape ``(N,)``."""

    residues: ndarray
    """Residue matrix, shape ``(N, A)``."""

    d: ndarray
    """Direct (constant) term, shape ``(A,)``."""

    e: ndarray
    """Linear term, shape ``(A,)``. Zero for graph kernels."""

    rmse: float
    """Root-mean-square error of the fit."""

    iters: int
    """Number of pole relocation iterations performed."""

    def __repr__(self):
        return (
            f"VFModel({len(self.poles)} poles, {self.residues.shape[1]} ch, "
            f"rmse={self.rmse:.3e}, {self.iters} iters)"
        )

    def __call__(self, s):
        """Evaluate the rational model at sample points.

        Parameters
        ----------
        s : array_like
            Points at which to evaluate the model.

        Returns
        -------
        ndarray, shape (len(s), A)
            Model response at each point.
        """
        s = asarray(s, dtype=complex).ravel()
        return _cauchy(s, self.poles) @ self.residues + self.d + s[:, None] * self.e

    # ------------------------------------------------------------------
    # Fitting classmethods
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, s, f, n_poles, *, poles=None, iters=30, tol=1e-9,
            fit_d=True, fit_e=False, real_only=False):
        """General-purpose Vector Fitting.

        Iteratively relocates poles to minimize the least-squares error
        of a partial-fraction rational approximation.

        Parameters
        ----------
        s : array_like
            Sample points (complex or real), shape ``(M,)``.
        f : array_like
            Response data, shape ``(M,)`` or ``(M, A)`` for multi-channel.
        n_poles : int
            Number of poles to fit.
        poles : array_like, optional
            Custom initial poles. If *None*, poles are auto-generated
            using logarithmic spacing across the data range.
        iters : int, default 30
            Maximum number of pole relocation iterations.
        tol : float, default 1e-9
            Convergence tolerance on relative pole movement.
        fit_d : bool, default True
            Include a constant (direct) term in the model.
        fit_e : bool, default False
            Include a linear term in the model.
        real_only : bool, default False
            Force all poles to be real and negative.

        Returns
        -------
        VFModel
            Fitted rational model.
        """
        s = asarray(s, dtype=complex).ravel()
        f = atleast_2d(asarray(f, dtype=complex))
        if f.shape[0] != len(s):
            f = f.T

        a = (asarray(poles, complex)
             if poles is not None
             else _initial_poles(s, n_poles, real_only))

        for it in range(iters):
            a_prev = a.copy()
            a = _relocate(s, f, a, fit_d, fit_e, real_only)
            if (abs(a - a_prev) / (abs(a_prev) + 1e-15)).max() < tol:
                break

        X = _lstsq(_basis(s, a, fit_d, fit_e), f)
        N, A = len(a), f.shape[1]
        d = X[N] if fit_d else zeros(A, complex)
        e = X[N + fit_d] if fit_e else zeros(A, complex)

        model = cls(a, X[:N], d, e, 0.0, it + 1)
        model.rmse = float(sqrt(mean(abs(f - model(s)) ** 2)))
        return model

    @classmethod
    def kernel(cls, lam, g, n_poles, **kw):
        """Fit a graph spectral kernel with real negative poles.

        Performs a constrained Vector Fitting and returns a
        :class:`~sgwt.util.VFKernel` ready for use with
        :class:`~sgwt.Convolve` and :class:`~sgwt.DyConvolve`.
        Enforces ``real_only=True`` and ``fit_e=False`` for stability
        in the CHOLMOD convolution pipeline.

        The resulting kernel satisfies:

        .. math::

            \\mathbf{g}(\\lambda) \\approx \\mathbf{d}
            + \\sum_{k=1}^{K} \\frac{\\mathbf{r}_k}{\\lambda + q_k}

        where all :math:`q_k > 0` are real.

        Parameters
        ----------
        lam : array_like
            Eigenvalues (real, non-negative).
        g : array_like
            Target filter response evaluated at *lam*, shape ``(M,)``
            or ``(M, A)``.
        n_poles : int
            Number of poles.
        **kw
            Forwarded to :meth:`fit`. ``fit_d`` defaults to True.

        Returns
        -------
        VFKernel
            Kernel with real negative poles, ready for graph convolution.
        """
        from ..util import VFKernel

        kw.setdefault('fit_d', True)
        kw['fit_e'] = False
        model = cls.fit(asarray(lam, float), asarray(g, float),
                        n_poles, real_only=True, **kw)
        return VFKernel(R=model.residues, Q=model.poles, D=model.d)