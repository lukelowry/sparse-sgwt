# -*- coding: utf-8 -*-
"""
Spectral Graph Modal Analysis (SGMA)
------------------------------------
Module for performing joint spatial-temporal wavelet analysis on graph signals.

This module implements the SGMA framework for identifying oscillatory modes
in power system time-domain responses through joint wavelet transformation.

The joint wavelet transform decomposes a time-vertex signal into the 
wavenumber-frequency domain, enabling scale-dependent mode identification [1].

Author: Luke Lowery (lukel@tamu.edu)

"""
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy.stats import gaussian_kde

# Optional dependency for peak finding
try:
    from skimage.feature import peak_local_max
except ImportError:
    peak_local_max = None

from .cholconv import DyConvolve
from .functions import gaussian_wavelet
from .util import impulse


class SGMA:
    """
    Spectral Graph Modal Analysis (SGMA) engine.

    Encapsulates the logic for performing joint spatial-temporal wavelet
    transforms on graph signals to identify oscillatory modes. The SGMA
    decomposes a time-vertex signal into the wavenumber-frequency domain,
    enabling scale-dependent mode identification [1].

    The joint wavelet transform is computed as:

    .. math::

        m_{n,\\tau}(\\Lambda \\times S) \\approx L_n X R_\\tau

    where :math:`L_n` is the SGWT localized at bus :math:`n`, :math:`X` is
    the time-vertex signal, and :math:`R_\\tau` is the temporal wavelet
    matrix centered at time :math:`\\tau` [2].

    Parameters
    ----------
    L : csc_matrix
        The graph Laplacian matrix of shape ``(n_buses, n_buses)``.
        Must be symmetric positive semi-definite. Branch weights should
        be squared inverse distances for wavelength interpretation [1].
    s : array_like
        Spatial scales for the SGWT. Logarithmically spaced values are
        recommended (e.g., ``np.geomspace(1e-3, 1e1, 150)``).
    freqs : array_like
        Temporal frequencies (in Hz) to analyze.
    time_target : float
        The time instant (in seconds) to center the temporal wavelet.
    order : int, optional
        Order of the bandpass filter. Default is 10.

    Attributes
    ----------
    Ts : ndarray
        Temporal scales (seconds) derived from frequencies and w0.
    wavlen : ndarray
        Approximate wavelengths (``sqrt(s)``) for each spatial scale.
    poles : list
        Poles for DyConvolve, computed as ``1/scale`` [1].

    See Also
    --------
    DyConvolve : Dynamic convolution context with pre-factored poles.
    gaussian_wavelet : Temporal wavelet generating kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from sgwt import SGMA
    >>> scales = np.geomspace(1e-2, 1e1, 50)
    >>> freqs = np.linspace(0.1, 2.0, 60)
    >>> sgma = SGMA(L, s=scales, freqs=freqs, time_target=5.0)
    >>> Y_mag = sgma.transform(V, t, bus_idx=0)
    >>> peaks = sgma.peaks_from_spectrum(Y_mag, top_n=5)
    >>> sgma.close()  # Release resources when done
    """

    def __init__(
        self,
        L,
        s: np.ndarray,
        freqs: np.ndarray,
        time_target: float,
        order: int = 10,
        w0: float = 2 * np.pi
    ):
        self.L = L
        self.s = np.atleast_1d(s)
        self.freqs = np.atleast_1d(freqs)
        self.time_target = time_target
        self.order = order
        self.w0 = w0

        # Derived parameters
        self.Ts = self.w0 / (2 * np.pi * self.freqs)
        self.wavlen = np.sqrt(self.s)          # Wavelength approximation

        # Convert scales to poles for DyConvolve: pole = 1/scale [1]
        self.poles = [1.0 / scale for scale in self.s]

        # Cached convolution context (lazy initialization)
        self._conv: Optional[DyConvolve] = None
        
        # Cached temporal matrix
        self._B: Optional[np.ndarray] = None
        self._t_cached: Optional[np.ndarray] = None

    def _get_conv(self) -> DyConvolve:
        """
        Get or create the DyConvolve context.

        DyConvolve pre-factors all shifted systems ``(L + qI)`` at 
        initialization, making repeated bandpass operations efficient [1].

        Returns
        -------
        DyConvolve
            Convolution context with pre-factored poles.
        """
        if self._conv is None:
            self._conv = DyConvolve(self.L, poles=self.poles)
            self._conv.__enter__()
        return self._conv

    def _build_temporal_matrix(self, t: np.ndarray) -> np.ndarray:
        """
        Construct the temporal wavelet matrix R_τ.

        Builds the right transformation matrix in the joint wavelet
        transform using Gaussian wavelets centered at ``time_target``.
        Caches the result to avoid recomputation for repeated calls.

        Parameters
        ----------
        t : ndarray
            Time vector of shape ``(n_time,)``.

        Returns
        -------
        ndarray
            Temporal wavelet matrix of shape ``(n_time, n_freqs)``.
        """
        # Check if we can use cached matrix
        if self._B is not None and self._t_cached is not None:
            if len(t) == len(self._t_cached) and np.allclose(t, self._t_cached):
                return self._B
        
        # Build and cache
        self._B = np.stack([
            gaussian_wavelet(t, a=sc, b=self.time_target, w0=self.w0)
            for sc in self.Ts
        ]).T
        self._t_cached = t.copy()
        return self._B

    def transform(
        self,
        V: np.ndarray,
        t: np.ndarray,
        bus_idx: int,
        VB: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the SGMA transform magnitude at a specific bus.
        
        Parameters
        ----------
        V : ndarray
            Voltage signal matrix of shape ``(n_buses, n_time)``.
        t : ndarray
            Time vector of shape ``(n_time,)``.
        bus_idx : int
            Bus index for localized analysis (required).
        VB : ndarray, optional
            Pre-computed ``V @ B`` matrix. If provided, skips temporal
            matrix multiplication for efficiency in batch operations.
        
        Returns
        -------
        ndarray
            Transform magnitude of shape ``(n_scales, n_freqs)``.
        """
        # Validate bus_idx to prevent out-of-bounds errors
        n_buses = self.L.shape[0]
        if not (0 <= bus_idx < n_buses):
            raise ValueError(f"bus_idx {bus_idx} is out of bounds for the graph with {n_buses} nodes.")

        # 1. Temporal Transform - use pre-computed VB if available
        if VB is None:
            B = self._build_temporal_matrix(t)
            VB = V @ B
        
        # 2. Spatial Transform using DyConvolve singleton method
        conv = self._get_conv()
        X_imp = impulse(self.L, n=bus_idx)
        spatial_responses = conv.bandpass(X_imp, order=self.order)
        
        # Build spatial transform matrix A (L_n in formulation)
        A = np.column_stack([resp.flatten() for resp in spatial_responses]).T
        
        # 3. Joint transform: m_{n,τ} ≈ L_n (V @ B)
        Y = A @ VB
        
        return np.sqrt(np.abs(Y))

    def peaks_from_spectrum(
        self,
        Y: np.ndarray,
        top_n: int = 5,
        min_dist: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Identify local maxima in the transform magnitude.

        Uses morphological peak detection to find dominant modes
        in the wavelength-frequency plane.

        Parameters
        ----------
        Y: ndarray
            Transform magnitude of shape ``(n_scales, n_freqs)``.
        top_n : int, optional
            Maximum peaks to return. Default is 5.
        min_dist : int, optional
            Minimum index distance between peaks. Default is 5.

        Returns
        -------
        dict
            Peak information with keys:

            - ``Wavelength``: ndarray of spatial wavelengths (sqrt of scale)
            - ``Frequency``: ndarray of temporal frequencies in Hz
            - ``Magnitude``: ndarray of transform magnitudes at peaks

        Raises
        ------
        ImportError
            If scikit-image is not installed.

        Notes
        -----
        Larger wavelengths correspond to inter-area oscillation modes,
        while smaller wavelengths indicate local oscillations [1].
        """
        if peak_local_max is None: # pragma: nocover
            raise ImportError(
                "scikit-image is required for peak finding. "
                "Install with: pip install scikit-image"
            )

        # Ensure the input is real-valued magnitude for peak detection
        Y_mag = np.abs(Y)

        coords = peak_local_max(Y_mag, min_distance=min_dist)

        if coords.size == 0:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Magnitude']}

        # Extract magnitudes and sort
        magnitudes = Y_mag[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(magnitudes)[::-1][:top_n]

        return {
            'Wavelength': self.wavlen[coords[sort_idx, 0]],
            'Frequency': self.freqs[coords[sort_idx, 1]],
            'Magnitude': magnitudes[sort_idx]
        }
    
    def find_system_wide_peaks(
        self,
        V: np.ndarray,
        t: np.ndarray,
        bus_indices: Optional[List[int]] = None,
        top_n: int = 5,
        min_dist: int = 5,
        verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract peaks from SGMA transforms across specified buses.
        
        Pre-computes ``V @ B`` once and reuses it for all buses,
        significantly reducing computation time [1].

        Returns
        -------
        tuple of (dict, dict)
            A tuple containing:
            - Master peaks: dict with 'Wavelength', 'Frequency', 'Magnitude', 'Bus_ID' (all ndarrays).
            - Clustered peaks: dict with 'Wavelength', 'Frequency', 'Density' (all ndarrays).
        """
        if bus_indices is None:
            bus_indices = list(range(V.shape[0]))
        
        n_buses = len(bus_indices)
        
        # Pre-compute temporal matrix and V @ B (constant across all buses)
        B = self._build_temporal_matrix(t)
        VB = V @ B  # Computed ONCE, not n_buses times
        
        all_w, all_f, all_m, all_b = [], [], [], []

        for i, bus_idx in enumerate(bus_indices):
            # Pass pre-computed VB to avoid redundant multiplication
            Y = self.transform(V, t, bus_idx=bus_idx, VB=VB)
            
            p = self.peaks_from_spectrum(Y, top_n=top_n, min_dist=min_dist)
            if p['Wavelength'].size > 0:
                all_w.append(p['Wavelength'])
                all_f.append(p['Frequency'])
                all_m.append(p['Magnitude'])
                all_b.append(np.full(p['Wavelength'].shape, bus_idx, dtype=int))
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_buses} buses...")
        
        if not all_w:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Magnitude', 'Bus_ID']}, \
                   {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}
        
        master_peaks = {
            'Wavelength': np.concatenate(all_w),
            'Frequency': np.concatenate(all_f),
            'Magnitude': np.concatenate(all_m),
            'Bus_ID': np.concatenate(all_b)
        }
        
        # --- Density Clustering ---
        cluster_peaks = self._compute_density_clusters(master_peaks, top_n, min_dist)

        return master_peaks, cluster_peaks

    def _compute_density_clusters(self, peaks_dict: Dict[str, np.ndarray], top_n: int, min_dist: int) -> Dict[str, np.ndarray]:
        """Helper to compute density-based clusters from peak data."""
        if peaks_dict['Wavelength'].size < 2:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

        try:
            x, y = np.log10(peaks_dict['Wavelength']), peaks_dict['Frequency']
            kernel = gaussian_kde(np.vstack([x, y]))
            
            X_grid, Y_grid = np.meshgrid(np.log10(self.wavlen), self.freqs, indexing='ij')
            Z = kernel(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
            
            cluster_peaks = self.peaks_from_spectrum(Z, top_n=top_n, min_dist=min_dist)
            
            # Rename 'Magnitude' to 'Density' for these cluster peaks
            cluster_peaks['Density'] = cluster_peaks.pop('Magnitude')
            
            return cluster_peaks
        except Exception:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

    def close(self):
        """
        Release cached convolution resources.
        
        Call this method when finished using the SGMA instance to
        free CHOLMOD memory allocations.
        """
        if self._conv is not None:
            self._conv.__exit__(None, None, None)
            self._conv = None
        
        # Clear cached matrices
        self._B = None
        self._t_cached = None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()