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
from typing import Optional, List, Tuple, Dict, NamedTuple
from scipy.stats import gaussian_kde

# Optional dependency for peak finding, with a NumPy/SciPy fallback
try:
    from skimage.feature import peak_local_max
except ImportError:
    from scipy.ndimage import maximum_filter

    def _peak_local_max_fallback(
        image: np.ndarray, min_distance: int = 1, num_peaks: int = np.inf, exclude_border: bool = False
    ) -> np.ndarray:
        """
        Fallback for scikit-image's peak_local_max using SciPy.

        Finds peaks in an image and returns them as coordinates.
        Peaks are the local maxima in a region of `2 * min_distance + 1`.
        This is a simplified implementation that uses a square neighborhood
        for non-maximum suppression.
        """
        if min_distance < 1:
            min_distance = 1

        # Find all pixels that are local maxima in a (2*min_dist + 1) neighborhood
        size = 2 * min_distance + 1
        local_max = image == maximum_filter(image, size=size, mode="constant")

        # Exclude peaks with zero magnitude
        local_max[image == 0] = False

        # Get coordinates of candidate peaks
        coords = np.argwhere(local_max)
        if coords.shape[0] == 0:
            return np.empty((0, image.ndim), dtype=np.intp)

        # Sort candidates by magnitude in descending order
        magnitudes = image[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(magnitudes)[::-1]
        coords = coords[sort_idx]

        # Iteratively select peaks and suppress neighbors (non-maximum suppression)
        final_coords = []
        is_suppressed = np.zeros(image.shape, dtype=bool)
        for r, c in coords:
            if not is_suppressed[r, c]:
                final_coords.append([r, c])
                if len(final_coords) == num_peaks:
                    break
                r_min, r_max = max(0, r - min_distance), min(image.shape[0], r + min_distance + 1)
                c_min, c_max = max(0, c - min_distance), min(image.shape[1], c + min_distance + 1)
                is_suppressed[r_min:r_max, c_min:c_max] = True

        return np.array(final_coords, dtype=np.intp)

    peak_local_max = _peak_local_max_fallback

from .cholconv import DyConvolve
from .functions import gaussian_wavelet
from .util import impulse


NetworkAnalysisResult = NamedTuple('NetworkAnalysisResult', [('peaks', Dict), ('clusters', Dict)])


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
        Must be symmetric positive semi-definite.
    scales : array_like
        Spatial scales for the SGWT. Logarithmically spaced values are
        recommended (e.g., ``np.geomspace(1e-3, 1e1, 150)``).
    freqs : array_like
        Temporal frequencies (in Hz) to analyze.
    order : int, optional
        Order of the bandpass filter. Default is 10.

    Attributes
    ----------
    Ts : ndarray
        Temporal scales (seconds) derived from frequencies and w0.
    wavlen : ndarray
        Approximate wavelengths (``sqrt(s)``) for each spatial scale.
    poles : list
        Poles for DyConvolve, computed as ``1/scale``.

    See Also
    --------
    DyConvolve : Dynamic convolution context with pre-factored poles.
    gaussian_wavelet : Temporal wavelet generating kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from sgwt import SGMA
    >>> # L is a sparse graph Laplacian matrix
    >>> scales = np.geomspace(1e-2, 1e1, 50)
    >>> freqs = np.linspace(0.1, 2.0, 60)
    >>> sgma = SGMA(L, scales=scales, freqs=freqs)
    >>> # V is a signal matrix (buses x time), t is a time vector
    >>> peaks = sgma.analyze(V, t, bus=0, time=5.0, top_n=5)
    >>> sgma.close()  # Release resources when done
    """

    def __init__(
        self,
        L,
        scales: np.ndarray,
        freqs: np.ndarray,
        order: int = 10,
        w0: float = 2 * np.pi
    ):
        self.L = L
        self.scales = np.atleast_1d(scales)
        self.freqs = np.atleast_1d(freqs)
        self.order = order
        self.w0 = w0

        # Derived parameters
        self.Ts = self.w0 / (2 * np.pi * self.freqs)
        self.wavlen = np.sqrt(self.scales)     # Wavelength approximation

        # Convert scales to poles for DyConvolve: pole = 1/scale
        self.poles = [1.0 / scale for scale in self.scales]

        # Cached convolution context (lazy initialization)
        self._conv: Optional[DyConvolve] = None
        
        # Cached temporal matrix
        self._B: Optional[np.ndarray] = None
        self._t_cached: Optional[np.ndarray] = None
        self._time_target_cached: Optional[float] = None

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

    def _build_temporal_matrix(self, t: np.ndarray, time_target: float) -> np.ndarray:
        """
        Construct the temporal wavelet matrix R_τ.

        Builds the right transformation matrix in the joint wavelet
        transform using Gaussian wavelets centered at ``time_target``.
        Caches the result to avoid recomputation for repeated calls.

        Parameters
        ----------
        t : ndarray
            Time vector of shape ``(n_time,)``.
        time_target : float
            The time instant (in seconds) to center the temporal wavelet.

        Returns
        -------
        ndarray
            Temporal wavelet matrix of shape ``(n_time, n_freqs)``.
        """
        # Check if we can use cached matrix
        if (self._B is not None and self._t_cached is not None and
                self._time_target_cached is not None):
            if (len(t) == len(self._t_cached) and np.allclose(t, self._t_cached) and
                    self._time_target_cached == time_target):
                return self._B
        
        # Build and cache
        self._B = np.stack([
            gaussian_wavelet(t, a=sc, b=time_target, w0=self.w0)
            for sc in self.Ts
        ]).T
        self._t_cached = t.copy()
        self._time_target_cached = time_target
        return self._B

    def spectrum(
        self,
        V: np.ndarray,
        t: np.ndarray,
        bus: int,
        time: float,
        VB: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the SGMA spectrum magnitude at a specific bus and time.
        
        Parameters
        ----------
        V : ndarray
            Voltage signal matrix of shape ``(n_buses, n_time)``.
        t : ndarray
            Time vector of shape ``(n_time,)``.
        bus : int
            Bus index for localized analysis.
        time : float
            The time instant (in seconds) to center the temporal wavelet.
        VB : ndarray, optional
            Pre-computed ``V @ B`` matrix. If provided, skips temporal
            matrix multiplication. Note: The provided VB must be computed
            for the given `time`.
        
        Returns
        -------
        ndarray
            Spectrum magnitude of shape ``(n_scales, n_freqs)``.
        """
        # Validate bus to prevent out-of-bounds errors
        n_buses = self.L.shape[0]
        if not (0 <= bus < n_buses):
            raise ValueError(f"bus {bus} is out of bounds for the graph with {n_buses} nodes.")

        # 1. Temporal Transform - use pre-computed VB if available
        if VB is None:
            B = self._build_temporal_matrix(t, time_target=time)
            VB = V @ B
        
        # 2. Spatial Transform using DyConvolve singleton method
        conv = self._get_conv()
        X_imp = impulse(self.L, n=bus)
        spatial_responses = conv.bandpass(X_imp, order=self.order)
        
        # Build spatial transform matrix A (L_n in formulation)
        A = np.column_stack([resp.flatten() for resp in spatial_responses]).T
        
        # 3. Joint transform: m_{n,τ} ≈ L_n (V @ B)
        Y = A @ VB
        
        return np.sqrt(np.abs(Y))

    def analyze(
        self,
        V: np.ndarray,
        t: np.ndarray,
        bus: int,
        time: float,
        top_n: int = 5,
        min_dist: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform a full SGMA analysis for a single bus.

        This is a convenience method that combines computing the spectrum
        and finding peaks.

        Parameters
        ----------
        V : ndarray
            Signal matrix of shape ``(n_buses, n_time)``.
        t : ndarray
            Time vector of shape ``(n_time,)``.
        bus : int
            Bus index for localized analysis.
        time : float
            The time instant (in seconds) to center the temporal wavelet.
        top_n : int, optional
            Maximum peaks to return. Default is 5.
        min_dist : int, optional
            Minimum index distance between peaks. Default is 5.

        Returns
        -------
        dict
            Peak information with keys: 'Wavelength', 'Frequency', 'Magnitude'.
        """
        Y_mag = self.spectrum(V, t, bus=bus, time=time)
        return self.find_peaks(Y_mag, top_n=top_n, min_dist=min_dist)

    def find_peaks(
        self,
        spectrum: np.ndarray,
        top_n: int = 5,
        min_dist: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Identify local maxima in the transform magnitude.

        Uses morphological peak detection to find dominant modes
        in the wavelength-frequency plane.

        Parameters
        ----------
        spectrum: ndarray
            Spectrum magnitude of shape ``(n_scales, n_freqs)``.
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

        Notes
        -----
        Larger wavelengths correspond to inter-area oscillation modes,
        while smaller wavelengths indicate local oscillations [1].
        """
        # Ensure the input is real-valued magnitude for peak detection
        Y_mag = np.abs(spectrum)

        # We use exclude_border=False to ensure peaks near the edges of the 
        # spectrum (e.g. high frequency or large scale) are not lost.
        coords = peak_local_max(Y_mag, min_distance=min_dist, num_peaks=top_n, exclude_border=False)

        if coords.size == 0:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Magnitude']}

        # Extract magnitudes and sort to handle case where peak_local_max doesn't sort
        magnitudes = Y_mag[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(magnitudes)[::-1]

        return {
            'Wavelength': self.wavlen[coords[sort_idx, 0]],
            'Frequency': self.freqs[coords[sort_idx, 1]],
            'Magnitude': magnitudes[sort_idx]
        }
    
    def analyze_many(
        self,
        V: np.ndarray,
        t: np.ndarray,
        time: float,
        buses: Optional[List[int]] = None,
        top_n: int = 5,
        min_dist: int = 5,
        verbose: bool = True
    ) -> NetworkAnalysisResult:
        """
        Extract peaks from SGMA transforms across many specified buses.
        
        Pre-computes ``V @ B`` once and reuses it for all buses,
        significantly reducing computation time [1].

        Parameters
        ----------
        V : ndarray
            Signal matrix of shape ``(n_buses, n_time)``.
        t : ndarray
            Time vector of shape ``(n_time,)``.
        time : float
            The time instant (in seconds) to center the temporal wavelet.
        buses : list of int, optional
            List of bus indices to analyze. If None, all buses are used.
        top_n : int, optional
            Maximum peaks to return per bus. Default is 5.
        min_dist : int, optional
            Minimum index distance between peaks. Default is 5.
        verbose : bool, optional
            If True, prints progress updates. Default is True.

        Returns
        -------
        NetworkAnalysisResult
            A named tuple containing:
            - ``peaks``: dict with 'Wavelength', 'Frequency', 'Magnitude', 'Bus_ID'.
            - ``clusters``: dict with 'Wavelength', 'Frequency', 'Density'.
        """
        if buses is None:
            buses = list(range(V.shape[0]))
        
        n_buses = len(buses)
        
        # Pre-compute temporal matrix and V @ B (constant across all buses)
        B = self._build_temporal_matrix(t, time_target=time)
        VB = V @ B  # Computed ONCE, not n_buses times
        
        all_w, all_f, all_m, all_b = [], [], [], []

        for i, bus_idx in enumerate(buses):
            # Pass pre-computed VB to avoid redundant multiplication
            Y = self.spectrum(V, t, bus=bus_idx, time=time, VB=VB)
            
            p = self.find_peaks(Y, top_n=top_n, min_dist=min_dist)
            if p['Wavelength'].size > 0:
                all_w.append(p['Wavelength'])
                all_f.append(p['Frequency'])
                all_m.append(p['Magnitude'])
                all_b.append(np.full(p['Wavelength'].shape, bus_idx, dtype=int))
            
            if verbose and (i + 1) % 50 == 0: # pragma: no cover
                print(f"  Processed {i + 1}/{n_buses} buses...")
        
        empty_peaks = {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Magnitude', 'Bus_ID']}
        empty_clusters = {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

        if not all_w:
            return NetworkAnalysisResult(peaks=empty_peaks, clusters=empty_clusters)
        
        master_peaks = {
            'Wavelength': np.concatenate(all_w),
            'Frequency': np.concatenate(all_f),
            'Magnitude': np.concatenate(all_m),
            'Bus_ID': np.concatenate(all_b)
        }
        
        # --- Density Clustering ---
        cluster_peaks = self._compute_density_clusters(master_peaks, top_n, min_dist)

        return NetworkAnalysisResult(peaks=master_peaks, clusters=cluster_peaks)

    def _compute_density_clusters(self, peaks_dict: Dict[str, np.ndarray], top_n: int, min_dist: int) -> Dict[str, np.ndarray]:
        """Helper to compute density-based clusters from peak data."""
        if peaks_dict['Wavelength'].size < 2:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

        try:
            x, y = np.log10(peaks_dict['Wavelength']), peaks_dict['Frequency']
            kernel = gaussian_kde(np.vstack([x, y]))
            
            X_grid, Y_grid = np.meshgrid(np.log10(self.wavlen), self.freqs, indexing='ij')
            Z = kernel(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)
            
            cluster_peaks = self.find_peaks(Z, top_n=top_n, min_dist=min_dist)
            
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
        self._time_target_cached = None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()