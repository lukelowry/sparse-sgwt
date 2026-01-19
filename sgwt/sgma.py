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
from typing import Optional, List, Dict, NamedTuple
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


class ModeTable:
    """
    Container for identified oscillatory modes with tabular display.

    Stores mode parameters (frequency, damping ratio, wavelength, magnitude)
    and provides a formatted string representation for easy inspection.

    Attributes
    ----------
    frequency : ndarray
        Oscillation frequencies in Hz.
    damping : ndarray
        Damping ratios (dimensionless). Positive values indicate stable modes.
    wavelength : ndarray
        Spatial wavelengths (sqrt of spatial scale). Larger values indicate
        inter-area modes; smaller values indicate local modes.
    magnitude : ndarray
        Transform magnitudes at peak locations.
    """

    def __init__(self, frequency: np.ndarray, damping: np.ndarray,
                 wavelength: np.ndarray, magnitude: np.ndarray):
        self.frequency = np.atleast_1d(frequency)
        self.damping = np.atleast_1d(damping)
        self.wavelength = np.atleast_1d(wavelength)
        self.magnitude = np.atleast_1d(magnitude)
        self.n_modes = len(self.frequency)

    def __repr__(self) -> str:
        if self.n_modes == 0:
            return "ModeTable (empty - no modes identified)"

        lines = [
            f"ModeTable ({self.n_modes} mode{'s' if self.n_modes > 1 else ''} identified)",
            "-" * 60,
            f"{'#':>3}  {'Freq (Hz)':>10}  {'Damping':>10}  {'Wavelength':>12}  {'Magnitude':>10}",
            "-" * 60,
        ]

        for i in range(self.n_modes):
            lines.append(
                f"{i+1:>3}  {self.frequency[i]:>10.4f}  {self.damping[i]:>10.4f}  "
                f"{self.wavelength[i]:>12.2f}  {self.magnitude[i]:>10.4f}"
            )

        lines.append("-" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return mode data as a dictionary."""
        return {
            'Frequency': self.frequency,
            'Damping': self.damping,
            'Wavelength': self.wavelength,
            'Magnitude': self.magnitude
        }

    def to_array(self) -> np.ndarray:
        """Return mode data as a 2D array (n_modes x 4)."""
        return np.column_stack([
            self.frequency, self.damping, self.wavelength, self.magnitude
        ])


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
    matrix centered at time :math:`\\tau`.

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

        # Derived: temporal scale a = w0/(2πf) gives wavelet centered at frequency f
        self.Ts = self.w0 / (2 * np.pi * self.freqs)
        self.wavlen = np.sqrt(self.scales)
        self.poles = [1.0 / scale for scale in self.scales]

        # Lazy-initialized caches
        self._conv: Optional[DyConvolve] = None
        self._B: Optional[np.ndarray] = None
        self._t_cached: Optional[np.ndarray] = None
        self._time_target_cached: Optional[float] = None

    def _get_conv(self) -> DyConvolve:
        """Get or lazily create the DyConvolve context."""
        if self._conv is None:
            self._conv = DyConvolve(self.L, poles=self.poles)
            self._conv.__enter__()
        return self._conv

    def _build_temporal_matrix(self, t: np.ndarray, time_target: float) -> np.ndarray:
        """
        Construct the temporal wavelet matrix R_τ (cached).

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
        if (self._B is not None and self._t_cached is not None and
                self._time_target_cached is not None):
            if (len(t) == len(self._t_cached) and np.allclose(t, self._t_cached) and
                    self._time_target_cached == time_target):
                return self._B

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
        VB: Optional[np.ndarray] = None,
        return_complex: bool = False
    ) -> np.ndarray:
        """
        Compute the SGMA spectrum at a specific bus and time.

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
        return_complex : bool, optional
            If True, return the complex spectrum instead of magnitude.
            Default is False (returns magnitude).

        Returns
        -------
        ndarray
            Spectrum of shape ``(n_scales, n_freqs)``. Complex if
            ``return_complex=True``, otherwise magnitude (sqrt of abs).
        """
        n_buses = self.L.shape[0]
        if not (0 <= bus < n_buses):
            raise ValueError(f"bus {bus} is out of bounds for the graph with {n_buses} nodes.")

        # Temporal transform
        if VB is None:
            B = self._build_temporal_matrix(t, time_target=time)
            VB = V @ B

        # Spatial transform
        conv = self._get_conv()
        X_imp = impulse(self.L, n=bus)
        spatial_responses = conv.bandpass(X_imp, order=self.order)
        A = np.column_stack([resp.flatten() for resp in spatial_responses]).T

        # Joint transform: m_{n,τ} ≈ L_n (V @ B)
        Y = A @ VB

        return Y if return_complex else np.abs(Y)

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
        min_dist: int = 5,
        return_indices: bool = False
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
        return_indices : bool, optional
            If True, also return 'ScaleIdx' and 'FreqIdx' arrays containing
            the indices into the spectrum array. Default is False.

        Returns
        -------
        dict
            Peak information with keys:

            - ``Wavelength``: ndarray of spatial wavelengths (sqrt of scale)
            - ``Frequency``: ndarray of temporal frequencies in Hz
            - ``Magnitude``: ndarray of transform magnitudes at peaks
            - ``ScaleIdx``: (if return_indices=True) ndarray of scale indices
            - ``FreqIdx``: (if return_indices=True) ndarray of frequency indices

        """
        Y_mag = np.abs(spectrum)
        coords = peak_local_max(Y_mag, min_distance=min_dist, num_peaks=top_n, exclude_border=False)

        if coords.size == 0:
            keys = ['Wavelength', 'Frequency', 'Magnitude']
            if return_indices:
                keys.extend(['ScaleIdx', 'FreqIdx'])
            return {k: np.array([]) for k in keys}

        magnitudes = Y_mag[coords[:, 0], coords[:, 1]]
        sort_idx = np.argsort(magnitudes)[::-1]
        sorted_coords = coords[sort_idx]

        result = {
            'Wavelength': self.wavlen[sorted_coords[:, 0]],
            'Frequency': self.freqs[sorted_coords[:, 1]],
            'Magnitude': magnitudes[sort_idx]
        }

        if return_indices:
            result['ScaleIdx'] = sorted_coords[:, 0]
            result['FreqIdx'] = sorted_coords[:, 1]

        return result

    def find_modes(
        self,
        spectrum: np.ndarray,
        top_n: int = 5,
        min_dist: int = 5
    ) -> ModeTable:
        """
        Identify oscillatory modes with frequency, damping ratio, and magnitude.

        Extends peak detection by computing damping ratios from the phase slope
        of the complex spectrum. The damping ratio :math:`\\zeta` is estimated
        at each peak using the phase-frequency relationship:

        .. math::

            \\zeta = -\\frac{1}{f_0 \\cdot d\\phi/df}

        where :math:`f_0` is the peak frequency and :math:`d\\phi/df` is computed
        using centered finite differences on the unwrapped phase.

        Parameters
        ----------
        spectrum : ndarray
            Complex spectrum of shape ``(n_scales, n_freqs)`` from
            ``spectrum(..., return_complex=True)``.
        top_n : int, optional
            Maximum number of modes to identify. Default is 5.
        min_dist : int, optional
            Minimum index distance between peaks. Default is 5.

        Returns
        -------
        ModeTable
            Table with identified modes. Attributes include ``frequency``,
            ``damping``, ``wavelength``, and ``magnitude``.

        Examples
        --------
        >>> M = sgma.spectrum(V, t, bus=0, time=5.0, return_complex=True)
        >>> modes = sgma.find_modes(M, top_n=5)
        >>> print(modes)
        ModeTable (5 modes identified)
        ------------------------------------------------------------
          #   Freq (Hz)     Damping    Wavelength   Magnitude
        ------------------------------------------------------------
          1      0.3500      0.0423         45.21      1.2345
          2      0.7200      0.0156         12.34      0.9876
        ...
        """
        if not np.iscomplexobj(spectrum):
            raise ValueError(
                "find_modes requires the complex spectrum. "
                "Use spectrum(..., return_complex=True)."
            )

        Y_mag = np.abs(spectrum)
        peaks = self.find_peaks(Y_mag, top_n=top_n, min_dist=min_dist, return_indices=True)

        if peaks['Frequency'].size == 0:
            return ModeTable(
                frequency=np.array([]),
                damping=np.array([]),
                wavelength=np.array([]),
                magnitude=np.array([])
            )

        # Unwrap phase to handle discontinuities at ±π
        #z = np.log(spectrum) # sigma + j phase
        
        phase =  np.unwrap(np.angle(spectrum), axis=1)

        scale_idx = peaks['ScaleIdx']
        freq_idx = peaks['FreqIdx']
        n_freqs = len(self.freqs)

        # Compute phase slope using centered differences (boundary conditions if at edge)
        phase_slope = np.zeros(len(freq_idx))
        for i, (si, fi) in enumerate(zip(scale_idx, freq_idx)):
            if fi > 0 and fi < n_freqs - 1:
                phase_slope[i] = (phase[si, fi + 1] - phase[si, fi - 1]) / \
                                 (self.freqs[fi + 1] - self.freqs[fi - 1])
            elif fi == 0:
                phase_slope[i] = (phase[si, fi + 1] - phase[si, fi]) / \
                                 (self.freqs[fi + 1] - self.freqs[fi])
            else:
                phase_slope[i] = (phase[si, fi] - phase[si, fi - 1]) / \
                                 (self.freqs[fi] - self.freqs[fi - 1])

        # Damping: ζ = -1/(f₀ · dφ/df) from 2nd-order transfer function theory
        f0 = peaks['Frequency']
        with np.errstate(divide='ignore', invalid='ignore'):
            damping = -1.0 / (f0 * phase_slope)
            damping = np.where(np.isfinite(damping), damping, 0.0)

        return ModeTable(
            frequency=f0,
            damping=damping,
            wavelength=peaks['Wavelength'],
            magnitude=peaks['Magnitude']
        )

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
        B = self._build_temporal_matrix(t, time_target=time)
        VB = V @ B

        all_w, all_f, all_m, all_b = [], [], [], []
        for i, bus_idx in enumerate(buses):
            Y = self.spectrum(V, t, bus=bus_idx, time=time, VB=VB)
            p = self.find_peaks(Y, top_n=top_n, min_dist=min_dist)
            if p['Wavelength'].size > 0:
                all_w.append(p['Wavelength'])
                all_f.append(p['Frequency'])
                all_m.append(p['Magnitude'])
                all_b.append(np.full(p['Wavelength'].shape, bus_idx, dtype=int))

            if verbose and (i + 1) % 50 == 0:  # pragma: no cover
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

        cluster_peaks = self._compute_density_clusters(master_peaks, top_n, min_dist)
        return NetworkAnalysisResult(peaks=master_peaks, clusters=cluster_peaks)

    def _compute_density_clusters(self, peaks_dict: Dict[str, np.ndarray], top_n: int, min_dist: int) -> Dict[str, np.ndarray]:
        """Compute density-based clusters from peak data using KDE."""
        if peaks_dict['Wavelength'].size < 2:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

        try:
            x, y = np.log10(peaks_dict['Wavelength']), peaks_dict['Frequency']
            kernel = gaussian_kde(np.vstack([x, y]))

            X_grid, Y_grid = np.meshgrid(np.log10(self.wavlen), self.freqs, indexing='ij')
            Z = kernel(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)

            cluster_peaks = self.find_peaks(Z, top_n=top_n, min_dist=min_dist)
            cluster_peaks['Density'] = cluster_peaks.pop('Magnitude')
            return cluster_peaks
        except Exception:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}

    def close(self):
        """Release cached convolution resources and free CHOLMOD memory."""
        if self._conv is not None:
            self._conv.__exit__(None, None, None)
            self._conv = None
        self._B = None
        self._t_cached = None
        self._time_target_cached = None

    def __del__(self):
        self.close()