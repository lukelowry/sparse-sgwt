# -*- coding: utf-8 -*-
"""
Spectral Graph Modal Analysis (SGMA)
------------------------------------
Module for performing joint spatial-temporal wavelet analysis on graph signals.
Implements the SGMA framework for identifying oscillatory modes in power system
time-domain responses through joint wavelet transformation into the 
wavenumber-frequency domain.

Author: Luke Lowery (lukel@tamu.edu)
"""
import numpy as np
from typing import Optional, List, Dict, NamedTuple
from scipy.stats import gaussian_kde

try:
    from skimage.feature import peak_local_max
except ImportError:
    from scipy.ndimage import maximum_filter

    def _peak_local_max_fallback(
        image: np.ndarray, min_distance: int = 1, num_peaks: int = np.inf, exclude_border: bool = False
    ) -> np.ndarray:
        """Fallback for skimage peak_local_max using scipy maximum_filter."""
        min_distance = max(1, min_distance)
        size = 2 * min_distance + 1
        local_max = (image == maximum_filter(image, size=size, mode="constant")) & (image > 0)
        coords = np.argwhere(local_max)
        if coords.shape[0] == 0:
            return np.empty((0, image.ndim), dtype=np.intp)
        magnitudes = image[coords[:, 0], coords[:, 1]]
        coords = coords[np.argsort(magnitudes)[::-1]]
        final_coords, suppressed = [], np.zeros(image.shape, dtype=bool)
        for r, c in coords:
            if not suppressed[r, c]:
                final_coords.append([r, c])
                if len(final_coords) == num_peaks:
                    break
                r_lo, r_hi = max(0, r - min_distance), min(image.shape[0], r + min_distance + 1)
                c_lo, c_hi = max(0, c - min_distance), min(image.shape[1], c + min_distance + 1)
                suppressed[r_lo:r_hi, c_lo:c_hi] = True
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
    r"""
    Spectral Graph Modal Analysis (SGMA) engine.

    Performs joint spatial-temporal wavelet transforms on graph signals to
    identify oscillatory modes. The joint wavelet transform is computed as:

    .. math::

        W_{n,\tau}(s, f) = \langle \psi_s^{(n)}, X \phi_{f,\tau} \rangle

    where :math:`\psi_s^{(n)}` is the SGWT kernel localized at node n and scale s,
    and :math:`\phi_{f,\tau}` is the temporal wavelet at frequency f centered at
    time :math:`\tau`.

    Parameters
    ----------
    L : sparse matrix
        Graph Laplacian of shape (n_nodes, n_nodes). Must be symmetric PSD.
    scales : array_like
        Spatial scales for the SGWT (recommend log-spaced).
    freqs : array_like
        Temporal frequencies (Hz) to analyze.
    order : int, optional
        Bandpass filter order. Default is 10.
    w0 : float, optional
        Wavelet center frequency parameter. Default is 2*pi.

    Attributes
    ----------
    Ts : ndarray
        Temporal scales derived from frequencies.
    wavlen : ndarray
        Spatial wavelengths (sqrt of scales).
    """

    def __init__(self, L, scales: np.ndarray, freqs: np.ndarray,
                 order: int = 10, w0: float = 2 * np.pi):
        self.L = L
        self.scales = np.atleast_1d(scales)
        self.freqs = np.atleast_1d(freqs)
        self.order = order
        self.w0 = w0
        self.Ts = self.w0 / (2 * np.pi * self.freqs)
        self.wavlen = np.sqrt(self.scales)
        self.poles = [1.0 / s for s in self.scales]
        self._conv: Optional[DyConvolve] = None
        self._B: Optional[np.ndarray] = None
        self._t_cached: Optional[np.ndarray] = None
        self._time_target_cached: Optional[float] = None

    def _get_conv(self) -> DyConvolve:
        """Lazily create DyConvolve context."""
        if self._conv is None:
            self._conv = DyConvolve(self.L, poles=self.poles)
            self._conv.__enter__()
        return self._conv

    def _build_temporal_matrix(self, t: np.ndarray, time_target: float) -> np.ndarray:
        """Construct temporal wavelet matrix (cached)."""
        if (self._B is not None and self._t_cached is not None and
                len(t) == len(self._t_cached) and np.allclose(t, self._t_cached) and
                self._time_target_cached == time_target):
            return self._B
        self._B = np.stack([gaussian_wavelet(t, a=sc, b=time_target, w0=self.w0)
                           for sc in self.Ts]).T
        self._t_cached = t.copy()
        self._time_target_cached = time_target
        return self._B

    def spectrum(self, V: np.ndarray, t: np.ndarray, bus: int, time: float,
                 VB: Optional[np.ndarray] = None, return_complex: bool = False) -> np.ndarray:
        """
        Compute the SGMA spectrum at a specific bus and time.

        Parameters
        ----------
        V : ndarray
            Signal matrix of shape (n_nodes, n_time).
        t : ndarray
            Time vector of shape (n_time,).
        bus : int
            Node index for localized analysis.
        time : float
            Time instant (seconds) to center the temporal wavelet.
        VB : ndarray, optional
            Pre-computed V @ B matrix for the given time.
        return_complex : bool, optional
            If True, return complex spectrum. Default is False (magnitude).

        Returns
        -------
        ndarray
            Spectrum of shape (n_scales, n_freqs).
        """
        n_nodes = self.L.shape[0]
        if not (0 <= bus < n_nodes):
            raise ValueError(f"bus {bus} out of bounds for {n_nodes} nodes")
        if VB is None:
            VB = V @ self._build_temporal_matrix(t, time_target=time)
        conv = self._get_conv()
        spatial_responses = conv.bandpass(impulse(self.L, n=bus), order=self.order)
        A = np.column_stack([r.flatten() for r in spatial_responses]).T
        Y = A @ VB
        return Y if return_complex else np.abs(Y)

    def analyze(self, V: np.ndarray, t: np.ndarray, bus: int, time: float,
                top_n: int = 5, min_dist: int = 5) -> Dict[str, np.ndarray]:
        """Compute spectrum and find peaks for a single bus."""
        return self.find_peaks(self.spectrum(V, t, bus=bus, time=time), top_n=top_n, min_dist=min_dist)

    def find_peaks(self, spectrum: np.ndarray, top_n: int = 5, min_dist: int = 5,
                   return_indices: bool = False) -> Dict[str, np.ndarray]:
        """
        Identify local maxima in the spectrum.

        Parameters
        ----------
        spectrum : ndarray
            Spectrum of shape (n_scales, n_freqs).
        top_n : int, optional
            Maximum peaks to return.
        min_dist : int, optional
            Minimum index distance between peaks.
        return_indices : bool, optional
            If True, include ScaleIdx and FreqIdx in output.

        Returns
        -------
        dict
            Keys: Wavelength, Frequency, Magnitude, [ScaleIdx, FreqIdx].
        """
        Y_mag = np.abs(spectrum)
        coords = peak_local_max(Y_mag, min_distance=min_dist, num_peaks=top_n, exclude_border=False)
        keys = ['Wavelength', 'Frequency', 'Magnitude'] + (['ScaleIdx', 'FreqIdx'] if return_indices else [])
        if coords.size == 0:
            return {k: np.array([]) for k in keys}
        mags = Y_mag[coords[:, 0], coords[:, 1]]
        order = np.argsort(mags)[::-1]
        coords = coords[order]
        result = {
            'Wavelength': self.wavlen[coords[:, 0]],
            'Frequency': self.freqs[coords[:, 1]],
            'Magnitude': mags[order]
        }
        if return_indices:
            result['ScaleIdx'] = coords[:, 0]
            result['FreqIdx'] = coords[:, 1]
        return result

    def find_modes(self, spectrum: np.ndarray, top_n: int = 5, min_dist: int = 5) -> ModeTable:
        r"""
        Identify oscillatory modes with damping estimation.

        Damping is estimated via the log-gradient method. For spectrum
        :math:`M = |M|e^{j\phi}`:

        .. math::

            \frac{\partial \log M}{\partial \omega} =
            \frac{\partial \ln|M|}{\partial \omega} +
            j\frac{\partial \phi}{\partial \omega}

        At resonance for a second-order system:

        .. math::

            \frac{\partial \log H}{\partial \omega}\bigg|_{\omega_n} =
            \frac{-j}{\zeta \omega_n}

        Thus the damping ratio is:

        .. math::

            \zeta = \frac{-1}{\omega_n \cdot \mathrm{Im}(\partial \log M / \partial \omega)}
                  = \frac{-1}{2\pi f_0 \cdot (\partial \phi / \partial f)}

        Parameters
        ----------
        spectrum : ndarray
            Complex spectrum from spectrum(..., return_complex=True).
        top_n : int, optional
            Maximum modes to identify.
        min_dist : int, optional
            Minimum index distance between peaks.

        Returns
        -------
        ModeTable
            Identified modes with frequency, damping, wavelength, magnitude.
        """
        if not np.iscomplexobj(spectrum):
            raise ValueError("find_modes requires complex spectrum (use return_complex=True)")

        Y_mag = np.abs(spectrum)
        peaks = self.find_peaks(Y_mag, top_n=top_n, min_dist=min_dist, return_indices=True)

        if peaks['Frequency'].size == 0:
            return ModeTable(np.array([]), np.array([]), np.array([]), np.array([]))

        log_spectrum = np.log(spectrum + 1e-20)
        grad_f = np.gradient(log_spectrum, self.freqs, axis=1)
        grad_s = np.gradient(log_spectrum, self.wavlen, axis=0)

        si, fi = peaks['ScaleIdx'], peaks['FreqIdx']
        f0, s0 = peaks['Frequency'], peaks['Wavelength']
        omega_n = 2 * np.pi * f0

        d_phi_df = np.imag(grad_f[si, fi])
        d_phi_ds = np.imag(grad_s[si, fi])

        with np.errstate(divide='ignore', invalid='ignore'):
            # Primary: temporal phase gradient
            zeta_f = -1.0 / (omega_n * d_phi_df)

            # Secondary: spatial phase gradient (normalized)
            zeta_s = -s0 / (omega_n * d_phi_ds * s0**2 + 1e-10)

            # Confidence-weighted combination
            w_f = np.abs(d_phi_df) + 1e-10
            w_s = np.abs(d_phi_ds * s0) + 1e-10
            damping = (w_f * zeta_f + w_s * zeta_s) / (w_f + w_s)

            # Fallback: curvature-based for singular phase gradient
            singular = np.abs(d_phi_df) < 1e-8
            if np.any(singular):
                d2_df2 = np.gradient(np.gradient(np.log(Y_mag + 1e-20), self.freqs, axis=1), self.freqs, axis=1)
                curv = np.minimum(d2_df2[si, fi], -1e-10)
                zeta_curv = np.sqrt(-2.0 / curv) / (2.0 * f0)
                damping = np.where(singular, zeta_curv, damping)

            damping = np.clip(np.where(np.isfinite(damping), damping, 0.0), 0.0, 1.0)

        return ModeTable(f0, damping, s0, peaks['Magnitude'])

    def analyze_many(self, V: np.ndarray, t: np.ndarray, time: float,
                     buses: Optional[List[int]] = None, top_n: int = 5,
                     min_dist: int = 5, verbose: bool = True) -> NetworkAnalysisResult:
        """
        Extract peaks across multiple buses.

        Pre-computes V @ B once for efficiency.

        Parameters
        ----------
        V : ndarray
            Signal matrix of shape (n_nodes, n_time).
        t : ndarray
            Time vector.
        time : float
            Time instant for temporal wavelet center.
        buses : list of int, optional
            Bus indices to analyze. Default is all.
        top_n : int, optional
            Max peaks per bus.
        min_dist : int, optional
            Min index distance between peaks.
        verbose : bool, optional
            Print progress updates.

        Returns
        -------
        NetworkAnalysisResult
            Named tuple with peaks dict and clusters dict.
        """
        if buses is None:
            buses = list(range(V.shape[0]))
        VB = V @ self._build_temporal_matrix(t, time_target=time)

        all_w, all_f, all_m, all_b = [], [], [], []
        for i, bus_idx in enumerate(buses):
            p = self.find_peaks(self.spectrum(V, t, bus=bus_idx, time=time, VB=VB),
                               top_n=top_n, min_dist=min_dist)
            if p['Wavelength'].size > 0:
                all_w.append(p['Wavelength'])
                all_f.append(p['Frequency'])
                all_m.append(p['Magnitude'])
                all_b.append(np.full(p['Wavelength'].shape, bus_idx, dtype=int))
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(buses)} buses...")

        empty_peaks = {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Magnitude', 'Bus_ID']}
        empty_clusters = {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}
        if not all_w:
            return NetworkAnalysisResult(peaks=empty_peaks, clusters=empty_clusters)

        master_peaks = {
            'Wavelength': np.concatenate(all_w), 'Frequency': np.concatenate(all_f),
            'Magnitude': np.concatenate(all_m), 'Bus_ID': np.concatenate(all_b)
        }
        return NetworkAnalysisResult(peaks=master_peaks,
                                     clusters=self._compute_density_clusters(master_peaks, top_n, min_dist))

    def _compute_density_clusters(self, peaks_dict: Dict[str, np.ndarray],
                                  top_n: int, min_dist: int) -> Dict[str, np.ndarray]:
        """Compute density-based clusters from peak data using KDE."""
        if peaks_dict['Wavelength'].size < 2:
            return {k: np.array([]) for k in ['Wavelength', 'Frequency', 'Density']}
        try:
            x, y = np.log10(peaks_dict['Wavelength']), peaks_dict['Frequency']
            kernel = gaussian_kde((x, y))

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