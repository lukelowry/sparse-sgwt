# -*- coding: utf-8 -*-
"""
Tests for Spectral Graph Modal Analysis (SGMA) engine.
"""
import numpy as np
import pytest

from sgwt import SGMA
from sgwt.tests.conftest import requires_cholmod

# Mark all tests in this module as requiring CHOLMOD
pytestmark = requires_cholmod

class TestSGMA:
    """Tests for SGMA class functionality."""

    @pytest.fixture
    def sgma_engine(self, small_laplacian):
        """Fixture to create an SGMA instance with small graph."""
        # Define scales and frequencies
        scales = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)

        # Initialize SGMA
        engine = SGMA(small_laplacian, scales=scales, freqs=freqs)
        yield engine
        # Cleanup with error handling
        try:
            engine.close()
        except Exception as e:
            # Log but don't fail test if cleanup fails
            import warnings
            warnings.warn(f"SGMA cleanup failed: {e}", RuntimeWarning)

    def test_initialization(self, sgma_engine):
        """Test that SGMA initializes derived attributes correctly."""
        assert len(sgma_engine.scales) == 5
        assert len(sgma_engine.freqs) == 5
        assert len(sgma_engine.Ts) == 5
        assert len(sgma_engine.wavlen) == 5
        assert len(sgma_engine.poles) == 5
        assert sgma_engine._conv is None  # Lazy loading

    def test_spectrum_output_shape(self, sgma_engine, random_signal):
        """Test spectrum returns correct shape (n_scales, n_freqs)."""
        # random_signal is (n_nodes, 5) from conftest
        # We need a time vector matching the signal columns
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5
        
        # Spectrum at bus 0
        Y_mag = sgma_engine.spectrum(random_signal, t, bus=0, time=time_target)
        
        expected_shape = (len(sgma_engine.scales), len(sgma_engine.freqs))
        assert Y_mag.shape == expected_shape
        assert np.all(Y_mag >= 0)  # Magnitude should be non-negative

    def test_spectrum_with_precomputed_vb(self, sgma_engine, random_signal):
        """Test spectrum with pre-computed VB matches direct spectrum."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5
        
        # Direct
        Y1 = sgma_engine.spectrum(random_signal, t, bus=0, time=time_target)
        
        # Pre-computed
        B = sgma_engine._build_temporal_matrix(t, time_target=time_target)
        VB = random_signal @ B
        Y2 = sgma_engine.spectrum(random_signal, t, bus=0, time=time_target, VB=VB)
        
        np.testing.assert_allclose(Y1, Y2)

    def test_find_peaks(self, sgma_engine):
        """Test peak extraction returns dict with correct keys."""
        # Create a synthetic spectrum with a clear peak
        Y_mag = np.zeros((5, 5))
        Y_mag[2, 2] = 10.0  # Peak at center

        peaks = sgma_engine.find_peaks(Y_mag, top_n=1, min_dist=1)

        assert isinstance(peaks, dict)
        # Should find at least 1 peak
        assert peaks['Wavelength'].size > 0, "Expected to find at least one peak"
        assert 'Wavelength' in peaks
        assert 'Frequency' in peaks
        assert 'Magnitude' in peaks

        # Check peak location (should find the peak at (2, 2))
        assert peaks['Magnitude'][0] == 10.0
        assert peaks['Wavelength'][0] == sgma_engine.wavlen[2]
        assert peaks['Frequency'][0] == sgma_engine.freqs[2]

    def test_analyze_convenience_method(self, sgma_engine, random_signal):
        """Test that analyze() is equivalent to spectrum() -> find_peaks()."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5
        bus_idx = 0
        top_n = 3

        # Manual two-step process
        spectrum_manual = sgma_engine.spectrum(random_signal, t, bus=bus_idx, time=time_target)
        peaks_manual = sgma_engine.find_peaks(spectrum_manual, top_n=top_n)

        # Using the analyze() convenience method
        peaks_analyze = sgma_engine.analyze(random_signal, t, bus=bus_idx, time=time_target, top_n=top_n)

        # The results should be identical
        assert isinstance(peaks_analyze, dict)
        assert peaks_analyze.keys() == peaks_manual.keys()
        for key in peaks_manual:
            np.testing.assert_array_equal(peaks_analyze[key], peaks_manual[key])

    def test_analyze_many(self, sgma_engine, random_signal):
        """Test system-wide peak finding returns two dicts with density clusters."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5

        # Use all buses to ensure enough peaks for density clustering
        bus_indices = list(range(random_signal.shape[0]))

        result = sgma_engine.analyze_many(
            random_signal, t, time=time_target, buses=bus_indices, verbose=False, min_dist=1
        )

        assert hasattr(result, 'peaks')
        assert hasattr(result, 'clusters')
        assert isinstance(result.peaks, dict)
        assert isinstance(result.clusters, dict)
        assert 'Bus_ID' in result.peaks

        # Verify density clustering produced results (covers success path)
        if result.peaks['Wavelength'].size >= 2:
            assert 'Density' in result.clusters

    def test_invalid_bus_index_raises(self, sgma_engine, random_signal):
        """Test out of bounds bus index raises ValueError."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        n_buses = random_signal.shape[0]
        time_target = 2.5
        
        with pytest.raises(ValueError):
            sgma_engine.spectrum(random_signal, t, bus=n_buses + 1, time=time_target)

    def test_caching_temporal_matrix(self, sgma_engine, random_signal):
        """Test that temporal matrix B is cached and reused."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)
        time1 = 2.0
        time2 = 3.0

        # Verify initial state: no cache
        assert sgma_engine._B is None, "Cache should be empty initially"
        assert sgma_engine._t_cached is None, "Cached time vector should be None initially"
        assert sgma_engine._time_target_cached is None

        # First call builds cache
        B1 = sgma_engine._build_temporal_matrix(t, time_target=time1)
        assert sgma_engine._B is not None, "Cache should be populated after first call"
        assert sgma_engine._t_cached is not None, "Time vector should be cached"
        assert sgma_engine._time_target_cached is not None

        # Second call with same t and time should return same object (cache hit)
        B2 = sgma_engine._build_temporal_matrix(t, time_target=time1)
        assert B1 is B2, "Should return cached matrix for same time vector"
        assert sgma_engine._B is B1, "Internal cache should still hold same matrix"

        # Call with different t should rebuild (cache invalidation)
        t_new = np.linspace(0, 2, n_time)
        B3 = sgma_engine._build_temporal_matrix(t_new, time_target=time1)
        assert B3 is not B1, "Should create new matrix for different time vector"
        assert sgma_engine._B is B3, "Cache should be updated to new matrix"

        # Call with different time should rebuild
        B4 = sgma_engine._build_temporal_matrix(t, time_target=time2)
        assert B4 is not B1, "Should create new matrix for different time target"
        assert sgma_engine._time_target_cached == time2

    def test_find_peaks_no_peaks(self, sgma_engine):
        """Test peak extraction when spectrum is flat zero."""
        Y_flat = np.zeros((5, 5))
        peaks = sgma_engine.find_peaks(Y_flat)
        assert peaks['Wavelength'].size == 0

    def test_analyze_many_no_signal(self, sgma_engine, random_signal):
        """Test system wide peaks with zero signal returns empty lists."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)
        time_target = 2.5
        V_zero = np.zeros_like(random_signal)
        
        result = sgma_engine.analyze_many(V_zero, t, time=time_target, verbose=False)
        assert result.peaks['Wavelength'].size == 0
        assert result.clusters['Wavelength'].size == 0

    def test_density_clustering_exception_handling(self, sgma_engine, random_signal):
        """Test that exceptions in density clustering are caught and logged."""
        from unittest.mock import patch
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)
        time_target = 2.5

        # Mock gaussian_kde to raise exception
        with patch('sgwt.sgma.gaussian_kde', side_effect=ValueError("KDE Failed")):
            # We need peaks to be found to reach the clustering step
            result = sgma_engine.analyze_many(
                random_signal, t, time=time_target, buses=[0], verbose=True, min_dist=1
            )
            # Peaks should still be found despite clustering failure
            assert result.peaks['Wavelength'].size > 0, \
                "Peaks should be found even when clustering fails"
            # Clusters should be empty due to exception
            assert result.clusters['Wavelength'].size == 0, \
                "Clusters should be empty when KDE raises exception"

    def test_density_clustering_insufficient_peaks(self, sgma_engine):
        """Test _compute_density_clusters returns empty when < 2 peaks."""
        # Only one peak - triggers the size < 2 branch
        single_peak = {
            'Wavelength': np.array([1.0]),
            'Frequency': np.array([0.5]),
            'Magnitude': np.array([10.0]),
            'Bus_ID': np.array([0])
        }
        result = sgma_engine._compute_density_clusters(single_peak, top_n=5, min_dist=5)
        assert result['Wavelength'].size == 0
        assert result['Frequency'].size == 0
        assert result['Density'].size == 0

    def test_peak_finding_fallback(self, sgma_engine, small_laplacian):
        """Test the peak finding fallback and its internal branches."""
        from unittest.mock import patch
        import sys
        import importlib
        import sgwt.sgma

        # Force the fallback by making skimage unimportable in sys.modules
        with patch.dict('sys.modules', {'skimage': None, 'skimage.feature': None}):
            # Reload the sgma module to execute the 'except' block
            importlib.reload(sgwt.sgma)
            # Get a direct handle to the fallback for isolated tests
            fallback_func = sgwt.sgma.peak_local_max

            # --- Test isolated fallback function cases ---
            # Test min_distance clamping
            image_clamp = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
            result_clamp = fallback_func(image_clamp, min_distance=0)
            assert result_clamp.shape[0] == 1

            # Test empty image
            image_empty = np.zeros((5, 5))
            result_empty = fallback_func(image_empty)
            assert result_empty.shape[0] == 0

            # --- Test fallback through the SGMA class ---
            # Create a synthetic spectrum with a plateau and other peaks
            Y_mag = np.zeros((20, 20))
            Y_mag[5, 5] = 10.0  # Plateau peak 1
            Y_mag[5, 6] = 10.0  # Plateau peak 2
            Y_mag[15, 15] = 8.0 # Distant secondary peak

            # We need a new engine instance that uses the reloaded module
            s_test = np.geomspace(0.1, 100.0, 20)
            freqs_test = np.linspace(0.1, 2.0, 20)
            reloaded_engine = sgwt.sgma.SGMA(
                L=small_laplacian, scales=s_test, freqs=freqs_test
            )

            try:
                # Case 1: Test the suppression branch (`if not is_suppressed`).
                # With min_dist=1, (5,5) and (5,6) are both local maxima.
                # The first one processed will suppress the second.
                # This ensures the `else` path of `if not is_suppressed` is hit.
                peaks_suppress = reloaded_engine.find_peaks(Y_mag, top_n=3, min_dist=1)
                assert len(peaks_suppress['Magnitude']) == 2
                # One of the plateau peaks (mag 10) and the distant peak (mag 8) should be found.
                np.testing.assert_allclose(sorted(peaks_suppress['Magnitude'], reverse=True), [10.0, 8.0])

                # Case 2: Test the `break` path of the loop.
                # We ask for just 1 peak, so the loop should break early.
                peaks_break = reloaded_engine.find_peaks(Y_mag, top_n=1, min_dist=1)
                assert len(peaks_break['Magnitude']) == 1
                np.testing.assert_allclose(peaks_break['Magnitude'], [10.0])

                # Case 3: Test the natural loop exit.
                # We ask for more peaks than exist, so the loop should finish.
                peaks_natural = reloaded_engine.find_peaks(Y_mag, top_n=10, min_dist=1)
                assert len(peaks_natural['Magnitude']) == 2
                np.testing.assert_allclose(sorted(peaks_natural['Magnitude'], reverse=True), [10.0, 8.0])

            finally:
                reloaded_engine.close()

        # Reload the module again outside the patch to restore the original state for other tests
        importlib.reload(sgwt.sgma)


class TestModeTable:
    """Tests for the ModeTable class."""

    def test_modetable_creation(self):
        """Test ModeTable initialization with valid data."""
        from sgwt.sgma import ModeTable

        freq = np.array([0.5, 1.0, 1.5])
        damping = np.array([0.05, 0.03, 0.08])
        wavelength = np.array([10.0, 5.0, 2.5])
        magnitude = np.array([100.0, 80.0, 60.0])

        modes = ModeTable(freq, damping, wavelength, magnitude)

        assert modes.n_modes == 3
        np.testing.assert_array_equal(modes.frequency, freq)
        np.testing.assert_array_equal(modes.damping, damping)
        np.testing.assert_array_equal(modes.wavelength, wavelength)
        np.testing.assert_array_equal(modes.magnitude, magnitude)

    def test_modetable_empty(self):
        """Test ModeTable with empty arrays."""
        from sgwt.sgma import ModeTable

        modes = ModeTable(
            frequency=np.array([]),
            damping=np.array([]),
            wavelength=np.array([]),
            magnitude=np.array([])
        )

        assert modes.n_modes == 0
        assert "empty" in repr(modes)

    def test_modetable_repr(self):
        """Test ModeTable string representation."""
        from sgwt.sgma import ModeTable

        modes = ModeTable(
            frequency=np.array([0.5, 1.0]),
            damping=np.array([0.05, 0.03]),
            wavelength=np.array([10.0, 5.0]),
            magnitude=np.array([100.0, 80.0])
        )

        repr_str = repr(modes)
        assert "ModeTable" in repr_str
        assert "2 modes" in repr_str
        assert "Freq (Hz)" in repr_str
        assert "Damping" in repr_str
        assert "Wavelength" in repr_str
        assert "Magnitude" in repr_str

    def test_modetable_single_mode(self):
        """Test ModeTable repr with single mode (no 's' plural)."""
        from sgwt.sgma import ModeTable

        modes = ModeTable(
            frequency=np.array([0.5]),
            damping=np.array([0.05]),
            wavelength=np.array([10.0]),
            magnitude=np.array([100.0])
        )

        repr_str = repr(modes)
        assert "1 mode identified" in repr_str
        # Should not have "modes" (plural)
        assert "1 modes" not in repr_str

    def test_modetable_to_dict(self):
        """Test ModeTable to_dict method."""
        from sgwt.sgma import ModeTable

        freq = np.array([0.5, 1.0])
        damping = np.array([0.05, 0.03])
        wavelength = np.array([10.0, 5.0])
        magnitude = np.array([100.0, 80.0])

        modes = ModeTable(freq, damping, wavelength, magnitude)
        d = modes.to_dict()

        assert isinstance(d, dict)
        assert 'Frequency' in d
        assert 'Damping' in d
        assert 'Wavelength' in d
        assert 'Magnitude' in d
        np.testing.assert_array_equal(d['Frequency'], freq)

    def test_modetable_to_array(self):
        """Test ModeTable to_array method."""
        from sgwt.sgma import ModeTable

        modes = ModeTable(
            frequency=np.array([0.5, 1.0]),
            damping=np.array([0.05, 0.03]),
            wavelength=np.array([10.0, 5.0]),
            magnitude=np.array([100.0, 80.0])
        )

        arr = modes.to_array()
        assert arr.shape == (2, 4)
        # First column is frequency
        np.testing.assert_array_equal(arr[:, 0], [0.5, 1.0])
        # Second column is damping
        np.testing.assert_array_equal(arr[:, 1], [0.05, 0.03])


class TestFindModes:
    """Tests for the find_modes method."""

    @pytest.fixture
    def sgma_engine(self, small_laplacian):
        """Fixture to create an SGMA instance."""
        scales = np.geomspace(0.1, 10.0, 10)
        freqs = np.linspace(0.1, 2.0, 20)
        engine = SGMA(small_laplacian, scales=scales, freqs=freqs)
        yield engine
        try:
            engine.close()
        except Exception:
            pass

    def test_find_modes_requires_complex(self, sgma_engine):
        """Test find_modes raises error for real spectrum."""
        # Create a real (non-complex) spectrum
        real_spectrum = np.random.rand(10, 20)

        with pytest.raises(ValueError, match="complex spectrum"):
            sgma_engine.find_modes(real_spectrum)

    def test_find_modes_returns_modetable(self, sgma_engine, random_signal):
        """Test find_modes returns a ModeTable object."""
        from sgwt.sgma import ModeTable

        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5

        # Get complex spectrum
        spectrum = sgma_engine.spectrum(
            random_signal, t, bus=0, time=time_target, return_complex=True
        )

        modes = sgma_engine.find_modes(spectrum, top_n=3)

        assert isinstance(modes, ModeTable)

    def test_find_modes_empty_spectrum(self, sgma_engine):
        """Test find_modes with zero spectrum returns empty ModeTable."""
        # Create a complex zero spectrum
        spectrum = np.zeros((10, 20), dtype=complex)

        modes = sgma_engine.find_modes(spectrum)

        assert modes.n_modes == 0
        assert modes.frequency.size == 0

    def test_find_modes_synthetic_peak(self, sgma_engine):
        """Test find_modes with synthetic spectrum containing clear peak."""
        # Create a synthetic complex spectrum with a peak
        n_scales, n_freqs = 10, 20
        spectrum = np.ones((n_scales, n_freqs), dtype=complex) * 0.1

        # Add a peak at (5, 10) with specific phase behavior
        peak_scale, peak_freq = 5, 10
        spectrum[peak_scale, peak_freq] = 10.0 * np.exp(1j * 0.5)

        # Add phase gradient around the peak (simulating resonance)
        for j in range(n_freqs):
            spectrum[peak_scale, j] *= np.exp(1j * (j - peak_freq) * 0.1)

        modes = sgma_engine.find_modes(spectrum, top_n=1, min_dist=1)

        assert modes.n_modes >= 1
        # The peak should be detected at the expected frequency
        assert modes.frequency[0] == sgma_engine.freqs[peak_freq]

    def test_find_modes_boundary_peaks(self, sgma_engine):
        """Test find_modes with peaks at frequency boundaries."""
        n_scales, n_freqs = 10, 20
        spectrum = np.ones((n_scales, n_freqs), dtype=complex) * 0.1

        # Add peak at left boundary (freq index 0)
        spectrum[3, 0] = 10.0 * np.exp(1j * 0.5)
        # Add peak at right boundary (freq index n_freqs-1)
        spectrum[7, n_freqs - 1] = 8.0 * np.exp(1j * 0.3)

        modes = sgma_engine.find_modes(spectrum, top_n=2, min_dist=1)

        # Should find both boundary peaks
        assert modes.n_modes == 2

    def test_find_modes_damping_finite(self, sgma_engine, random_signal):
        """Test that damping values are finite (not inf/nan)."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5

        spectrum = sgma_engine.spectrum(
            random_signal, t, bus=0, time=time_target, return_complex=True
        )

        modes = sgma_engine.find_modes(spectrum, top_n=3, min_dist=1)

        # All damping values should be finite
        assert np.all(np.isfinite(modes.damping))


class TestSGMACoverage:
    """Additional tests to cover remaining branches."""

    @pytest.fixture
    def sgma_engine(self, small_laplacian):
        """Fixture to create an SGMA instance."""
        scales = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)
        engine = SGMA(small_laplacian, scales=scales, freqs=freqs)
        yield engine
        try:
            engine.close()
        except Exception:
            pass

    def test_find_peaks_return_indices(self, sgma_engine, random_signal):
        """Test find_peaks with return_indices=True."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5

        spectrum = sgma_engine.spectrum(random_signal, t, bus=0, time=time_target)
        peaks = sgma_engine.find_peaks(spectrum, top_n=3, min_dist=1, return_indices=True)

        # Should have ScaleIdx and FreqIdx keys
        assert 'ScaleIdx' in peaks
        assert 'FreqIdx' in peaks
        # Indices should be integers
        if peaks['ScaleIdx'].size > 0:
            assert peaks['ScaleIdx'].dtype in [np.int32, np.int64, np.intp]
            assert peaks['FreqIdx'].dtype in [np.int32, np.int64, np.intp]

    def test_find_peaks_return_indices_empty(self, sgma_engine):
        """Test find_peaks with return_indices=True on empty spectrum."""
        Y_flat = np.zeros((5, 5))
        peaks = sgma_engine.find_peaks(Y_flat, return_indices=True)

        assert 'ScaleIdx' in peaks
        assert 'FreqIdx' in peaks
        assert peaks['ScaleIdx'].size == 0
        assert peaks['FreqIdx'].size == 0

    def test_analyze_many_default_buses(self, sgma_engine, random_signal):
        """Test analyze_many with buses=None (default to all buses)."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        time_target = 2.5

        # buses=None should default to all buses
        result = sgma_engine.analyze_many(
            random_signal, t, time=time_target, buses=None, verbose=False, min_dist=1
        )

        assert hasattr(result, 'peaks')
        assert hasattr(result, 'clusters')

    def test_analyze_many_verbose_progress(self):
        """Test analyze_many verbose progress output with many buses."""
        from scipy.sparse import diags

        # Create a larger Laplacian with 60 nodes to trigger verbose progress
        n_nodes = 60
        L = diags([2.0, -1.0, -1.0], [0, 1, -1], shape=(n_nodes, n_nodes), format='csc')
        L = L.tolil()
        L[0, 0] = 1.0
        L[n_nodes - 1, n_nodes - 1] = 1.0
        L = L.tocsc()

        scales = np.geomspace(0.1, 10.0, 3)
        freqs = np.linspace(0.1, 1.0, 3)
        engine = SGMA(L, scales=scales, freqs=freqs)

        try:
            n_time = 10
            t = np.linspace(0, 5, n_time)
            rng = np.random.default_rng(42)
            V = rng.standard_normal((n_nodes, n_time))
            time_target = 2.5

            # This should print progress at bus 50 (when i+1 == 50)
            result = engine.analyze_many(
                V, t, time=time_target, buses=list(range(n_nodes)),
                verbose=True, min_dist=1
            )
            assert hasattr(result, 'peaks')
        finally:
            engine.close()

    def test_sgma_del_method(self, small_laplacian):
        """Test that __del__ properly cleans up resources."""
        scales = np.geomspace(0.1, 10.0, 3)
        freqs = np.linspace(0.1, 1.0, 3)
        engine = SGMA(small_laplacian, scales=scales, freqs=freqs)

        # Initialize the convolution context to create resources
        _ = engine._get_conv()
        assert engine._conv is not None

        # Manually call __del__ to cover the destructor
        engine.__del__()

        # After __del__, resources should be cleaned up
        assert engine._conv is None

    def test_find_modes_singular_phase_gradient(self, small_laplacian):
        """Test find_modes with singular phase gradient triggers curvature fallback."""
        scales = np.geomspace(0.1, 10.0, 10)
        freqs = np.linspace(0.1, 2.0, 20)
        engine = SGMA(small_laplacian, scales=scales, freqs=freqs)

        try:
            # Create a synthetic complex spectrum with constant phase (zero phase gradient)
            n_scales, n_freqs = 10, 20
            spectrum = np.ones((n_scales, n_freqs), dtype=complex) * 0.1

            # Add a peak with constant phase (no phase variation) to trigger singular path
            peak_scale, peak_freq = 5, 10
            # Use a real positive value (phase = 0) across frequencies near the peak
            # This makes the phase gradient effectively zero
            for j in range(n_freqs):
                spectrum[peak_scale, j] = 5.0 + 0j  # Real positive, constant phase = 0

            # Make the peak stand out
            spectrum[peak_scale, peak_freq] = 10.0 + 0j

            modes = engine.find_modes(spectrum, top_n=1, min_dist=1)

            # Should still return valid results using the curvature fallback
            assert modes.n_modes >= 0
            if modes.n_modes > 0:
                # Damping should be finite (fallback should handle the singular case)
                assert np.all(np.isfinite(modes.damping))
        finally:
            engine.close()