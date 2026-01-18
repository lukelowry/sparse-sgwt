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
        s = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)
        time_target = 2.5

        # Initialize SGMA
        engine = SGMA(small_laplacian, s=s, freqs=freqs, time_target=time_target)
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
        assert len(sgma_engine.s) == 5
        assert len(sgma_engine.freqs) == 5
        assert len(sgma_engine.Ts) == 5
        assert len(sgma_engine.wavlen) == 5
        assert len(sgma_engine.poles) == 5
        assert sgma_engine._conv is None  # Lazy loading

    def test_transform_output_shape(self, sgma_engine, random_signal):
        """Test transform returns correct shape (n_scales, n_freqs)."""
        # random_signal is (n_nodes, 5) from conftest
        # We need a time vector matching the signal columns
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        
        # Transform at bus 0
        Y_mag = sgma_engine.transform(random_signal, t, bus_idx=0)
        
        expected_shape = (len(sgma_engine.s), len(sgma_engine.freqs))
        assert Y_mag.shape == expected_shape
        assert np.all(Y_mag >= 0)  # Magnitude should be non-negative

    def test_transform_with_precomputed_vb(self, sgma_engine, random_signal):
        """Test transform with pre-computed VB matches direct transform."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        
        # Direct
        Y1 = sgma_engine.transform(random_signal, t, bus_idx=0)
        
        # Pre-computed
        B = sgma_engine._build_temporal_matrix(t)
        VB = random_signal @ B
        Y2 = sgma_engine.transform(random_signal, t, bus_idx=0, VB=VB)
        
        np.testing.assert_allclose(Y1, Y2)

    def test_peaks_from_spectrum(self, sgma_engine):
        """Test peak extraction returns dict with correct keys."""
        # Create a synthetic spectrum with a clear peak
        Y_mag = np.zeros((5, 5))
        Y_mag[2, 2] = 10.0  # Peak at center

        peaks = sgma_engine.peaks_from_spectrum(Y_mag, top_n=1, min_dist=1)

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

    def test_find_system_wide_peaks(self, sgma_engine, random_signal):
        """Test system-wide peak finding returns two dicts with density clusters."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)

        # Use all buses to ensure enough peaks for density clustering
        bus_indices = list(range(random_signal.shape[0]))

        peaks, clusters = sgma_engine.find_system_wide_peaks(
            random_signal, t, bus_indices=bus_indices, verbose=False, min_dist=1
        )

        assert isinstance(peaks, dict)
        assert isinstance(clusters, dict)
        assert 'Bus_ID' in peaks

        # Verify density clustering produced results (covers success path)
        if peaks['Wavelength'].size >= 2:
            assert 'Density' in clusters

    def test_invalid_bus_index_raises(self, sgma_engine, random_signal):
        """Test out of bounds bus index raises ValueError."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        n_buses = random_signal.shape[0]
        
        with pytest.raises(ValueError):
            sgma_engine.transform(random_signal, t, bus_idx=n_buses + 1)

    def test_caching_temporal_matrix(self, sgma_engine, random_signal):
        """Test that temporal matrix B is cached and reused."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)

        # Verify initial state: no cache
        assert sgma_engine._B is None, "Cache should be empty initially"
        assert sgma_engine._t_cached is None, "Cached time vector should be None initially"

        # First call builds cache
        B1 = sgma_engine._build_temporal_matrix(t)
        assert sgma_engine._B is not None, "Cache should be populated after first call"
        assert sgma_engine._t_cached is not None, "Time vector should be cached"

        # Second call with same t should return same object (cache hit)
        B2 = sgma_engine._build_temporal_matrix(t)
        assert B1 is B2, "Should return cached matrix for same time vector"
        assert sgma_engine._B is B1, "Internal cache should still hold same matrix"

        # Call with different t should rebuild (cache invalidation)
        t_new = np.linspace(0, 2, n_time)
        B3 = sgma_engine._build_temporal_matrix(t_new)
        assert B3 is not B1, "Should create new matrix for different time vector"
        assert sgma_engine._B is B3, "Cache should be updated to new matrix"
        # Verify old cache was released (testing memory efficiency indirectly)
        assert not np.array_equal(sgma_engine._t_cached, t), \
            "Cached time vector should be updated"

    def test_peaks_extraction_no_peaks(self, sgma_engine):
        """Test peak extraction when spectrum is flat zero."""
        Y_flat = np.zeros((5, 5))
        peaks = sgma_engine.peaks_from_spectrum(Y_flat)
        assert peaks['Wavelength'].size == 0

    def test_system_wide_peaks_no_signal(self, sgma_engine, random_signal):
        """Test system wide peaks with zero signal returns empty lists."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)
        V_zero = np.zeros_like(random_signal)
        
        peaks, clusters = sgma_engine.find_system_wide_peaks(V_zero, t, verbose=False)
        assert peaks['Wavelength'].size == 0
        assert clusters['Wavelength'].size == 0

    def test_density_clustering_exception_handling(self, sgma_engine, random_signal):
        """Test that exceptions in density clustering are caught and logged."""
        from unittest.mock import patch
        n_time = random_signal.shape[1]
        t = np.linspace(0, 1, n_time)

        # Mock gaussian_kde to raise exception
        with patch('sgwt.sgma.gaussian_kde', side_effect=ValueError("KDE Failed")):
            # We need peaks to be found to reach the clustering step
            peaks, clusters = sgma_engine.find_system_wide_peaks(
                random_signal, t, bus_indices=[0], verbose=True, min_dist=1
            )
            # Peaks should still be found despite clustering failure
            assert peaks['Wavelength'].size > 0, \
                "Peaks should be found even when clustering fails"
            # Clusters should be empty due to exception
            assert clusters['Wavelength'].size == 0, \
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
        """Test the peak finding fallback when scikit-image is not available."""
        from unittest.mock import patch
        import sys
        import importlib
        import sgwt.sgma

        # Create a synthetic spectrum with multiple peaks
        Y_mag = np.zeros((10, 10))
        Y_mag[2, 2] = 10.0  # Main peak
        Y_mag[8, 8] = 8.0   # Secondary peak
        Y_mag[2, 3] = 9.0   # A nearby point to test min_dist suppression

        # Force the fallback by making skimage unimportable in sys.modules
        with patch.dict('sys.modules', {'skimage': None, 'skimage.feature': None}):
            # Reload the sgma module to execute the 'except' block
            importlib.reload(sgwt.sgma)

        # We need a new engine instance that uses the reloaded module
        # and has scales/freqs matching the synthetic Y_mag shape.
        s_test = np.geomspace(0.1, 100.0, 10)
        freqs_test = np.linspace(0.1, 2.0, 10)
        reloaded_engine = sgwt.sgma.SGMA(
            L=small_laplacian,
            s=s_test,
            freqs=freqs_test,
            time_target=sgma_engine.time_target
        )

        try:
            # With min_dist=2, the peak at (2,3) should be suppressed by (2,2)
            peaks = reloaded_engine.peaks_from_spectrum(Y_mag, top_n=2, min_dist=2)
            assert len(peaks['Magnitude']) == 2
            np.testing.assert_allclose(peaks['Magnitude'], [10.0, 8.0])
        finally:
            importlib.reload(sgwt.sgma)
            reloaded_engine.close()


class TestPeakLocalMaxFallback:
    """Tests for the _peak_local_max_fallback function."""

    @pytest.fixture
    def fallback_func(self):
        """Load the fallback function by forcing skimage import to fail."""
        from unittest.mock import patch
        import importlib
        import sgwt.sgma

        with patch.dict('sys.modules', {'skimage': None, 'skimage.feature': None}):
            importlib.reload(sgwt.sgma)
            func = sgwt.sgma.peak_local_max
        yield func
        importlib.reload(sgwt.sgma)

    def test_fallback_min_distance_clamp(self, fallback_func):
        """Test that min_distance < 1 is clamped to 1."""
        image = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        result = fallback_func(image, min_distance=0)
        assert result.shape[0] == 1
        np.testing.assert_array_equal(result[0], [1, 1])

    def test_fallback_empty_image(self, fallback_func):
        """Test fallback returns empty array for all-zero image."""
        image = np.zeros((5, 5))
        result = fallback_func(image)
        assert result.shape[0] == 0
        assert result.shape[1] == 2