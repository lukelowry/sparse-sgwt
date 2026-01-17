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
        # Cleanup
        engine.close()

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
        assert peaks['Wavelength'].size > 0
        assert 'Wavelength' in peaks
        assert 'Frequency' in peaks
        assert 'Magnitude' in peaks
        
        # Check peak location
        assert peaks['Magnitude'][0] == 10.0
        assert peaks['Wavelength'][0] == sgma_engine.wavlen[2]
        assert peaks['Frequency'][0] == sgma_engine.freqs[2]

    def test_find_system_wide_peaks(self, sgma_engine, random_signal):
        """Test system-wide peak finding returns two dicts."""
        n_time = random_signal.shape[1]
        t = np.linspace(0, 5, n_time)
        
        # Use a subset of buses for speed
        bus_indices = [0, 1]
        
        peaks, clusters = sgma_engine.find_system_wide_peaks(
            random_signal, t, bus_indices=bus_indices, verbose=False
        )
        
        assert isinstance(peaks, dict)
        assert isinstance(clusters, dict)
        
        if peaks['Wavelength'].size > 0:
            assert 'Bus_ID' in peaks

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
        
        # First call builds cache
        B1 = sgma_engine._build_temporal_matrix(t)
        assert sgma_engine._B is not None
        
        # Second call with same t should return same object
        B2 = sgma_engine._build_temporal_matrix(t)
        assert B1 is B2
        
        # Call with different t should rebuild
        t_new = np.linspace(0, 2, n_time)
        B3 = sgma_engine._build_temporal_matrix(t_new)
        assert B3 is not B1

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
            assert peaks['Wavelength'].size > 0
            assert clusters['Wavelength'].size == 0