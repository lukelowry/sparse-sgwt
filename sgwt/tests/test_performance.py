# -*- coding: utf-8 -*-
"""
Performance and stress tests for sgwt package.

This module contains performance tests and stress tests for core SGWT operations:
- Graph convolution (static and dynamic)
- SGMA transformations
- Topology updates
- Kernel construction

Tests are organized into:
1. Performance tests - Regular tests that verify performance-critical operations
   - Work in VSCode test sidebar and pytest command line
   - Test correctness and scalability across different input sizes
2. Stress tests - Larger tests marked with @pytest.mark.slow
   - Use realistic library laplacians (DELAY_TEXAS, DELAY_USA, DELAY_WECC)
   - Automatically skipped when running full suite (unless --run-slow flag)
   - Run individually from VSCode test sidebar without any special flags

Usage:
    # Run all performance tests (skip slow tests)
    pytest test_performance.py

    # Run with ALL tests including slow stress tests
    pytest test_performance.py --run-slow

    # Run individual slow test from VSCode - works automatically!
    # Just click the test in the sidebar

    # Run just performance tests (skip slow)
    pytest sgwt/tests/test_performance.py -m "not slow"

Note: pytest-benchmark is optional. These tests work as regular tests without it.
"""
import numpy as np
import pytest

import sgwt
from sgwt.tests.conftest import requires_cholmod

# Mark all tests in this module as requiring CHOLMOD
pytestmark = requires_cholmod


# ---------------------------------------------------------------------------
# Helper functions for synthetic data generation
# ---------------------------------------------------------------------------
def create_path_graph_laplacian(n):
    """
    Create an n-node path graph Laplacian.

    Path graph: 0 - 1 - 2 - ... - (n-1)

    Parameters
    ----------
    n : int
        Number of nodes

    Returns
    -------
    L : scipy.sparse.csc_matrix
        Laplacian matrix of shape (n, n)
    """
    from scipy.sparse import diags

    L = diags([2.0, -1.0, -1.0], [0, 1, -1], shape=(n, n), format='csc')
    # Fix boundary conditions
    L = L.tolil()
    L[0, 0] = 1.0
    L[n-1, n-1] = 1.0
    return L.tocsc()


def create_random_signal(n_nodes, n_timesteps, seed=42):
    """
    Create a random signal for testing.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (rows)
    n_timesteps : int
        Number of time samples (columns)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    signal : np.ndarray
        Random signal of shape (n_nodes, n_timesteps)
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_nodes, n_timesteps))


# ---------------------------------------------------------------------------
# Module-scoped fixtures for large graphs (expensive to create)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def medium_path_laplacian():
    """Medium-sized path graph (100 nodes) for benchmarks."""
    return create_path_graph_laplacian(100)


@pytest.fixture(scope='module')
def large_path_laplacian():
    """Large path graph (1000 nodes) for benchmarks."""
    return create_path_graph_laplacian(1000)


@pytest.fixture(scope='module')
def simple_vfkernel():
    """Simple VFKernel with 3 poles for benchmarks."""
    return sgwt.VFKernel(
        Q=np.array([1.0, 0.5, 0.25]),
        R=np.array([[1.0], [0.5], [0.25]]),
        D=np.array([0.0])
    )


@pytest.fixture(scope='module')
def texas_laplacian():
    """Load DELAY_TEXAS Laplacian from library."""
    return sgwt.DELAY_TEXAS


@pytest.fixture(scope='module')
def usa_laplacian():
    """Load DELAY_USA Laplacian from library."""
    return sgwt.DELAY_USA


@pytest.fixture(scope='module')
def wecc_laplacian():
    """Load DELAY_WECC Laplacian from library."""
    return sgwt.DELAY_WECC


# ---------------------------------------------------------------------------
# Convolution Performance Benchmarks
# ---------------------------------------------------------------------------
class TestConvolutionPerformance:
    """Benchmark tests for static and dynamic graph convolution."""

    @pytest.mark.parametrize("n_nodes", [10, 50, 100])
    def test_lowpass_convolution_scaling(self, n_nodes):
        """Test lowpass convolution scaling with graph size."""
        L = create_path_graph_laplacian(n_nodes)
        X = create_random_signal(n_nodes, 1)
        scales = [1.0]

        with sgwt.Convolve(L) as conv:
            result = conv.lowpass(X, scales)

        assert len(result) == len(scales)
        assert result[0].shape == (n_nodes, 1)

    @pytest.mark.parametrize("n_nodes", [10, 50, 100])
    def test_bandpass_convolution_scaling(self, n_nodes):
        """Test bandpass convolution scaling with graph size."""
        L = create_path_graph_laplacian(n_nodes)
        X = create_random_signal(n_nodes, 1)
        scales = [1.0]

        with sgwt.Convolve(L) as conv:
            result = conv.bandpass(X, scales)

        assert len(result) == len(scales)
        assert result[0].shape == (n_nodes, 1)

    def test_multi_scale_convolution(self, medium_path_laplacian):
        """Test convolution with multiple scales."""
        X = create_random_signal(100, 1)
        scales = [0.1, 0.5, 1.0, 5.0, 10.0]

        with sgwt.Convolve(medium_path_laplacian) as conv:
            result = conv.lowpass(X, scales)

        assert len(result) == len(scales)
        assert all(r.shape == X.shape for r in result)

    def test_multi_signal_convolution(self, medium_path_laplacian):
        """Test convolution with multiple signals."""
        n_signals = 10
        X = create_random_signal(100, n_signals)
        scales = [1.0]

        with sgwt.Convolve(medium_path_laplacian) as conv:
            result = conv.lowpass(X, scales)

        assert len(result) == len(scales)
        assert result[0].shape == X.shape

    def test_vfkernel_convolution(self, medium_path_laplacian, simple_vfkernel):
        """Test VFKernel convolution."""
        X = create_random_signal(100, 1)

        with sgwt.Convolve(medium_path_laplacian) as conv:
            result = conv.convolve(X, simple_vfkernel)

        assert result.shape[:2] == X.shape  # Result may have additional dimension for poles

    @pytest.mark.parametrize("n_updates", [1, 10, 50])
    def test_topology_update_scaling(self, n_updates):
        """Test topology updates with varying numbers of updates."""
        L = create_path_graph_laplacian(100)
        X = create_random_signal(100, 1)
        poles = [0.1, 1.0, 10.0]

        with sgwt.DyConvolve(L, poles) as conv:
            Y = conv.lowpass(X)
            # Perform n_updates rank-1 updates (adding new branches)
            for i in range(n_updates):
                # Add branch between non-adjacent nodes
                # Using larger gaps to ensure branches don't already exist
                node_a = (i * 2) % 98
                node_b = (i * 2 + 5) % 98 + 1  # Ensure different node
                # Small edge weight
                ok = conv.addbranch(node_a, node_b, 0.1)
            result = conv.lowpass(X)

        assert len(result) == len(poles)
        assert result[0].shape == X.shape


# ---------------------------------------------------------------------------
# SGMA Performance Benchmarks
# ---------------------------------------------------------------------------
class TestSGMAPerformance:
    """Benchmark tests for Spectral Graph Modal Analysis."""

    @pytest.mark.parametrize("n_timesteps", [10, 100, 500])
    def test_sgma_transform_time_scaling(self, small_laplacian, n_timesteps):
        """Test SGMA transform with varying time series lengths."""
        n_nodes = small_laplacian.shape[0]
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 10, n_timesteps)

        s = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)

        engine = sgwt.SGMA(small_laplacian, s=s, freqs=freqs, time_target=2.5)
        try:
            result = engine.transform(V, t, bus_idx=0)
        finally:
            engine.close()

        assert result.shape == (len(s), len(freqs))

    @pytest.mark.parametrize("n_scales", [3, 5, 10])
    def test_sgma_scale_scaling(self, small_laplacian, n_scales):
        """Test SGMA transform with varying numbers of scales."""
        n_nodes = small_laplacian.shape[0]
        n_timesteps = 50
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 10, n_timesteps)

        s = np.geomspace(0.1, 10.0, n_scales)
        freqs = np.linspace(0.1, 1.0, 5)

        engine = sgwt.SGMA(small_laplacian, s=s, freqs=freqs, time_target=2.5)
        try:
            result = engine.transform(V, t, bus_idx=0)
        finally:
            engine.close()

        assert result.shape == (n_scales, len(freqs))

    def test_sgma_peak_finding(self, small_laplacian):
        """Test SGMA peak detection from spectrum."""
        n_nodes = small_laplacian.shape[0]
        n_timesteps = 100
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 10, n_timesteps)

        s = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)

        engine = sgwt.SGMA(small_laplacian, s=s, freqs=freqs, time_target=2.5)

        try:
            Y_mag = engine.transform(V, t, bus_idx=0)
            peaks = engine.peaks_from_spectrum(Y_mag, top_n=3, min_dist=1)
            assert 'Wavelength' in peaks
            assert 'Frequency' in peaks
        finally:
            engine.close()

    def test_sgma_system_wide_peaks(self, small_laplacian):
        """Test system-wide peak finding across buses."""
        n_nodes = small_laplacian.shape[0]
        n_timesteps = 100
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 10, n_timesteps)

        s = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)

        engine = sgwt.SGMA(small_laplacian, s=s, freqs=freqs, time_target=2.5)

        try:
            bus_indices = [0, 1, 2]  # Small subset for testing
            peaks, clusters = engine.find_system_wide_peaks(
                V, t, bus_indices=bus_indices, verbose=False
            )
            assert isinstance(peaks, dict)
            assert isinstance(clusters, dict)
        finally:
            engine.close()


# ---------------------------------------------------------------------------
# Kernel Construction Benchmarks
# ---------------------------------------------------------------------------
class TestKernelConstructionPerformance:
    """Benchmark tests for kernel approximation and construction."""

    @pytest.mark.parametrize("order", [5, 10, 20])
    def test_chebyshev_approximation_order(self, order, small_laplacian):
        """Test Chebyshev approximation with varying polynomial orders."""
        f = lambda x: np.exp(-x)

        kernel = sgwt.ChebyKernel.from_function_on_graph(
            small_laplacian, f, order=order
        )

        # Verify kernel has coefficients (C matrix should have at most order+1 rows)
        assert kernel.C.shape[0] <= order + 1
        assert kernel.C.shape[0] >= 1

    def test_chebyshev_from_function(self):
        """Test Chebyshev kernel from function."""
        f = lambda x: np.exp(-x)

        kernel = sgwt.ChebyKernel.from_function(f, order=10, spectrum_bound=2.0)

        # Verify kernel has coefficients
        assert kernel.C.shape[0] >= 1
        assert kernel.spectrum_bound == 2.0

    def test_vfkernel_initialization(self):
        """Test VFKernel construction."""
        n_poles = 10
        Q = np.random.randn(n_poles)
        R = np.random.randn(n_poles, 1)
        D = np.array([0.0])

        kernel = sgwt.VFKernel(Q=Q, R=R, D=D)

        # Verify kernel has poles (Q array)
        assert len(kernel.Q) == n_poles
        assert kernel.R.shape[0] == n_poles

    def test_chebyshev_convolution(self, medium_path_laplacian):
        """Test Chebyshev-based convolution."""
        X = create_random_signal(100, 1)
        f = lambda x: np.exp(-x)
        kernel = sgwt.ChebyKernel.from_function_on_graph(
            medium_path_laplacian, f, order=10
        )

        with sgwt.ChebyConvolve(medium_path_laplacian) as conv:
            result = conv.convolve(X, kernel)

        # Result may have extra dimension for multi-column kernels
        assert result.shape[:2] == X.shape


# ---------------------------------------------------------------------------
# Stress Tests (Large-scale operations)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestStressTests:
    """
    Stress tests for large graphs and time series.

    These tests are marked as 'slow' and can be run with:
        pytest test_performance.py --run-slow

    They use library laplacians (DELAY_TEXAS, DELAY_USA, DELAY_WECC) to
    test realistic large-scale scenarios.
    """

    def test_large_graph_convolution_texas(self, texas_laplacian):
        """Stress test: Convolution on DELAY_TEXAS graph."""
        n_nodes = texas_laplacian.shape[0]
        X = create_random_signal(n_nodes, 1)
        scales = [0.1, 1.0, 10.0]

        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.lowpass(X, scales)

        assert len(result) == len(scales)
        assert all(r.shape == X.shape for r in result)
        print(f"TEXAS graph: {n_nodes} nodes, convolution successful")

    def test_large_graph_convolution_usa(self, usa_laplacian):
        """Stress test: Convolution on DELAY_USA graph (largest library graph)."""
        n_nodes = usa_laplacian.shape[0]
        X = create_random_signal(n_nodes, 1)
        scales = [0.1, 1.0, 10.0]

        with sgwt.Convolve(usa_laplacian) as conv:
            result = conv.lowpass(X, scales)

        assert len(result) == len(scales)
        assert all(r.shape == X.shape for r in result)
        print(f"USA graph: {n_nodes} nodes, convolution successful")

    def test_long_time_series_sgma(self, texas_laplacian):
        """Stress test: SGMA with long time series (1000+ samples)."""
        n_nodes = texas_laplacian.shape[0]
        n_timesteps = 1000
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 100, n_timesteps)

        s = np.geomspace(0.1, 10.0, 5)
        freqs = np.linspace(0.1, 1.0, 5)

        engine = sgwt.SGMA(texas_laplacian, s=s, freqs=freqs, time_target=10.0)
        try:
            result = engine.transform(V, t, bus_idx=0)
            assert result.shape == (len(s), len(freqs))
            print(f"SGMA on {n_timesteps} timesteps successful")
        finally:
            engine.close()

    def test_many_topology_updates(self, large_path_laplacian):
        """Stress test: 500 topology updates on 1000-node graph."""
        n_nodes = large_path_laplacian.shape[0]
        X = create_random_signal(n_nodes, 1)
        poles = [0.1, 1.0, 10.0]
        n_updates = 500

        with sgwt.DyConvolve(large_path_laplacian, poles) as conv:
            Y_initial = conv.lowpass(X)

            # Perform many updates (adding new branches)
            n_added = 0
            for i in range(n_updates):
                # Add branches between non-adjacent nodes
                node_a = (i * 3) % (n_nodes - 10)
                node_b = (i * 3 + 7) % (n_nodes - 10) + 5
                ok = conv.addbranch(node_a, node_b, 0.01)
                if ok:
                    n_added += 1

            Y_final = conv.lowpass(X)

        assert len(Y_final) == len(poles)
        assert Y_final[0].shape == X.shape
        # Verify updates had an effect
        max_diff = np.max(np.abs(Y_final[0] - Y_initial[0]))
        assert max_diff > 1e-6, f"Updates should affect result (max_diff={max_diff:.2e})"
        print(f"{n_added}/{n_updates} topology updates on {n_nodes}-node graph successful")

    def test_full_sgma_system_wecc(self, wecc_laplacian):
        """Stress test: Full SGMA system-wide peak analysis on WECC graph."""
        n_nodes = wecc_laplacian.shape[0]
        n_timesteps = 200
        V = create_random_signal(n_nodes, n_timesteps)
        t = np.linspace(0, 20, n_timesteps)

        s = np.geomspace(0.1, 50.0, 8)
        freqs = np.linspace(0.1, 2.0, 10)

        engine = sgwt.SGMA(wecc_laplacian, s=s, freqs=freqs, time_target=5.0)
        try:
            # Use subset of buses for system-wide analysis
            bus_indices = list(range(min(20, n_nodes)))

            peaks, clusters = engine.find_system_wide_peaks(
                V, t, bus_indices=bus_indices, verbose=True
            )

            assert isinstance(peaks, dict)
            assert isinstance(clusters, dict)
            print(f"WECC graph: {n_nodes} nodes, {len(bus_indices)} buses analyzed")
            print(f"Found {len(peaks['Wavelength'])} peaks, {len(clusters['Wavelength'])} clusters")
        finally:
            engine.close()
