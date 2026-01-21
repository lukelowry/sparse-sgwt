# -*- coding: utf-8 -*-
"""
Performance and stress tests for the sgwt.cholconv module.

This module contains performance tests for Cholesky-based graph convolution
solvers (`sgwt.Convolve` and `sgwt.DyConvolve`).

Tests are organized into two main classes:
1. `TestScalingPerformance`: Benchmarks how solver performance scales with the
   number of graph nodes (N) across various synthetic and real-world graphs.
2. `TestParameterScaling`: Benchmarks how performance is affected by key
   parameters on a fixed-size graph:
   - Number of signals (M)
   - Number of filter scales (J)
   - Bandpass filter order (K)

Stress tests on very large graphs are marked with `@pytest.mark.slow`.

Usage:
    # Run all performance benchmarks
    pytest sgwt/tests/test_performance.py -m benchmark

    # To skip the slow stress tests:
    pytest sgwt/tests/test_performance.py -m "benchmark and not slow"
"""
import numpy as np
import pytest

import sgwt
from sgwt.tests.conftest import requires_cholmod

# Mark all tests in this module as requiring CHOLMOD and as benchmarks
pytestmark = [requires_cholmod, pytest.mark.benchmark]

# Random seed for reproducible test data
DEFAULT_SEED = 42


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


def create_random_signal(n_nodes, n_timesteps, seed=DEFAULT_SEED):
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
def xlarge_path_laplacian():
    """Extra-large path graph (10000 nodes) for benchmarks."""
    return create_path_graph_laplacian(10000)


@pytest.fixture(scope='module')
def xxlarge_path_laplacian():
    """Extra-extra-large path graph (50000 nodes) for benchmarks."""
    return create_path_graph_laplacian(50000)


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


@pytest.fixture(scope='module')
def hawaii_laplacian():
    """Load DELAY_HAWAII Laplacian from library."""
    return sgwt.DELAY_HAWAII


@pytest.fixture(scope='module')
def eastwest_laplacian():
    """Load DELAY_EASTWEST Laplacian from library."""
    return sgwt.DELAY_EASTWEST


# ---------------------------------------------------------------------------
# Scaling & Performance Tests (Parametrized)
# ---------------------------------------------------------------------------

# Configuration for graphs to test: (fixture_name, id, is_slow)
GRAPH_DEFINITIONS = [
    ("medium_path_laplacian", "100", False),
    ("large_path_laplacian", "1k", False),
    ("texas_laplacian", "Texas(2k)", False),
    ("hawaii_laplacian", "Hawaii(37)", False),
    ("wecc_laplacian", "WECC(240)", True),
    ("xlarge_path_laplacian", "10k", True),
    ("xxlarge_path_laplacian", "50k", True),
    ("eastwest_laplacian", "EastWest(65k)", True),
    ("usa_laplacian", "USA(82k)", True),
]

GRAPH_CASES = [
    pytest.param(g[0], marks=pytest.mark.slow, id=g[1]) if g[2] else pytest.param(g[0], id=g[1])
    for g in GRAPH_DEFINITIONS
]

SIGNAL_SAMPLES = [2**i for i in range(11)]  # 1 to 1024
SCALE_SAMPLES = [2**i for i in range(8)]    # 1 to 128
ORDER_SAMPLES = [2**i for i in range(7)]    # 1 to 64

class TestScalingPerformance:
    """
    Comprehensive scaling tests for all filters and graph sizes.

    Iterates through:
    - Graphs: Synthetic (100, 1k) -> Real (37, 240, 2k, 65k, 82k)
    - Filters: Lowpass, Bandpass, Highpass
    - Contexts: Static (Convolve), Dynamic (DyConvolve)
    """

    @pytest.mark.parametrize("graph_name", GRAPH_CASES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_static_convolution(self, graph_name, method_name, request, benchmark):
        """Benchmark static convolution (Convolve) scaling."""
        L = request.getfixturevalue(graph_name)
        n_nodes = L.shape[0]
        benchmark.extra_info['num_edges'] = (L.nnz - n_nodes) // 2
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        scales = [1.0]

        with sgwt.Convolve(L) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X, scales)

    @pytest.mark.parametrize("graph_name", GRAPH_CASES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_dynamic_convolution(self, graph_name, method_name, request, benchmark):
        """Benchmark dynamic convolution (DyConvolve) scaling."""
        L = request.getfixturevalue(graph_name)
        n_nodes = L.shape[0]
        benchmark.extra_info['num_edges'] = (L.nnz - n_nodes) // 2
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        poles = [1.0]

        with sgwt.DyConvolve(L, poles) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X)


class TestParameterScaling:
    """
    Tests scaling with respect to signal dimensions and filter parameters
    on a fixed graph size (Texas ~2k nodes), covering all filter types.
    """

    @pytest.mark.parametrize("n_signals", SIGNAL_SAMPLES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_signal_scaling(self, texas_laplacian, n_signals, method_name, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, n_signals)
        scales = [1.0]
        with sgwt.Convolve(L) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X, scales)

    @pytest.mark.parametrize("n_signals", SIGNAL_SAMPLES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_signal_scaling_dynamic(self, texas_laplacian, n_signals, method_name, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, n_signals)
        poles = [1.0]
        with sgwt.DyConvolve(L, poles) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X)

    @pytest.mark.parametrize("n_scales", SCALE_SAMPLES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_scale_scaling(self, texas_laplacian, n_scales, method_name, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        scales = list(np.geomspace(0.1, 10.0, n_scales))
        with sgwt.Convolve(L) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X, scales)

    @pytest.mark.parametrize("n_scales", SCALE_SAMPLES)
    @pytest.mark.parametrize("method_name", ["lowpass", "bandpass", "highpass"])
    def test_scale_scaling_dynamic(self, texas_laplacian, n_scales, method_name, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        poles = list(np.geomspace(0.1, 10.0, n_scales))
        with sgwt.DyConvolve(L, poles) as conv:
            func = getattr(conv, method_name)
            benchmark(func, X)

    @pytest.mark.parametrize("order", ORDER_SAMPLES)
    def test_order_scaling_bandpass(self, texas_laplacian, order, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        scales = [1.0]
        with sgwt.Convolve(L) as conv:
            benchmark(conv.bandpass, X, scales, order=order)

    @pytest.mark.parametrize("order", ORDER_SAMPLES)
    def test_order_scaling_bandpass_dynamic(self, texas_laplacian, order, benchmark):
        L = texas_laplacian
        n_nodes = L.shape[0]
        X = create_random_signal(n_nodes, 1)
        poles = [1.0]
        with sgwt.DyConvolve(L, poles) as conv:
            benchmark(conv.bandpass, X, order=order)
