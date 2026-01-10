# -*- coding: utf-8 -*-
"""
pytest configuration and shared fixtures for the sgwt test suite.

Fixtures are organized in layers:
- Low-level: synthetic graph data (small_laplacian, random_signal)
- Mid-level: DLL availability guards (HAS_CHOLMOD, requires_dll)
- High-level: context managers for convolution tests
"""
import numpy as np
import pytest
from scipy.sparse import diags, csc_matrix

# ---------------------------------------------------------------------------
# DLL availability detection
# ---------------------------------------------------------------------------
try:
    import sgwt
    _dll = sgwt.get_cholmod_dll()
    HAS_CHOLMOD = True
except Exception:
    HAS_CHOLMOD = False

try:
    import sgwt
    _klu = sgwt.get_klu_dll()
    HAS_KLU = True
except Exception:
    HAS_KLU = False

# ---------------------------------------------------------------------------
# Skip markers for DLL-dependent tests
# ---------------------------------------------------------------------------
requires_cholmod = pytest.mark.skipif(
    not HAS_CHOLMOD, reason="CHOLMOD DLL not available"
)
requires_klu = pytest.mark.skipif(
    not HAS_KLU, reason="KLU DLL not available"
)

# ---------------------------------------------------------------------------
# Low-level fixtures: synthetic graph data
# ---------------------------------------------------------------------------
@pytest.fixture
def small_laplacian():
    """Create a synthetic 10x10 path graph Laplacian (no DLL required)."""
    n = 10
    L = diags([2.0, -1.0, -1.0], [0, 1, -1], shape=(n, n), format='csc')
    # Fix boundary conditions for path graph
    L = L.tolil()
    L[0, 0] = 1.0
    L[n-1, n-1] = 1.0
    return L.tocsc()


@pytest.fixture
def medium_laplacian():
    """Create a synthetic 50x50 path graph Laplacian for larger tests."""
    n = 50
    L = diags([2.0, -1.0, -1.0], [0, 1, -1], shape=(n, n), format='csc')
    L = L.tolil()
    L[0, 0] = 1.0
    L[n-1, n-1] = 1.0
    return L.tocsc()


@pytest.fixture
def random_signal(small_laplacian):
    """Create a random signal matching small_laplacian dimensions."""
    n = small_laplacian.shape[0]
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, 5))


@pytest.fixture
def impulse_signal(small_laplacian):
    """Create an impulse signal at node 0 for small_laplacian."""
    n = small_laplacian.shape[0]
    X = np.zeros((n, 1))
    X[0, 0] = 1.0
    return X


@pytest.fixture
def identity_signal(small_laplacian):
    """Create identity matrix as signal (impulse on every node)."""
    n = small_laplacian.shape[0]
    return np.eye(n)


# ---------------------------------------------------------------------------
# Library data fixtures (requires data files, not DLLs)
# ---------------------------------------------------------------------------
@pytest.fixture
def texas_laplacian():
    """Load DELAY_TEXAS Laplacian from library."""
    import sgwt
    return sgwt.DELAY_TEXAS


@pytest.fixture
def texas_signal(texas_laplacian):
    """Create impulse signal for Texas grid."""
    import sgwt
    return sgwt.impulse(texas_laplacian, n=100)


@pytest.fixture
def library_kernel():
    """Load MODIFIED_MORLET kernel from library."""
    import sgwt
    return sgwt.MODIFIED_MORLET


# ---------------------------------------------------------------------------
# Kernel fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_vfkernel():
    """Create a simple VFKernel with one pole."""
    import sgwt
    return sgwt.VFKernel(
        Q=np.array([1.0]),
        R=np.array([[1.0]]),
        D=np.array([0.0])
    )


@pytest.fixture
def simple_chebykernel(small_laplacian):
    """Create a simple ChebyKernel approximating exp(-x)."""
    import sgwt
    f = lambda x: np.exp(-x)
    return sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=10)


# ---------------------------------------------------------------------------
# High-level context fixtures (requires DLLs)
# ---------------------------------------------------------------------------
@pytest.fixture
def convolve_context(texas_laplacian):
    """Yield a Convolve context manager for Texas Laplacian."""
    import sgwt
    with sgwt.Convolve(texas_laplacian) as conv:
        yield conv


@pytest.fixture
def dyconvolve_context(texas_laplacian):
    """Yield a DyConvolve context manager with standard poles."""
    import sgwt
    poles = [0.1, 1.0, 10.0]
    with sgwt.DyConvolve(texas_laplacian, poles) as conv:
        yield conv


# ---------------------------------------------------------------------------
# Parametrized test data
# ---------------------------------------------------------------------------
SCALES = [0.1, 1.0, 10.0]
ORDERS = [1, 2, 3]
