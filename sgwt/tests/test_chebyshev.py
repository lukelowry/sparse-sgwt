# -*- coding: utf-8 -*-
"""
Tests for Chebyshev polynomial approximation and convolution (chebyconv module).
"""
import numpy as np
import pytest
from scipy.sparse import diags

import sgwt


class TestChebyKernel:
    """Tests for ChebyKernel construction and evaluation."""

    def test_from_function_approximates_linear(self):
        """ChebyKernel.from_function approximates f(x)=x correctly."""
        bound = 4.0
        f = lambda x: x
        kern = sgwt.ChebyKernel.from_function(f, order=5, spectrum_bound=bound)
        x_eval = np.linspace(0, bound, 20)
        np.testing.assert_allclose(kern.evaluate(x_eval).flatten(), f(x_eval), atol=1e-2)

    def test_from_function_on_graph_estimates_bound(self, small_laplacian):
        """from_function_on_graph estimates spectral bound and creates valid kernel."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=10)
        assert isinstance(kern, sgwt.ChebyKernel)
        assert kern.spectrum_bound > 0
        assert kern.C.shape[0] == 11  # order + 1

    @pytest.mark.parametrize("order", [5, 10, 20])
    def test_from_function_respects_order(self, small_laplacian, order):
        """Kernel C matrix has at most (order+1) rows (may be truncated for efficiency)."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=order)
        # Coefficients may be truncated if high-order terms are negligible
        assert kern.C.shape[0] <= order + 1
        assert kern.C.shape[0] >= 1


class TestChebyConvolve:
    """Tests for ChebyConvolve context manager."""

    def test_single_coefficient_kernel(self, small_laplacian, identity_signal):
        """Single-coefficient kernel (order=0) returns scaled input."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        C = np.array([[3.0]])  # Single coefficient: f(x) = 3
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            np.testing.assert_allclose(result.squeeze(), 3.0 * identity_signal, atol=1e-10)

    def test_identity_kernel_returns_input(self, small_laplacian, identity_signal):
        """Convolution with identity kernel f(x)=1 returns input."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        C = np.zeros((2, 1))
        C[0, 0] = 1.0  # T0 = 1, T1 = 0
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            np.testing.assert_allclose(result.squeeze(), identity_signal, atol=1e-10)

    def test_linear_kernel_applies_laplacian(self, small_laplacian, identity_signal):
        """Convolution with f(x)=x applies the Laplacian."""
        f = lambda x: x
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=10)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            expected = small_laplacian @ identity_signal
            np.testing.assert_allclose(result.squeeze(), expected, atol=1e-2)

    @pytest.mark.parametrize("order", [10, 30, 50])
    def test_high_order_is_stable(self, small_laplacian, identity_signal, order):
        """High-order polynomial remains numerically stable."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=order)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

    def test_convolve_with_random_signal(self, small_laplacian, random_signal):
        """Convolution works with multi-column random signal."""
        f = lambda x: 1.0 / (x + 1.0)  # lowpass-like
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=15)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(random_signal, kern)
            assert result.shape[0] == random_signal.shape[0]
            assert result.shape[1] == random_signal.shape[1]

    def test_convolve_with_1d_input(self, small_laplacian, identity_signal):
        """Convolution works with 1D input signal and returns squeezed output."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=10)
        # Ensure signal is 1D
        signal_1d = identity_signal.flatten()
        assert signal_1d.ndim == 1
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(signal_1d, kern)
            # Result should be 2D (n_vertices, n_dims) not 3D
            assert result.ndim == 2
            assert result.shape[0] == signal_1d.shape[0]

    def test_complex_input_handling(self, small_laplacian, identity_signal):
        """Complex signals are processed by splitting real/imag parts."""
        # Create a complex signal: x + i*x
        complex_signal = identity_signal + 1j * identity_signal
        
        # Simple kernel f(x) = x
        f = lambda x: x
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=5)
        
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(complex_signal, kern)
            
            # Expected: L(x) + i*L(x)
            expected_real = small_laplacian @ identity_signal
            expected = expected_real + 1j * expected_real
            
            np.testing.assert_allclose(result.squeeze(), expected, atol=1e-2)
            assert np.iscomplexobj(result)

    def test_c_contiguous_input(self, small_laplacian, random_signal):
        """C-contiguous inputs are converted to F-contiguous automatically."""
        c_contig_signal = np.ascontiguousarray(random_signal)
        assert not c_contig_signal.flags['F_CONTIGUOUS']
        
        # Simple kernel
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(small_laplacian, f, order=5)
        
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            # Should not raise error
            result = conv.convolve(c_contig_signal, kern)
            assert result.shape[0] == random_signal.shape[0]