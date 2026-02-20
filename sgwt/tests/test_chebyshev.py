# -*- coding: utf-8 -*-
"""
Tests for Chebyshev polynomial approximation and convolution (chebyconv module).
"""
import numpy as np
import pytest

import sgwt


class TestChebyKernel:
    """Tests for ChebyKernel construction and fitting via ChebyModel."""

    def test_kernel_approximates_linear(self):
        """ChebyModel.kernel approximates f(x)=x correctly."""
        bound = 4.0
        f = lambda x: x
        kern = sgwt.ChebyModel.kernel(f, order=5, spectrum_bound=bound)
        x_eval = np.linspace(0, bound, 20)
        np.testing.assert_allclose(kern.evaluate(x_eval).flatten(), f(x_eval), atol=1e-2)

    def test_kernel_on_graph_estimates_bound(self, small_laplacian):
        """kernel_on_graph estimates spectral bound and creates valid kernel."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=10)
        assert isinstance(kern, sgwt.ChebyKernel)
        # Spectral bound should be positive and reasonable
        assert kern.spectrum_bound > 0.01, \
            f"Expected reasonable spectral bound, got {kern.spectrum_bound}"
        assert kern.C.shape[0] == 11  # order + 1

    @pytest.mark.parametrize("order", [5, 10, 20])
    def test_kernel_respects_order(self, small_laplacian, order):
        """Kernel C matrix has at most (order+1) rows (may be truncated for efficiency)."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=order)
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

    @pytest.mark.parametrize("order", [10, 30, 50])
    def test_high_order_is_stable(self, small_laplacian, identity_signal, order):
        """High-order polynomial remains numerically stable."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=order)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

    def test_convolve_with_random_signal(self, small_laplacian, random_signal):
        """Convolution works with multi-column random signal."""
        f = lambda x: 1.0 / (x + 1.0)  # lowpass-like
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=15)
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(random_signal, kern)
            assert result.shape[0] == random_signal.shape[0]
            assert result.shape[1] == random_signal.shape[1]

    def test_convolve_with_1d_input(self, small_laplacian, identity_signal):
        """Convolution works with 1D input signal and returns squeezed output."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=10)
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
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=5)
        
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
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=5)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            # Should not raise error
            result = conv.convolve(c_contig_signal, kern)
            assert result.shape[0] == random_signal.shape[0]

    def test_convolve_multi_basic(self, small_laplacian, identity_signal):
        """convolve_multi applies multiple kernels efficiently."""
        f1 = lambda x: np.exp(-x)
        f2 = lambda x: 1.0 / (x + 1.0)
        kern1 = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f1, order=10)
        kern2 = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f2, order=10)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            results = conv.convolve_multi(identity_signal, [kern1, kern2])

            assert len(results) == 2
            # Compare with individual convolution results
            single1 = conv.convolve(identity_signal, kern1)
            single2 = conv.convolve(identity_signal, kern2)
            np.testing.assert_allclose(results[0], single1, atol=1e-10)
            np.testing.assert_allclose(results[1], single2, atol=1e-10)

    def test_convolve_multi_empty(self, small_laplacian, identity_signal):
        """convolve_multi with empty kernel list returns empty list."""
        with sgwt.ChebyConvolve(small_laplacian) as conv:
            results = conv.convolve_multi(identity_signal, [])
            assert results == []

    def test_convolve_multi_complex(self, small_laplacian, identity_signal):
        """convolve_multi handles complex inputs."""
        complex_signal = identity_signal + 1j * identity_signal
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=10)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            results = conv.convolve_multi(complex_signal, [kern])
            assert len(results) == 1
            assert np.iscomplexobj(results[0])

    def test_convolve_multi_1d_input(self, small_laplacian):
        """convolve_multi handles 1D input signal."""
        signal_1d = np.ones(small_laplacian.shape[0])
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=10)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            results = conv.convolve_multi(signal_1d, [kern])
            assert len(results) == 1
            # Result should be 2D (n_vertices, n_dims) for 1D input
            assert results[0].ndim == 2

    def test_zero_order_kernel(self, small_laplacian, identity_signal):
        """Zero-order/zero-dim kernel returns zeros."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        # Empty coefficients
        C = np.zeros((0, 1))
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            assert result.shape[0] == identity_signal.shape[0]

    def test_many_dimensions_einsum_path(self, small_laplacian, identity_signal):
        """Test _accumulate with n_dim > 4 triggers einsum path."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        # Create kernel with 6 dimensions (> 4 threshold)
        C = np.random.rand(5, 6)  # 5 orders, 6 dimensions
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(identity_signal, kern)
            assert result.shape[2] == 6  # 6 dimensions

    def test_cache_hit_same_spectrum_bound(self, small_laplacian, identity_signal):
        """Test that recurrence matrix is cached when spectrum_bound is same."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=10)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            # First convolution creates cache
            result1 = conv.convolve(identity_signal, kern)
            # Second convolution should hit cache
            result2 = conv.convolve(identity_signal, kern)
            np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_cache_miss_different_spectrum_bound(self, small_laplacian, identity_signal):
        """Test that recurrence matrix is recalculated with different spectrum_bound."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        C = np.array([[1.0], [0.5]])  # Simple 2-coefficient kernel

        kern1 = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)
        kern2 = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd * 2)  # Different bound

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result1 = conv.convolve(identity_signal, kern1)
            result2 = conv.convolve(identity_signal, kern2)
            # Results should be different due to different scaling
            assert not np.allclose(result1, result2)

    def test_complex_f_contiguous_input(self, small_laplacian):
        """Test complex input that is already F-contiguous skips conversion."""
        # Create F-contiguous complex signal
        signal = np.asfortranarray(np.ones((small_laplacian.shape[0], 2), dtype=np.complex128))
        signal = signal + 1j * signal
        assert signal.flags['F_CONTIGUOUS']

        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyModel.kernel_on_graph(small_laplacian, f, order=5)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            result = conv.convolve(signal, kern)
            assert np.iscomplexobj(result)

    def test_convolve_multi_single_order_kernel(self, small_laplacian, identity_signal):
        """Test convolve_multi with single-coefficient kernel (max_order=1)."""
        ubnd = sgwt.estimate_spectral_bound(small_laplacian)
        # Single coefficient kernel - only T_0 term
        C = np.array([[2.0]])
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)

        with sgwt.ChebyConvolve(small_laplacian) as conv:
            results = conv.convolve_multi(identity_signal, [kern])
            assert len(results) == 1
            # Should just scale input by 2.0
            np.testing.assert_allclose(results[0].squeeze(), 2.0 * identity_signal, atol=1e-10)