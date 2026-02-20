# -*- coding: utf-8 -*-
"""
Tests for the Chebyshev fitting module (fitting.chebyshev).
"""
import numpy as np
import pytest

import sgwt
from sgwt.fitting import ChebyModel


class TestChebyModel:
    """Tests for ChebyModel — Chebyshev polynomial approximation."""

    def test_fit_approximates_function(self):
        """fit() produces a model that approximates the target function."""
        f = lambda x: 1.0 / (x + 1)
        model = ChebyModel.fit(f, order=40, spectrum_bound=100.0)
        x = np.linspace(0.01, 100, 200)
        assert np.max(np.abs(np.ravel(model(x)) - f(x))) < 1e-3

    def test_kernel_returns_chebykernel(self, small_laplacian):
        """kernel() returns a ChebyKernel, not a ChebyModel."""
        f = lambda x: 1.0 / (x + 1)
        K = ChebyModel.kernel(small_laplacian, f, order=20)
        assert isinstance(K, sgwt.ChebyKernel)
        assert np.isclose(K.spectrum_bound, sgwt.estimate_spectral_bound(small_laplacian))

    def test_zero_function_keeps_constant_term(self, small_laplacian):
        """Fitting a zero function keeps at least the constant term."""
        kern = ChebyModel.kernel(small_laplacian, lambda x: np.zeros_like(x), order=5)
        assert kern.C.shape[0] >= 1

    def test_multioutput_function_preserves_2d_coeffs(self, small_laplacian):
        """Fitting a multi-output function preserves 2D coefficient structure."""
        def multi_func(x):
            return np.column_stack([np.exp(-x), np.sin(x)])

        kern = ChebyModel.kernel(small_laplacian, multi_func, order=5)
        assert kern.C.shape[1] == 2
        x_test = np.linspace(0, kern.spectrum_bound, 10)
        result = kern.evaluate(x_test)
        assert result.shape == (10, 2)

    @pytest.mark.parametrize("order", [0, -5])
    def test_invalid_order_raises_valueerror(self, small_laplacian, order):
        """Order < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Order must be >= 1"):
            ChebyModel.kernel(small_laplacian, lambda x: x, order=order)

    def test_all_negligible_coefficients(self, small_laplacian):
        """Fitting where all higher-order coefficients are negligible."""
        kern = ChebyModel.kernel(
            small_laplacian, lambda x: np.full_like(x, 1e-20), order=10
        )
        assert kern.C.shape[0] >= 1

    @pytest.mark.parametrize("sampling", ['linear', 'quadratic', 'logarithmic'])
    def test_sampling_strategies(self, small_laplacian, sampling):
        """kernel() works with various sampling strategies."""
        kern = ChebyModel.kernel(
            small_laplacian, lambda x: np.exp(-x), order=5, sampling=sampling
        )
        assert kern.C.shape[0] > 0
        x_test = np.linspace(0, min(2.0, kern.spectrum_bound), 20)
        result = kern.evaluate(x_test)
        expected = np.exp(-x_test)
        np.testing.assert_allclose(result.flatten(), expected, atol=0.1)

    def test_adaptive_fitting(self, small_laplacian):
        """kernel() with adaptive order selection."""
        kern = ChebyModel.kernel(
            small_laplacian, lambda x: np.exp(-x), order=5,
            adaptive=True, target_error=0.01, max_order=50
        )
        assert kern.C.shape[0] > 0
        x_test = np.linspace(0, min(2.0, kern.spectrum_bound), 100)
        result = kern.evaluate(x_test)
        expected = np.exp(-x_test)
        rel_error = np.max(np.abs(result.flatten() - expected) / np.maximum(np.abs(expected), 1e-15))
        assert rel_error < 0.1

    def test_adaptive_reaches_max_order(self, small_laplacian):
        """Adaptive fitting that hits max_order still produces valid kernel."""
        kern = ChebyModel.kernel(
            small_laplacian, lambda x: np.exp(-x), order=5,
            adaptive=True, target_error=1e-20, max_order=10
        )
        assert kern.C.shape[0] > 0
