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

    def test_kernel_returns_chebykernel(self):
        """kernel() returns a ChebyKernel, not a ChebyModel."""
        f = lambda x: 1.0 / (x + 1)
        K = ChebyModel.kernel(f, order=20, spectrum_bound=100.0)
        assert isinstance(K, sgwt.ChebyKernel)
        assert K.spectrum_bound == 100.0
