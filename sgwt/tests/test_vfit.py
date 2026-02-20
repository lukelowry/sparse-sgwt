# -*- coding: utf-8 -*-
"""
Tests for the Vector Fitting module (fitting.vfit).
"""
import numpy as np
import pytest

import sgwt
from sgwt.fitting import VFModel


class TestVFModel:
    """Tests for VFModel — Vector Fitting rational approximation."""

    def test_fit_recovers_known_rational(self):
        """Recovers poles, residues, d, and e from a known rational function."""
        s = 1j * np.linspace(1, 100, 300)
        # f(s) = 5 + 0.1*s + 1/(s+3) + 2/(s+20)
        f = 5.0 + 0.1 * s + 1.0 / (s + 3) + 2.0 / (s + 20)
        model = VFModel.fit(s, f, n_poles=2, fit_d=True, fit_e=True)

        recovered = sorted(model.poles.real)
        assert abs(recovered[0] + 20) < 2.0
        assert abs(recovered[1] + 3) < 2.0
        assert abs(model.d[0].real - 5.0) < 1.0
        assert abs(model.e[0].real - 0.1) < 0.05
        assert model.rmse < 1e-3
        assert '2 poles' in repr(model)

    def test_fit_multi_channel(self):
        """Multi-channel response produces correct residue shape."""
        s = 1j * np.linspace(1, 50, 200)
        f = np.column_stack([1.0 / (s + 3), 2.0 / (s + 10)])
        model = VFModel.fit(s, f, n_poles=2)
        assert model.residues.shape[1] == 2
        assert model.d.shape == (2,)
        assert model.rmse < 1e-2

    def test_fit_without_direct_term(self):
        """fit_d=False omits constant term from the model."""
        s = 1j * np.linspace(1, 50, 200)
        f = 1.0 / (s + 5)
        model = VFModel.fit(s, f, n_poles=1, fit_d=False)
        assert np.allclose(model.d, 0)

    def test_fit_transposed_input(self):
        """Handles f with shape (A, M) instead of (M, A)."""
        s = 1j * np.linspace(1, 50, 200)
        f = (1.0 / (s + 5)).reshape(1, -1)  # (1, 200)
        model = VFModel.fit(s, f, n_poles=1)
        assert model.rmse < 1e-3

    def test_kernel_returns_constrained_vfkernel(self):
        """kernel() returns VFKernel with real negative poles."""
        lam = np.linspace(0.01, 100, 500)
        g = 1.0 / (lam + 1)
        K = VFModel.kernel(lam, g, n_poles=4)
        assert isinstance(K, sgwt.VFKernel)
        assert np.allclose(K.Q.imag, 0)
        assert np.all(K.Q.real < 0)
        assert K.R.shape[0] == 4
