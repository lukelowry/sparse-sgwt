# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/test_chebyshev.py
Description: Tests for Chebyshev polynomial approximation and convolution.
"""
import unittest
import numpy as np
from scipy.sparse import diags
import sgwt

class TestChebyshev(unittest.TestCase):
    def setUp(self):
        # Create a simple 10x10 Laplacian (Path graph)
        n = 10
        # 2 on diag, -1 on off-diag
        self.L = diags([2, -1, -1], [0, 1, -1], shape=(n, n)).tocsc()
        self.X = np.eye(n) # Impulse on every node

    def test_kernel_fitting(self):
        """Test that ChebyKernel correctly approximates a function."""
        # f(x) = x
        # Domain [0, 4] (approx max eigenvalue of path graph is < 4)
        bound = 4.0
        f = lambda x: x
        
        # Fit order 5
        kern = sgwt.ChebyKernel.from_function(f, order=5, spectrum_bound=bound)
        
        # Evaluate at some points
        x_eval = np.linspace(0, bound, 20)
        y_true = f(x_eval)
        y_approx = kern.evaluate(x_eval)
        
        np.testing.assert_allclose(y_approx.flatten(), y_true, atol=1e-2)

    def test_from_function_on_graph(self):
        """Test the convenience method for fitting from a graph."""
        f = lambda x: np.exp(-x)
        # This should run without error and produce a valid kernel
        kern = sgwt.ChebyKernel.from_function_on_graph(self.L, f, order=10)
        self.assertIsInstance(kern, sgwt.ChebyKernel)
        self.assertGreater(kern.spectrum_bound, 0)
        self.assertEqual(kern.C.shape[0], 11)

    def test_convolve_identity(self):
        """Test convolution with identity kernel f(x)=1."""
        ubnd = sgwt.estimate_spectral_bound(self.L)
        # f(x) = 1
        # We can manually create a kernel: T0=1, others=0
        # C matrix: order+1 x 1
        C = np.zeros((2, 1))
        C[0, 0] = 1.0
        kern = sgwt.ChebyKernel(C=C, spectrum_bound=ubnd)
        
        with sgwt.ChebyConvolve(self.L) as conv:
            res = conv.convolve(self.X, kern)
            # Result should be X * 1 = X
            np.testing.assert_allclose(res.squeeze(), self.X, atol=1e-10)

    def test_convolve_laplacian(self):
        """Test convolution with f(x)=x, which should apply L."""
        # Fit f(x) = x
        f = lambda x: x
        kern = sgwt.ChebyKernel.from_function_on_graph(self.L, f, order=10)
        
        with sgwt.ChebyConvolve(self.L) as conv:
            res = conv.convolve(self.X, kern)
            # Result should be L @ X
            expected = self.L @ self.X
            np.testing.assert_allclose(res.squeeze(), expected, atol=1e-2)

    def test_high_order(self):
        """Test high order polynomial to ensure recurrence stability."""
        f = lambda x: np.exp(-x)
        kern = sgwt.ChebyKernel.from_function_on_graph(self.L, f, order=50)
        with sgwt.ChebyConvolve(self.L) as conv:
            res = conv.convolve(self.X, kern)
            self.assertFalse(np.any(np.isnan(res)))
            self.assertFalse(np.any(np.isinf(res)))