# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/test_functionality.py
Description: Core functionality tests validating filters and dynamic updates 
             without external dependencies like sksparse.
"""
import unittest
import numpy as np

class TestSGWTFunctionality(unittest.TestCase):
    
    def setUp(self):
        import sgwt
        self.sgwt = sgwt
        self.L = sgwt.DELAY_TEXAS
        self.K = sgwt.MODIFIED_MORLET
        self.VFKern = sgwt.VFKern
        
        self.X = self.sgwt.impulse(self.L, n=100)
        self.scales = [0.1, 1.0, 10.0]

    def test_cholmod_import(self):
        """Verify CHOLMOD DLL can be loaded."""
        self.sgwt.get_cholmod_dll()

    def test_convolve_static_filters(self):
        """Test basic analytical filters in static Convolve context."""
        with self.sgwt.Convolve(self.L) as conv:
            # Low-pass
            lp = conv.lowpass(self.X, self.scales)
            self.assertEqual(len(lp), len(self.scales))
            self.assertEqual(lp[0].shape, self.X.shape)
            
            # Band-pass
            bp = conv.bandpass(self.X, self.scales)
            self.assertEqual(len(bp), len(self.scales))
            
            # High-pass
            hp = conv.highpass(self.X, self.scales)
            self.assertEqual(len(hp), len(self.scales))

    def test_dynamic_filters(self):
        """Test all analytical filters in dynamic DyConvolve context."""
        poles = [1.0 / s for s in self.scales]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            lp = conv.lowpass(self.X)
            self.assertEqual(len(lp), len(poles))
            
            bp = conv.bandpass(self.X)
            self.assertEqual(len(bp), len(poles))
            
            hp = conv.highpass(self.X)
            self.assertEqual(len(hp), len(poles))

    def test_lowpass_bset_execution(self):
        """Verify low-pass filter with sparse subset (Bset) runs correctly."""
        from scipy.sparse import csc_matrix
        bset = csc_matrix((np.ones(1), ([100], [0])), shape=(self.L.shape[0], 1))
        X_single = self.X[:, :1].copy(order='F')
        
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.lowpass(X_single, self.scales, Bset=bset)
            self.assertEqual(len(res), len(self.scales))

    def test_vf_convolution(self):
        """Test Vector Fitting kernel convolution with both dict and VFKern."""
        with self.sgwt.Convolve(self.L) as conv:
            # Test with raw dict from library
            res = conv.convolve(self.X, self.K)
            self.assertEqual(res.shape[0], self.L.shape[0])
            
            # Test with VFKern object
            vk = self.VFKern.from_dict(self.K)
            res_vk = conv.convolve(self.X, vk)
            np.testing.assert_allclose(res, res_vk)

    def test_vf_validation(self):
        """Verify that convolve raises appropriate errors for invalid kernel inputs."""
        with self.sgwt.Convolve(self.L) as conv:
            with self.assertRaises(TypeError):
                conv.convolve(self.X, "not a kernel")
            
            with self.assertRaises(ValueError):
                conv.convolve(self.X, self.VFKern(Q=None, R=None, D=None))

    def test_vf_direct_term(self):
        """Verify that the direct term D in VFKern is applied correctly."""
        # Create a simple kernel: 1/(L+I) + 5
        # Result should be (L+I)^-1 * X + 5
        mock_k = self.VFKern(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([5.0])
        )
        
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.convolve(self.X, mock_k)
            lp = conv.lowpass(self.X, [1.0])[0]
            
            # res should be lp + 5 (broadcasting over nBus, nTime)
            expected = lp[:, :, None] + 5.0
            np.testing.assert_allclose(res, expected)

    def test_vf_multi_dim_direct_term(self):
        """Verify direct term D broadcasting for multi-dimensional kernels."""
        # Kernel with 2 dimensions, D = [5, 10]
        mock_k = self.VFKern(
            Q=np.array([1.0]),
            R=np.array([[1.0, 2.0]]), # 1 pole, 2 dims
            D=np.array([5.0, 10.0])
        )
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.convolve(self.X, mock_k)
            lp = conv.lowpass(self.X, [1.0])[0]
            
            # Dim 0: lp + 5, Dim 1: 2*lp + 10
            np.testing.assert_allclose(res[:, :, 0], lp + 5.0)
            np.testing.assert_allclose(res[:, :, 1], 2.0 * lp + 10.0)

    def test_dy_vf_direct_term(self):
        """Verify direct term D in DyConvolve context."""
        vk = self.VFKern(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([10.0])
        )
        with self.sgwt.DyConvolve(self.L, vk) as conv:
            res = conv.convolve(self.X)
            lp = conv.lowpass(self.X)[0]
            expected = lp[:, :, None] + 10.0
            np.testing.assert_allclose(res, expected)

    def test_convolve_consistency(self):
        """Verify consistency between DyConvolve and Convolve results for all filters and VF."""
        poles = [1.0 / s for s in self.scales]
        vk = self.VFKern.from_dict(self.K)
        
        with self.sgwt.DyConvolve(self.L, vk) as dy_conv:
            dy_vf = dy_conv.convolve(self.X)
        
        with self.sgwt.DyConvolve(self.L, poles) as dy_conv:
            dy_lp = dy_conv.lowpass(self.X)
            dy_bp = dy_conv.bandpass(self.X)
            dy_hp = dy_conv.highpass(self.X)
            
        with self.sgwt.Convolve(self.L) as st_conv:
            st_vf = st_conv.convolve(self.X, vk)
            st_lp = st_conv.lowpass(self.X, self.scales)
            st_bp = st_conv.bandpass(self.X, self.scales)
            st_hp = st_conv.highpass(self.X, self.scales)
            
        np.testing.assert_allclose(dy_vf, st_vf, atol=1e-10)
        for dy, st in zip(dy_lp, st_lp):
            np.testing.assert_allclose(dy, st, atol=1e-10)
        for dy, st in zip(dy_bp, st_bp):
            np.testing.assert_allclose(dy, st, atol=1e-10)
        for dy, st in zip(dy_hp, st_hp):
            np.testing.assert_allclose(dy, st, atol=1e-10)

    def test_dynamic_convolution_and_updates(self):
        """Test DyConvolve with apriori poles and topology updates."""
        poles = [1.0 / s for s in self.scales]
        
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # Initial convolution
            lp_before = conv.lowpass(self.X)
            self.assertEqual(len(lp_before), len(poles))
            
            # Add a branch (edge) between node 100 and 200
            # This should change the Laplacian and thus the filter response
            ok = conv.addbranch(100, 200, 1.0)
            self.assertTrue(ok, "Failed to add branch via updown")
            
            lp_after = conv.lowpass(self.X)
            
            # Verify that the signal changed at the affected nodes
            # Node 200 should now see the impulse from node 100 more strongly
            diff = np.abs(lp_before[0] - lp_after[0])
            self.assertGreater(np.max(diff), 0, "Topology update did not affect convolution")

    def test_multiple_branch_updates(self):
        """Test adding multiple branches sequentially in DyConvolve."""
        poles = [1.0]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # Add two branches
            ok1 = conv.addbranch(10, 20, 1.0)
            ok2 = conv.addbranch(30, 40, 1.0)
            self.assertTrue(ok1 and ok2, "Failed to add multiple branches")
            res = conv.lowpass(self.X)
            self.assertEqual(len(res), 1)

    def test_convolve_empty_signal(self):
        """Verify that convolving an all-zero signal returns zeros."""
        X_zero = np.zeros_like(self.X)
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.lowpass(X_zero, self.scales)
            for r in res:
                self.assertTrue(np.all(r == 0), "Convolution of zero signal should be zero")

    def test_impulse_utility(self):
        """Verify the impulse signal generator."""
        imp = self.sgwt.impulse(self.L, n=5, ntime=2)
        self.assertEqual(imp.shape, (self.L.shape[0], 2))
        self.assertEqual(imp[5, 0], 1.0)
        self.assertEqual(imp[5, 1], 1.0)
        self.assertEqual(np.sum(imp), 2.0)

    def test_impulse_out_of_bounds(self):
        """Verify that impulse raises IndexError for invalid vertex index."""
        with self.assertRaises(IndexError):
            self.sgwt.impulse(self.L, n=self.L.shape[0] + 1)