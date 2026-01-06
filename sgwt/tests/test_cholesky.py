# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/test_cholesky.py
Description: Core functionality tests validating filters and dynamic updates 
             without external dependencies like sksparse.
"""
import numpy as np
import pytest

class TestCholesky:
    
    def setup_method(self, method):
        import sgwt
        self.sgwt = sgwt
        self.L = sgwt.DELAY_TEXAS
        self.K = sgwt.MODIFIED_MORLET
        self.VFKernel = sgwt.VFKernel
        
        self.X = self.sgwt.impulse(self.L, n=100)
        self.scales = [0.1, 1.0, 10.0]

    def test_static_convolve_with_analytical_filters(self):
        """Test analytical filters (low, band, high) in static Convolve context."""
        with self.sgwt.Convolve(self.L) as conv:
            # Low-pass
            lp = conv.lowpass(self.X, self.scales)
            assert len(lp) == len(self.scales)
            assert lp[0].shape == self.X.shape
            
            # Band-pass
            bp = conv.bandpass(self.X, self.scales)
            assert len(bp) == len(self.scales)
            
            # High-pass
            hp = conv.highpass(self.X, self.scales)
            assert len(hp) == len(self.scales)

    def test_dynamic_convolve_with_analytical_filters(self):
        """Test analytical filters (low, band, high) in dynamic DyConvolve context."""
        poles = [1.0 / s for s in self.scales]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            lp = conv.lowpass(self.X)
            assert len(lp) == len(poles)
            
            bp = conv.bandpass(self.X)
            assert len(bp) == len(poles)
            
            hp = conv.highpass(self.X)
            assert len(hp) == len(poles)

    def test_lowpass_with_sparse_bset_runs_correctly(self):
        """Verify low-pass filter with a sparse subset (Bset) runs correctly."""
        from scipy.sparse import csc_matrix
        bset = csc_matrix((np.ones(1), ([100], [0])), shape=(self.L.shape[0], 1))
        X_single = self.X[:, :1].copy(order='F')
        
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.lowpass(X_single, self.scales, Bset=bset)
            assert len(res) == len(self.scales)

    def test_vf_kernel_convolution_with_dict_and_object(self):
        """Test VF kernel convolution using both a raw dict and a VFKernel object."""
        with self.sgwt.Convolve(self.L) as conv:
            # Test with raw dict from library
            res = conv.convolve(self.X, self.K)
            assert res.shape[0] == self.L.shape[0]
            
            # Test with VFKernel object
            vk = self.VFKernel.from_dict(self.K)
            res_vk = conv.convolve(self.X, vk)
            np.testing.assert_allclose(res, res_vk)

    def test_vf_kernel_convolution_raises_errors_for_invalid_input(self):
        """Verify convolve raises appropriate errors for invalid kernel inputs."""
        with self.sgwt.Convolve(self.L) as conv:
            with pytest.raises(TypeError):
                conv.convolve(self.X, "not a kernel")
            
            with pytest.raises(ValueError):
                conv.convolve(self.X, self.VFKernel(Q=None, R=None, D=None))

    def test_vf_kernel_direct_term_is_applied_correctly(self):
        """Verify that the direct term D in a VFKernel is applied correctly."""
        # Create a simple kernel: 1/(L+I) + 5
        # Result should be (L+I)^-1 * X + 5
        mock_k = self.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([5.0])
        )
        
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.convolve(self.X, mock_k)
            lp = conv.lowpass(self.X, [1.0])[0]
            
            # res should be lp + 5 (broadcasting over nBus, nTime)
            expected = lp[:, :, None] + self.X[:, :, None] * 5.0
            np.testing.assert_allclose(res, expected)

    def test_vf_kernel_multidim_direct_term_broadcasts_correctly(self):
        """Verify direct term D broadcasting for multi-dimensional VF kernels."""
        # Kernel with 2 dimensions, D = [5, 10]
        mock_k = self.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0, 2.0]]), # 1 pole, 2 dims
            D=np.array([5.0, 10.0])
        )
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.convolve(self.X, mock_k)
            lp = conv.lowpass(self.X, [1.0])[0]
            
            # Dim 0: lp + 5, Dim 1: 2*lp + 10
            np.testing.assert_allclose(res[:, :, 0], lp + self.X * 5.0)
            np.testing.assert_allclose(res[:, :, 1], 2.0 * lp + self.X * 10.0)

    def test_dynamic_convolve_applies_vf_kernel_direct_term(self):
        """Verify the direct term D of a VF kernel is applied in DyConvolve."""
        vk = self.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([10.0])
        )
        with self.sgwt.DyConvolve(self.L, vk) as conv:
            res = conv.convolve(self.X)
            lp = conv.lowpass(self.X)[0]
            expected = lp[:, :, None] + self.X[:, :, None] * 10.0
            np.testing.assert_allclose(res, expected)

    def test_static_and_dynamic_convolve_produce_consistent_results(self):
        """Verify consistency between DyConvolve and Convolve for all filter types."""
        poles = [1.0 / s for s in self.scales]
        vk = self.VFKernel.from_dict(self.K)
        
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

    def test_dynamic_convolve_updates_topology_with_addbranch(self):
        """Test DyConvolve with topology updates via the addbranch method."""
        poles = [1.0 / s for s in self.scales]
        
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # Initial convolution
            lp_before = conv.lowpass(self.X)
            assert len(lp_before) == len(poles)
            
            # Add a branch (edge) between node 100 and 200
            # This should change the Laplacian and thus the filter response
            ok = conv.addbranch(100, 200, 1.0)
            assert ok, "Failed to add branch via updown"
            
            lp_after = conv.lowpass(self.X)
            
            # Verify that the signal changed at the affected nodes
            # Node 200 should now see the impulse from node 100 more strongly
            diff = np.abs(lp_before[0] - lp_after[0])
            assert np.max(diff) > 0, "Topology update did not affect convolution"

    def test_dynamic_convolve_handles_multiple_branch_updates(self):
        """Test adding multiple branches sequentially in a DyConvolve context."""
        poles = [1.0]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # Add two branches
            ok1 = conv.addbranch(10, 20, 1.0)
            ok2 = conv.addbranch(30, 40, 1.0)
            assert ok1 and ok2, "Failed to add multiple branches"
            res = conv.lowpass(self.X)
            assert len(res) == 1

    def test_convolution_on_zero_signal_returns_zero(self):
        """Verify that convolving an all-zero signal returns an all-zero signal."""
        X_zero = np.zeros_like(self.X)
        with self.sgwt.Convolve(self.L) as conv:
            res = conv.lowpass(X_zero, self.scales)
            for r in res:
                assert np.all(r == 0), "Convolution of zero signal should be zero"

    def test_impulse_signal_generator_utility(self):
        """Verify the impulse signal generator."""
        imp = self.sgwt.impulse(self.L, n=5, n_timesteps=2)
        assert imp.shape == (self.L.shape[0], 2)
        assert imp[5, 0] == 1.0
        assert imp[5, 1] == 1.0
        assert np.sum(imp) == 2.0

    def test_impulse_generator_raises_indexerror_for_invalid_node(self):
        """Verify impulse() raises IndexError for an out-of-bounds vertex index."""
        with pytest.raises(IndexError):
            self.sgwt.impulse(self.L, n=self.L.shape[0] + 1)

    def test_addbranch_with_out_of_bounds_indices_is_handled(self):
        """Verify addbranch handles out-of-bounds node indices without crashing."""
        poles = [1.0]
        n_nodes = self.L.shape[0]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # CHOLMOD's C code could crash here if not handled.
            # The wrapper should catch this and fail gracefully. We test that
            # it doesn't segfault and returns False.
            ok = conv.addbranch(n_nodes, n_nodes + 1, 1.0)
            assert not ok, "addbranch should fail for out-of-bounds indices"

    def test_addbranch_with_negative_weight_raises_error(self):
        """Verify addbranch raises a ValueError for negative weights due to sqrt."""
        poles = [1.0]
        with self.sgwt.DyConvolve(self.L, poles) as conv:
            # The implementation uses np.sqrt(w), which will fail for w < 0.
            with pytest.raises(ValueError, match="domain error"):
                conv.addbranch(10, 20, -1.0)

    def test_bandpass_with_order_greater_than_one(self):
        """Verify bandpass with order=2 is equivalent to applying the filter twice."""
        with self.sgwt.Convolve(self.L) as conv:
            # Compute with order=2 directly for the first scale
            bp_order2 = conv.bandpass(self.X, [self.scales[0]], order=2)
            # Compute by applying order=1 filter twice
            bp_order1_pass1 = conv.bandpass(self.X, [self.scales[0]], order=1)
            bp_order1_pass2 = conv.bandpass(bp_order1_pass1[0], [self.scales[0]], order=1)
            np.testing.assert_allclose(bp_order2[0], bp_order1_pass2[0], atol=1e-9)