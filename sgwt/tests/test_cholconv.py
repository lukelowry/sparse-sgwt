# -*- coding: utf-8 -*-
"""
Tests for Convolve and DyConvolve (cholconv module).

Tests graph convolution using Cholesky factorization with both static
and dynamic (topology-updating) contexts.
"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix

import sgwt
from sgwt.tests.conftest import requires_cholmod, SCALES


pytestmark = requires_cholmod


class TestConvolve:
    """Tests for static Convolve context manager."""

    def test_lowpass_returns_correct_shape(self, texas_laplacian, texas_signal):
        """Lowpass filter returns one result per scale with correct shape."""
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.lowpass(texas_signal, SCALES)
            assert len(results) == len(SCALES)
            assert all(r.shape == texas_signal.shape for r in results)

    def test_bandpass_returns_correct_shape(self, texas_laplacian, texas_signal):
        """Bandpass filter returns one result per scale."""
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.bandpass(texas_signal, SCALES)
            assert len(results) == len(SCALES)

    def test_highpass_returns_correct_shape(self, texas_laplacian, texas_signal):
        """Highpass filter returns one result per scale."""
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.highpass(texas_signal, SCALES)
            assert len(results) == len(SCALES)

    def test_lowpass_with_bset_subset(self, texas_laplacian):
        """Lowpass with Bset sparse subset runs correctly."""
        bset = csc_matrix(
            (np.ones(1), ([100], [0])), shape=(texas_laplacian.shape[0], 1)
        )
        # Bset requires 2D signal with single column
        X_single = sgwt.impulse(texas_laplacian, n=100, n_timesteps=1).reshape(-1, 1).copy(order='F')
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.lowpass(X_single, SCALES, Bset=bset)
            assert len(results) == len(SCALES)

    def test_zero_signal_returns_zero(self, texas_laplacian, texas_signal):
        """Convolving zero signal returns zero."""
        X_zero = np.zeros_like(texas_signal)
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.lowpass(X_zero, SCALES)
            for r in results:
                assert np.allclose(r, 0)

    def test_empty_signal_raises_error(self, texas_laplacian):
        """Empty signal (0 rows) raises ValueError."""
        X_empty = np.empty((0, 1), dtype=float)
        with sgwt.Convolve(texas_laplacian) as conv:
            with pytest.raises((ValueError, IndexError)):
                conv.lowpass(X_empty, SCALES)

    def test_mismatched_dimensions_raises_error(self, texas_laplacian):
        """Signal with mismatched dimensions raises ValueError."""
        n = texas_laplacian.shape[0]
        X_wrong_size = np.random.randn(n + 10, 1)  # Wrong number of nodes
        with sgwt.Convolve(texas_laplacian) as conv:
            with pytest.raises((ValueError, IndexError), match=".*"):
                conv.lowpass(X_wrong_size, SCALES)

    @pytest.mark.parametrize("order", [1, 2])
    def test_bandpass_order_equivalence(self, texas_laplacian, texas_signal, order):
        """Bandpass order=N equals applying order=1 filter N times."""
        scale = SCALES[0]
        with sgwt.Convolve(texas_laplacian) as conv:
            bp_direct = conv.bandpass(texas_signal, [scale], order=order)[0]
            # Apply iteratively
            result = texas_signal
            for _ in range(order):
                result = conv.bandpass(result, [scale], order=1)[0]
            np.testing.assert_allclose(bp_direct, result, atol=1e-9)

    def test_lowpass_refactor_false(self, texas_laplacian, texas_signal):
        """Lowpass with refactor=False skips numeric factorization."""
        scale = SCALES[0]
        with sgwt.Convolve(texas_laplacian) as conv:
            # First call with refactor=True to ensure factorization exists
            result_with = conv.lowpass(texas_signal, [scale], refactor=True)[0]

            # Second call with refactor=False reuses the factorization
            result_without = conv.lowpass(texas_signal, [scale], refactor=False)[0]

            # Results should be identical when using the same scale
            # This tests that refactor=False doesn't break anything
            np.testing.assert_allclose(result_with, result_without, atol=1e-10)

    def test_multiple_scales_work_correctly(self, texas_laplacian, texas_signal):
        """Convolving with multiple different scales produces different results."""
        scale1, scale2 = SCALES[0], SCALES[1]
        with sgwt.Convolve(texas_laplacian) as conv:
            # Convolution with scale1
            result1 = conv.lowpass(texas_signal, [scale1])[0]

            # Convolution with different scale2
            result2 = conv.lowpass(texas_signal, [scale2])[0]

            # Results should be different for different scales
            diff = np.abs(result1 - result2)
            min_expected_diff = 1e-6
            assert np.max(diff) > min_expected_diff, \
                f"Different scales should produce different results (max diff={np.max(diff):.2e})"

    def test_complex_signal_handling(self, texas_laplacian, texas_signal):
        """Complex signals are processed by splitting real/imag parts."""
        # Create a complex signal: x + i*x
        complex_signal = texas_signal + 1j * texas_signal
        scale = SCALES[0]
        
        with sgwt.Convolve(texas_laplacian) as conv:
            # Process complex signal
            results_complex = conv.lowpass(complex_signal, [scale])[0]
            
            # Process real part manually for verification
            results_real = conv.lowpass(texas_signal, [scale])[0]
            
            # Check linearity: L(x + iy) = L(x) + iL(y)
            expected = results_real + 1j * results_real
            np.testing.assert_allclose(results_complex, expected, atol=1e-10)
            assert np.iscomplexobj(results_complex)

    def test_convolve_complex_returns_array(self, texas_laplacian, texas_signal, library_kernel):
        """Convolve with complex signal returns ndarray (covers else branch in _process_signal)."""
        complex_signal = texas_signal + 1j * texas_signal
        
        with sgwt.Convolve(texas_laplacian) as conv:
            # convolve returns np.ndarray, not list, triggering the 'else' block
            result = conv.convolve(complex_signal, library_kernel)
            assert isinstance(result, np.ndarray)
            assert np.iscomplexobj(result)
            
            # Verify linearity
            real_res = conv.convolve(texas_signal, library_kernel)
            expected = real_res + 1j * real_res
            np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_c_contiguous_input_conversion(self, texas_laplacian, texas_signal):
        """C-contiguous input is converted to Fortran (covers B conversion in _process_signal)."""
        # Create 2D signal with >1 columns so C and F layouts are distinct
        c_signal = np.column_stack([texas_signal, texas_signal]).copy(order='C')
        assert not c_signal.flags['F_CONTIGUOUS']

        with sgwt.Convolve(texas_laplacian) as conv:
            # This calls _process_signal -> lowpass_impl
            res = conv.lowpass(c_signal, [1.0])
            assert len(res) == 1

    def test_float32_input_converted(self, texas_laplacian, texas_signal):
        """Float32 input is converted to float64 and processed correctly."""
        X_32 = texas_signal.astype(np.float32)
        with sgwt.Convolve(texas_laplacian) as conv:
            # Should not raise error
            res = conv.lowpass(X_32, SCALES)
            assert len(res) == len(SCALES)
            # Result should be close to float64 result
            res_64 = conv.lowpass(texas_signal, SCALES)
            np.testing.assert_allclose(res[0], res_64[0], atol=1e-5)

    def test_int_input_converted(self, texas_laplacian, texas_signal):
        """Integer input is converted to float64 and processed correctly."""
        # Create integer signal (rounded)
        X_int = np.round(texas_signal).astype(np.int32)
        with sgwt.Convolve(texas_laplacian) as conv:
            res = conv.lowpass(X_int, SCALES)
            assert len(res) == len(SCALES)

    def test_scalar_scale_returns_array(self, texas_laplacian, texas_signal):
        """Passing scalar scale returns single array instead of list."""
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.lowpass(texas_signal, 1.0)
            assert isinstance(result, np.ndarray)
            assert result.shape == texas_signal.shape

    def test_scalar_scale_matches_list_scale(self, texas_laplacian, texas_signal):
        """Scalar scale result matches first element of list scale result."""
        scale = 1.0
        with sgwt.Convolve(texas_laplacian) as conv:
            scalar_result = conv.lowpass(texas_signal, scale)
            list_result = conv.lowpass(texas_signal, [scale])
            np.testing.assert_allclose(scalar_result, list_result[0])

    def test_scalar_scale_bandpass(self, texas_laplacian, texas_signal):
        """Scalar scale works for bandpass filter."""
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.bandpass(texas_signal, 1.0)
            assert isinstance(result, np.ndarray)

    def test_scalar_scale_highpass(self, texas_laplacian, texas_signal):
        """Scalar scale works for highpass filter."""
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.highpass(texas_signal, 1.0)
            assert isinstance(result, np.ndarray)

    def test_1d_signal_lowpass(self, texas_laplacian):
        """1D signal input returns 1D output for lowpass."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.lowpass(signal_1d, 1.0)
            assert result.ndim == 1
            assert result.shape[0] == texas_laplacian.shape[0]

    def test_1d_signal_matches_2d(self, texas_laplacian):
        """1D signal result matches 2D signal with single column."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        signal_2d = signal_1d.reshape(-1, 1)
        with sgwt.Convolve(texas_laplacian) as conv:
            result_1d = conv.lowpass(signal_1d, 1.0)
            result_2d = conv.lowpass(signal_2d, 1.0)
            np.testing.assert_allclose(result_1d, result_2d.squeeze())

    def test_1d_signal_bandpass(self, texas_laplacian):
        """1D signal input works for bandpass filter."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.bandpass(signal_1d, 1.0)
            assert result.ndim == 1

    def test_1d_signal_highpass(self, texas_laplacian):
        """1D signal input works for highpass filter."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.highpass(signal_1d, 1.0)
            assert result.ndim == 1

    def test_1d_signal_convolve(self, texas_laplacian, library_kernel):
        """1D signal input works for convolve."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.convolve(signal_1d, library_kernel)
            # Result should be 2D: (n_vertices, nDim)
            assert result.ndim == 2
            assert result.shape[0] == texas_laplacian.shape[0]

    def test_1d_signal_list_scales(self, texas_laplacian):
        """1D signal with list of scales returns list of 1D arrays."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        with sgwt.Convolve(texas_laplacian) as conv:
            results = conv.lowpass(signal_1d, SCALES)
            assert isinstance(results, list)
            assert len(results) == len(SCALES)
            for r in results:
                assert r.ndim == 1


class TestConvolveVFKernel:
    """Tests for VFKernel convolution in Convolve context."""

    def test_vfkernel_from_dict(self, texas_laplacian, texas_signal, library_kernel):
        """VFKernel from dict produces valid output."""
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.convolve(texas_signal, library_kernel)
            assert result.shape[0] == texas_laplacian.shape[0]

    def test_vfkernel_object_matches_dict(self, texas_laplacian, texas_signal, library_kernel):
        """VFKernel object produces same result as dict."""
        vk = sgwt.VFKernel.from_dict(library_kernel)
        with sgwt.Convolve(texas_laplacian) as conv:
            res_dict = conv.convolve(texas_signal, library_kernel)
            res_obj = conv.convolve(texas_signal, vk)
            np.testing.assert_allclose(res_dict, res_obj)

    def test_invalid_kernel_raises_typeerror(self, texas_laplacian, texas_signal):
        """Invalid kernel type raises TypeError with helpful message."""
        with sgwt.Convolve(texas_laplacian) as conv:
            with pytest.raises(TypeError, match=".*Kernel.*"):
                conv.convolve(texas_signal, "not a kernel")

    def test_empty_kernel_raises_valueerror(self, texas_laplacian, texas_signal):
        """Empty VFKernel raises ValueError with descriptive message."""
        empty_kernel = sgwt.VFKernel(Q=None, R=None, D=None)
        with sgwt.Convolve(texas_laplacian) as conv:
            with pytest.raises(ValueError, match=".*residues.*poles.*"):
                conv.convolve(texas_signal, empty_kernel)

    def test_vfkernel_without_direct_term(self, texas_laplacian, texas_signal):
        """VFKernel with empty D term works correctly (no direct term added)."""
        kernel = sgwt.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([])  # Empty direct term
        )
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.convolve(texas_signal, kernel)
            lp = conv.lowpass(texas_signal, [1.0])[0]
            # Without D, result should just be the lowpass response
            np.testing.assert_allclose(result.squeeze(), lp.squeeze(), atol=1e-10)

    def test_direct_term_applied(self, texas_laplacian, texas_signal):
        """Direct term D is correctly applied."""
        kernel = sgwt.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([5.0])
        )
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.convolve(texas_signal, kernel)
            lp = conv.lowpass(texas_signal, 1.0)
            # With 1D signal: result is (n_vertices, nDim), lp is (n_vertices,)
            expected = lp[:, None] + texas_signal[:, None] * 5.0
            np.testing.assert_allclose(result, expected)

    def test_multidim_direct_term_broadcasts(self, texas_laplacian, texas_signal):
        """Multi-dimensional D broadcasts correctly."""
        kernel = sgwt.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0, 2.0]]),
            D=np.array([5.0, 10.0])
        )
        with sgwt.Convolve(texas_laplacian) as conv:
            result = conv.convolve(texas_signal, kernel)
            lp = conv.lowpass(texas_signal, 1.0)
            # With 1D signal: result is (n_vertices, nDim), lp is (n_vertices,)
            np.testing.assert_allclose(result[:, 0], lp + texas_signal * 5.0)
            np.testing.assert_allclose(result[:, 1], 2.0 * lp + texas_signal * 10.0)


class TestDyConvolve:
    """Tests for dynamic DyConvolve context manager."""

    def test_analytical_filters(self, texas_laplacian, texas_signal):
        """Analytical filters work in DyConvolve."""
        poles = [1.0 / s for s in SCALES]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            lp = conv.lowpass(texas_signal)
            bp = conv.bandpass(texas_signal)
            hp = conv.highpass(texas_signal)
            assert len(lp) == len(poles)
            assert len(bp) == len(poles)
            assert len(hp) == len(poles)

    def test_vfkernel_direct_term(self, texas_laplacian, texas_signal):
        """Direct term D applied correctly in DyConvolve."""
        kernel = sgwt.VFKernel(
            Q=np.array([1.0]),
            R=np.array([[1.0]]),
            D=np.array([10.0])
        )
        with sgwt.DyConvolve(texas_laplacian, kernel) as conv:
            result = conv.convolve(texas_signal)
            lp = conv.lowpass(texas_signal)[0]
            # With 1D signal: result is (n_vertices, nDim), lp is (n_vertices,)
            expected = lp[:, None] + texas_signal[:, None] * 10.0
            np.testing.assert_allclose(result, expected)

    def test_consistency_with_static_convolve(self, texas_laplacian, texas_signal, library_kernel):
        """DyConvolve produces same results as Convolve."""
        poles = [1.0 / s for s in SCALES]
        vk = sgwt.VFKernel.from_dict(library_kernel)

        with sgwt.DyConvolve(texas_laplacian, vk) as dy:
            dy_vf = dy.convolve(texas_signal)

        with sgwt.DyConvolve(texas_laplacian, poles) as dy:
            dy_lp = dy.lowpass(texas_signal)
            dy_bp = dy.bandpass(texas_signal)
            dy_hp = dy.highpass(texas_signal)

        with sgwt.Convolve(texas_laplacian) as st:
            st_vf = st.convolve(texas_signal, vk)
            st_lp = st.lowpass(texas_signal, SCALES)
            st_bp = st.bandpass(texas_signal, SCALES)
            st_hp = st.highpass(texas_signal, SCALES)

        np.testing.assert_allclose(dy_vf, st_vf, atol=1e-10)
        for dy_r, st_r in zip(dy_lp, st_lp):
            np.testing.assert_allclose(dy_r, st_r, atol=1e-10)
        for dy_r, st_r in zip(dy_bp, st_bp):
            np.testing.assert_allclose(dy_r, st_r, atol=1e-10)
        for dy_r, st_r in zip(dy_hp, st_hp):
            np.testing.assert_allclose(dy_r, st_r, atol=1e-10)

    def test_float32_input_converted(self, texas_laplacian, texas_signal):
        """Float32 input is converted to float64 in DyConvolve."""
        X_32 = texas_signal.astype(np.float32)
        poles = [1.0 / s for s in SCALES]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            res = conv.lowpass(X_32)
            assert len(res) == len(poles)

    def test_1d_signal_lowpass(self, texas_laplacian):
        """1D signal input returns list of 1D outputs for lowpass."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        poles = [1.0 / s for s in SCALES]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            results = conv.lowpass(signal_1d)
            assert len(results) == len(poles)
            for r in results:
                assert r.ndim == 1
                assert r.shape[0] == texas_laplacian.shape[0]

    def test_1d_signal_matches_2d(self, texas_laplacian):
        """1D signal result matches 2D signal with single column."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        signal_2d = signal_1d.reshape(-1, 1)
        poles = [1.0]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            result_1d = conv.lowpass(signal_1d)[0]
            result_2d = conv.lowpass(signal_2d)[0]
            np.testing.assert_allclose(result_1d, result_2d.squeeze())

    def test_1d_signal_bandpass(self, texas_laplacian):
        """1D signal input works for bandpass filter."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        poles = [1.0]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            results = conv.bandpass(signal_1d)
            for r in results:
                assert r.ndim == 1

    def test_1d_signal_highpass(self, texas_laplacian):
        """1D signal input works for highpass filter."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        poles = [1.0]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            results = conv.highpass(signal_1d)
            for r in results:
                assert r.ndim == 1

    def test_1d_signal_convolve(self, texas_laplacian, library_kernel):
        """1D signal input works for convolve."""
        signal_1d = np.random.randn(texas_laplacian.shape[0])
        vk = sgwt.VFKernel.from_dict(library_kernel)
        with sgwt.DyConvolve(texas_laplacian, vk) as conv:
            result = conv.convolve(signal_1d)
            # Result should be 2D: (n_vertices, nDim)
            assert result.ndim == 2
            assert result.shape[0] == texas_laplacian.shape[0]


class TestDyConvolveTopology:
    """Tests for DyConvolve topology updates (addbranch)."""

    def test_addbranch_modifies_response(self, texas_laplacian, texas_signal):
        """Adding a branch changes the filter response."""
        poles = [1.0 / s for s in SCALES]
        with sgwt.DyConvolve(texas_laplacian, poles) as conv:
            lp_before = conv.lowpass(texas_signal)
            ok = conv.addbranch(100, 200, 1.0)
            assert ok, "addbranch should succeed"
            lp_after = conv.lowpass(texas_signal)
            diff = np.abs(lp_before[0] - lp_after[0])
            # Topology update should cause measurable change (> 1e-6)
            min_change = 1e-6
            max_diff = np.max(diff)
            assert max_diff > min_change, \
                f"Topology update should significantly affect response (max diff={max_diff:.2e}, expected >{min_change:.2e})"

    def test_multiple_branch_updates(self, texas_laplacian, texas_signal):
        """Multiple sequential branch additions work."""
        with sgwt.DyConvolve(texas_laplacian, [1.0]) as conv:
            ok1 = conv.addbranch(10, 20, 1.0)
            ok2 = conv.addbranch(30, 40, 1.0)
            assert ok1 and ok2
            result = conv.lowpass(texas_signal)
            assert len(result) == 1

    def test_out_of_bounds_indices_fail_gracefully(self, texas_laplacian, texas_signal):
        """Out-of-bounds node indices return False, not crash."""
        n = texas_laplacian.shape[0]
        with sgwt.DyConvolve(texas_laplacian, [1.0]) as conv:
            ok = conv.addbranch(n, n + 1, 1.0)
            assert not ok

    def test_negative_weight_raises_error(self, texas_laplacian):
        """Negative edge weight raises ValueError."""
        with sgwt.DyConvolve(texas_laplacian, [1.0]) as conv:
            with pytest.raises(ValueError, match="domain error"):
                conv.addbranch(10, 20, -1.0)


class TestImpulse:
    """Tests for impulse signal generator utility."""

    def test_impulse_default_1d(self, texas_laplacian):
        """Default impulse (n_timesteps=1) returns 1D array."""
        imp = sgwt.impulse(texas_laplacian, n=5)
        assert imp.ndim == 1
        assert imp.shape == (texas_laplacian.shape[0],)
        assert imp[5] == 1.0
        assert np.sum(imp) == 1.0

    def test_impulse_shape_and_values(self, texas_laplacian):
        """Impulse with n_timesteps>1 has correct 2D shape."""
        imp = sgwt.impulse(texas_laplacian, n=5, n_timesteps=2)
        assert imp.shape == (texas_laplacian.shape[0], 2)
        assert imp[5, 0] == 1.0
        assert imp[5, 1] == 1.0
        assert np.sum(imp) == 2.0

    def test_impulse_invalid_node_raises(self, texas_laplacian):
        """Out-of-bounds node index raises IndexError."""
        with pytest.raises(IndexError):
            sgwt.impulse(texas_laplacian, n=texas_laplacian.shape[0] + 1)
