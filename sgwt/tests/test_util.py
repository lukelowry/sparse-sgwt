# -*- coding: utf-8 -*-
"""
Tests for utility functions, resource loading, and data integrity.
"""
import numpy as np
import pytest
from ctypes import CDLL
from scipy.sparse import csc_matrix

import sgwt
from sgwt.tests.conftest import requires_cholmod, requires_klu, HAS_CHOLMOD, HAS_KLU


class TestDLLLoading:
    """Tests for DLL loading utilities."""

    @requires_cholmod
    def test_cholmod_dll_loads(self):
        """CHOLMOD DLL loads successfully."""
        dll = sgwt.get_cholmod_dll()
        assert isinstance(dll, CDLL)

    @requires_klu
    def test_klu_dll_loads(self):
        """KLU DLL loads successfully."""
        dll = sgwt.get_klu_dll()
        assert isinstance(dll, CDLL)


class TestLibraryKernels:
    """Tests for built-in VFKernel loading."""

    @pytest.mark.parametrize("kernel_name", [
        "MEXICAN_HAT", "MODIFIED_MORLET", "SHANNON"
    ])
    def test_kernel_loads_with_valid_data(self, kernel_name):
        """Built-in kernels load with non-empty poles and residues."""
        kernel_dict = getattr(sgwt, kernel_name)
        kern = sgwt.VFKernel.from_dict(kernel_dict)
        assert isinstance(kern, sgwt.VFKernel)
        assert len(kern.Q) > 0
        assert len(kern.R) > 0

    def test_vfkernel_from_dict_parses_correctly(self):
        """VFKernel.from_dict correctly parses poles, residues, and D."""
        mock_data = {
            'poles': [
                {'q': 1.0, 'r': [0.1, 0.2]},
                {'q': 2.0, 'r': [0.3, 0.4]}
            ],
            'd': [0.5, 0.6]
        }
        kern = sgwt.VFKernel.from_dict(mock_data)
        np.testing.assert_array_equal(kern.Q, [1.0, 2.0])
        np.testing.assert_array_equal(kern.R, [[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_array_equal(kern.D, [0.5, 0.6])


class TestChebyKernelEdgeCases:
    """Tests for ChebyKernel edge cases."""

    def test_evaluate_empty_coefficients(self):
        """ChebyKernel.evaluate with empty C returns empty array."""
        kern = sgwt.ChebyKernel(C=np.array([]).reshape(0, 0), spectrum_bound=1.0)
        result = kern.evaluate(np.array([0.5]))
        assert result.shape == (1, 0)


class TestLibraryLaplacians:
    """Tests for built-in Laplacian loading."""

    @pytest.mark.parametrize("laplacian_name", [
        "DELAY_TEXAS", "IMPEDANCE_HAWAII", "LENGTH_WECC"
    ])
    def test_laplacian_is_square_csc(self, laplacian_name):
        """Built-in Laplacians are square csc_matrix with nonzero entries."""
        L = getattr(sgwt, laplacian_name)
        # Use .format check for coverage compatibility (scipy class identity issues)
        assert L.format == "csc"
        assert L.shape[0] == L.shape[1]
        assert L.nnz > 0

    def test_laplacian_is_symmetric(self):
        """Built-in Laplacians are symmetric."""
        L = sgwt.DELAY_TEXAS
        # Use toarray() to avoid coverage-induced sparse class identity issues
        assert np.allclose(L.toarray(), L.T.toarray())


class TestLibrarySignals:
    """Tests for built-in coordinate signals."""

    @pytest.mark.parametrize("signal_name", ["COORD_TEXAS", "COORD_USA"])
    def test_signal_is_2d_array(self, signal_name):
        """Coordinate signals are 2D numpy arrays."""
        S = getattr(sgwt, signal_name)
        assert isinstance(S, np.ndarray)
        assert S.ndim == 2
        assert S.shape[1] in [2, 3]

    def test_laplacian_signal_dimension_match(self):
        """Laplacian and signal node counts match."""
        assert sgwt.DELAY_TEXAS.shape[0] == sgwt.COORD_TEXAS.shape[0]
        assert sgwt.DELAY_USA.shape[0] == sgwt.COORD_USA.shape[0]


class TestChebyKernelFromDict:
    """Tests for ChebyKernel.from_dict parsing."""

    def test_empty_approximations(self):
        """Empty approximations returns empty C array."""
        data = {'spectrum_bound': 2.0, 'approximations': []}
        kern = sgwt.ChebyKernel.from_dict(data)
        assert kern.C.shape == (0, 0)
        assert kern.spectrum_bound == 2.0

    def test_missing_approximations_key(self):
        """Missing 'approximations' key treated as empty."""
        data = {'spectrum_bound': 1.5}
        kern = sgwt.ChebyKernel.from_dict(data)
        assert kern.C.shape == (0, 0)

    def test_valid_approximations(self):
        """Valid approximations are stacked correctly."""
        data = {
            'spectrum_bound': 3.0,
            'approximations': [
                {'coeffs': [1.0, 2.0, 3.0]},
                {'coeffs': [4.0, 5.0, 6.0]}
            ]
        }
        kern = sgwt.ChebyKernel.from_dict(data)
        assert kern.C.shape == (3, 2)
        np.testing.assert_array_equal(kern.C[:, 0], [1.0, 2.0, 3.0])

    def test_mismatched_coeffs_raises(self):
        """Mismatched coefficient lengths raise ValueError."""
        data = {
            'spectrum_bound': 1.0,
            'approximations': [
                {'coeffs': [1.0, 2.0]},
                {'coeffs': [3.0, 4.0, 5.0]}
            ]
        }
        with pytest.raises(ValueError, match="same length"):
            sgwt.ChebyKernel.from_dict(data)


class TestChebyKernelFromFunction:
    """Tests for ChebyKernel.from_function edge cases."""

    def test_zero_function_keeps_constant_term(self):
        """Fitting a zero function keeps at least the constant term."""
        kern = sgwt.ChebyKernel.from_function(lambda x: np.zeros_like(x), order=5, spectrum_bound=1.0)
        assert kern.C.shape[0] >= 1

    def test_multioutput_function_preserves_2d_coeffs(self):
        """Fitting a multi-output function preserves 2D coefficient structure."""
        # Function returning 2D array (multi-output)
        def multi_func(x):
            return np.column_stack([np.exp(-x), np.sin(x)])
        
        kern = sgwt.ChebyKernel.from_function(multi_func, order=5, spectrum_bound=4.0)
        # Should have 2 dimensions (one per output)
        assert kern.C.shape[1] == 2
        # Verify evaluation works for both outputs
        x_test = np.linspace(0, 4, 10)
        result = kern.evaluate(x_test)
        assert result.shape == (10, 2)

    def test_order_less_than_one_raises_valueerror(self):
        """Order < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Order must be >= 1"):
            sgwt.ChebyKernel.from_function(lambda x: x, order=0, spectrum_bound=1.0)


class TestMatLoader:
    """Tests for _mat_loader edge cases."""

    def test_empty_mat_raises(self, tmp_path):
        """MAT file with no variables raises ValueError."""
        from scipy.io import savemat
        from sgwt.util import _mat_loader
        mat_path = tmp_path / "empty.mat"
        savemat(str(mat_path), {})
        with pytest.raises(ValueError, match="No data variables"):
            _mat_loader(str(mat_path))

    def test_multiple_variables_stacked(self, tmp_path):
        """MAT file with multiple variables stacks them into columns."""
        from scipy.io import savemat
        from sgwt.util import _mat_loader
        mat_path = tmp_path / "multi.mat"
        savemat(str(mat_path), {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])})
        result = _mat_loader(str(mat_path))
        assert result.shape == (3, 2)

    def test_dense_to_csc_conversion(self, tmp_path):
        """Dense matrix is converted to CSC when to_csc=True."""
        from scipy.io import savemat
        from sgwt.util import _mat_loader
        mat_path = tmp_path / "dense.mat"
        savemat(str(mat_path), {'L': np.eye(3)})
        result = _mat_loader(str(mat_path), to_csc=True)
        assert result.format == "csc"
        assert result.shape == (3, 3)

    def test_single_variable_no_transpose(self, tmp_path):
        """Single 2D variable with multiple rows returns unchanged."""
        from scipy.io import savemat
        from sgwt.util import _mat_loader
        mat_path = tmp_path / "single.mat"
        savemat(str(mat_path), {'x': np.array([[1, 2], [3, 4], [5, 6]])})
        result = _mat_loader(str(mat_path))
        assert result.shape == (3, 2)


class TestDLLLoadingErrors:
    """Tests for DLL loading error paths."""

    def test_oserror_gives_helpful_message(self):
        """OSError during DLL load provides helpful error message."""
        from unittest.mock import patch
        from sgwt.util import _load_dll
        with patch('sgwt.util.CDLL', side_effect=OSError("cannot load")):
            with pytest.raises(OSError, match="Failed to load DLL"):
                _load_dll("fake.dll")


class TestModuleDir:
    """Tests for module __dir__ function."""

    def test_includes_lazy_registry(self):
        """Module __dir__ includes lazy-loaded resources."""
        import sgwt.util
        names = dir(sgwt.util)
        assert 'MEXICAN_HAT' in names
        assert 'DELAY_TEXAS' in names
        assert 'ChebyKernel' in names


class TestResourceErrors:
    """Tests for error handling in resource loading."""

    def test_nonexistent_resource_raises_filenotfounderror(self):
        """Loading non-existent resource raises FileNotFoundError."""
        from sgwt.util import _load_resource
        with pytest.raises(FileNotFoundError):
            _load_resource("library/NON_EXISTENT_FILE.mat", lambda p: p)


class TestEstimateSpectralBound:
    """Tests for estimate_spectral_bound utility."""

    def test_returns_positive_value(self, small_laplacian):
        """Spectral bound estimate is positive."""
        bound = sgwt.estimate_spectral_bound(small_laplacian)
        assert bound > 0

    def test_bound_exceeds_max_eigenvalue(self, small_laplacian):
        """Bound is >= largest eigenvalue (with small margin)."""
        from scipy.sparse.linalg import eigsh
        bound = sgwt.estimate_spectral_bound(small_laplacian)
        # Compute actual max eigenvalue
        max_eig = eigsh(small_laplacian.astype(float), k=1, which='LM', return_eigenvectors=False)[0]
        assert bound >= max_eig * 0.99  # allow small numerical tolerance


class TestChebyKernelFromFunctionOnGraph:
    """Tests for ChebyKernel.from_function_on_graph convenience method."""

    def test_creates_kernel_from_graph(self, small_laplacian):
        """from_function_on_graph estimates spectral bound and fits kernel."""
        from sgwt.util import ChebyKernel
        kernel = ChebyKernel.from_function_on_graph(
            small_laplacian, lambda x: np.exp(-x), order=10
        )
        assert kernel.C.shape[0] > 0
        assert kernel.spectrum_bound > 0


class TestChebyKernelEvaluate:
    """Tests for ChebyKernel.evaluate method."""

    def test_evaluate_multidimensional(self):
        """evaluate returns 2D array for multi-column coefficients."""
        from sgwt.util import ChebyKernel
        # Create kernel with 2 columns of coefficients (2 filters)
        C = np.array([[1.0, 0.5], [0.5, 0.25], [0.1, 0.05]])
        kernel = ChebyKernel(C=C, spectrum_bound=2.0)
        x = np.array([0.0, 1.0, 2.0])
        result = kernel.evaluate(x)
        assert result.shape == (3, 2)


class TestImpulse:
    """Tests for impulse signal generator."""

    def test_impulse_creates_correct_signal(self, small_laplacian):
        """impulse creates signal with 1 at specified vertex."""
        signal = sgwt.impulse(small_laplacian, n=2, n_timesteps=5)
        assert signal.shape == (small_laplacian.shape[0], 5)
        assert signal[2, 0] == 1.0
        assert np.sum(signal[:, 0]) == 1.0