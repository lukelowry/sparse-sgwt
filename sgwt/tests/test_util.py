# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/test_util.py
Description: Tests for general utilities, resource loading, and data integrity.
"""
import pytest
import numpy as np
from scipy.sparse import csc_matrix

from ctypes import CDLL

class TestSGWTUtil:

    def setup_method(self, method):
        import sgwt
        self.sgwt = sgwt

    def test_cholmod_dll_is_locatable_and_loadable(self):
        """Verify that the CHOLMOD DLL can be located and loaded."""
        try:
            dll = self.sgwt.get_cholmod_dll()
            assert isinstance(dll, CDLL)
        except OSError as e:
            pytest.fail(f"CHOLMOD DLL Load Error (OSError): {e}")
        except Exception as e:
            pytest.fail(f"get_cholmod_dll raised Exception: {e}")

    def test_klu_dll_is_locatable_and_loadable(self):
        """Verify that the KLU DLL can be located and loaded."""
        try:
            dll = self.sgwt.get_klu_dll()
            assert isinstance(dll, CDLL)
        except OSError as e:
            pytest.fail(f"KLU DLL Load Error (OSError): {e}")
        except Exception as e:
            pytest.fail(f"get_klu_dll raised Exception: {e}")

    def test_builtin_vf_kernels_load_with_valid_data(self):
        """Verify built-in kernels load as VFKernel objects with valid data."""
        kernels = [self.sgwt.MEXICAN_HAT, self.sgwt.MODIFIED_MORLET, self.sgwt.SHANNON]
        for kjson in kernels:
            k = self.sgwt.VFKernel.from_dict(kjson)
            assert isinstance(k, self.sgwt.VFKernel)
            assert len(k.Q) > 0, "Kernel poles (Q) should not be empty"
            assert len(k.R) > 0, "Kernel residues (R) should not be empty"

    def test_builtin_laplacians_load_as_square_csc_matrix(self):
        """Verify built-in Laplacians load as square csc_matrix instances."""
        # Test a representative subset of Laplacians
        laps = [self.sgwt.DELAY_TEXAS, self.sgwt.IMPEDANCE_HAWAII, self.sgwt.LENGTH_WECC]
        for L in laps:
            assert isinstance(L, csc_matrix)
            assert L.shape[0] == L.shape[1], "Laplacian must be square"
            assert L.nnz > 0, "Laplacian should have non-zero entries"

    def test_builtin_laplacians_are_symmetric(self):
        """Verify that built-in Laplacians are symmetric."""
        L = self.sgwt.DELAY_TEXAS
        # Check symmetry: L - L.T should be zero
        diff = (L - L.T).tocsr()
        assert diff.nnz == 0, "Laplacian should be symmetric"

    def test_builtin_signals_load_as_2d_numpy_arrays(self):
        """Verify built-in signals load with correct type and dimensions."""
        # Test a representative subset of signals
        signals = [self.sgwt.COORD_TEXAS, self.sgwt.COORD_USA]
        for S in signals:
            assert isinstance(S, np.ndarray)
            assert S.ndim == 2, "Coordinate signals should be 2D (N x Dim)"
            assert S.shape[1] in [2, 3], "Coordinates should typically be 2D or 3D"

    def test_builtin_laplacians_and_signals_have_matching_node_counts(self):
        """Ensure built-in Laplacians and their associated signals have matching node counts."""
        assert self.sgwt.DELAY_TEXAS.shape[0] == self.sgwt.COORD_TEXAS.shape[0]
        assert self.sgwt.DELAY_USA.shape[0] == self.sgwt.COORD_USA.shape[0]

    def test_vfkernel_from_dict_factory_parses_correctly(self):
        """Test the VFKernel.from_dict factory method with mock data."""
        mock_data = {
            'poles': [
                {'q': 1.0, 'r': [0.1, 0.2]},
                {'q': 2.0, 'r': [0.3, 0.4]}
            ],
            'd': [0.5, 0.6]
        }
        kern = self.sgwt.VFKernel.from_dict(mock_data)
        np.testing.assert_array_equal(kern.Q, [1.0, 2.0])
        np.testing.assert_array_equal(kern.R, [[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_array_equal(kern.D, [0.5, 0.6])

    def test_loading_nonexistent_resource_raises_filenotfounderror(self):
        """Verify loading a non-existent resource raises FileNotFoundError."""
        from sgwt.util import _load_resource
        with pytest.raises(FileNotFoundError):
            _load_resource("library/NON_EXISTENT_FILE.mat", lambda p: p)