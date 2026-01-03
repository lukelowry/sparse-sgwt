# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/test_util.py
Description: Tests for general utilities, resource loading, and data integrity.
"""
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from ctypes import CDLL

class TestSGWTUtil(unittest.TestCase):

    def setUp(self):
        import sgwt
        self.sgwt = sgwt

    def test_dll_loading(self):
        """Verify that the CHOLMOD DLL can be located and loaded."""
        try:
            dll = self.sgwt.get_cholmod_dll()
            self.assertIsInstance(dll, CDLL)
        except OSError as e:
            self.fail(f"DLL Load Error (OSError): {e}")
        except Exception as e:
            self.fail(f"get_cholmod_dll raised Exception: {e}")

    def test_kernel_loading(self):
        """Verify built-in kernels are loaded as VFKernel objects with valid data."""
        kernels = [self.sgwt.MEXICAN_HAT, self.sgwt.MODIFIED_MORLET, self.sgwt.SHANNON]
        for kjson in kernels:
            k = self.sgwt.VFKernel.from_dict(kjson)
            self.assertIsInstance(k, self.sgwt.VFKernel)
            self.assertGreater(len(k.Q), 0, "Kernel poles (Q) should not be empty")
            self.assertGreater(len(k.R), 0, "Kernel residues (R) should not be empty")

    def test_laplacian_loading(self):
        """Verify built-in Laplacians are loaded as csc_matrix and are square."""
        # Test a representative subset of Laplacians
        laps = [self.sgwt.DELAY_TEXAS, self.sgwt.IMPEDANCE_HAWAII, self.sgwt.LENGTH_WECC]
        for L in laps:
            self.assertIsInstance(L, csc_matrix)
            self.assertEqual(L.shape[0], L.shape[1], "Laplacian must be square")
            self.assertGreater(L.nnz, 0, "Laplacian should have non-zero entries")

    def test_laplacian_symmetry(self):
        """Verify that built-in Laplacians are symmetric."""
        L = self.sgwt.DELAY_TEXAS
        # Check symmetry: L - L.T should be zero
        diff = (L - L.T).tocsr()
        self.assertEqual(diff.nnz, 0, "Laplacian should be symmetric")

    def test_signal_loading(self):
        """Verify built-in signals are loaded with correct type and dimensions."""
        # Test a representative subset of signals
        signals = [self.sgwt.COORD_TEXAS, self.sgwt.COORD_USA]
        for S in signals:
            self.assertIsInstance(S, np.ndarray)
            self.assertEqual(S.ndim, 2, "Coordinate signals should be 2D (N x Dim)")
            self.assertIn(S.shape[1], [2, 3], "Coordinates should typically be 2D or 3D")

    def test_data_integrity_alignment(self):
        """Ensure Laplacians and associated signals have matching node counts."""
        self.assertEqual(self.sgwt.DELAY_TEXAS.shape[0], self.sgwt.COORD_TEXAS.shape[0])
        self.assertEqual(self.sgwt.DELAY_USA.shape[0], self.sgwt.COORD_USA.shape[0])

    def test_vfkernel_from_dict_logic(self):
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

    def test_resource_not_found(self):
        """Loading a non-existent resource raises FileNotFoundError."""
        from sgwt.util import _load_resource
        with self.assertRaises(FileNotFoundError):
            _load_resource("library/NON_EXISTENT_FILE.mat", lambda p: p)

if __name__ == '__main__':
    unittest.main()