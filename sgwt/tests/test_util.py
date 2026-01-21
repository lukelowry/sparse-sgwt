# -*- coding: utf-8 -*-
"""
Tests for utility functions, resource loading, and data integrity.
"""
import numpy as np
import pytest
from ctypes import CDLL

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
        # Kernels should have at least one pole and residue
        assert len(kern.Q) > 0, f"{kernel_name} should have at least one pole"
        assert len(kern.R) > 0, f"{kernel_name} should have at least one residue matrix row"

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
        assert L.format == "csc", f"{laplacian_name} should be CSC format"
        assert L.shape[0] == L.shape[1], f"{laplacian_name} should be square matrix"
        # Laplacians should have at least diagonal entries (n nonzeros minimum)
        min_nnz = L.shape[0]
        assert L.nnz >= min_nnz, \
            f"{laplacian_name} has {L.nnz} nonzeros, expected at least {min_nnz}"


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


class TestMeshSignals:
    """Tests for built-in mesh signals."""

    @pytest.mark.parametrize("signal_name, laplacian_name", [
        ("BUNNY_XYZ", "MESH_BUNNY"),
        ("HORSE_XYZ", "MESH_HORSE"),
    ])
    def test_mesh_signal_properties(self, signal_name, laplacian_name):
        """Mesh signals are (N, 3) arrays and match their Laplacians."""
        S = getattr(sgwt, signal_name)
        L = getattr(sgwt, laplacian_name)

        assert isinstance(S, np.ndarray)
        assert S.ndim == 2, f"{signal_name} should be a 2D array"
        assert S.shape[1] == 3, f"{signal_name} should have 3 columns (X, Y, Z)"
        assert L.shape[0] == S.shape[0], f"Node count for {laplacian_name} and {signal_name} should match"


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

    @pytest.mark.parametrize("order", [0, -5])
    def test_invalid_order_raises_valueerror(self, order):
        """Order < 1 raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Order must be >= 1"):
            sgwt.ChebyKernel.from_function(lambda x: x, order=order, spectrum_bound=1.0)


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
        # Spectral bound should be positive and reasonable (> 0.01 for real graphs)
        min_bound = 0.01
        assert bound > min_bound, \
            f"Expected spectral bound >{min_bound}, got {bound}"

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
        # Should produce at least 1 Chebyshev coefficient
        assert kernel.C.shape[0] > 0, "Kernel should have at least one Chebyshev coefficient"
        # Spectral bound should be positive and reasonable
        assert kernel.spectrum_bound > 0.01, \
            f"Expected reasonable spectral bound, got {kernel.spectrum_bound}"


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


class TestPlyParsing:
    """Tests for PLY file parsing functions."""

    @pytest.fixture
    def ascii_ply_file(self, tmp_path):
        """Create a simple ASCII PLY file for testing."""
        ply_content = """ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
3 0 1 2
3 0 2 3
"""
        ply_path = tmp_path / "test_ascii.ply"
        ply_path.write_text(ply_content)
        return str(ply_path)

    @pytest.fixture
    def binary_ply_file(self, tmp_path):
        """Create a simple binary little-endian PLY file for testing."""
        import struct
        ply_path = tmp_path / "test_binary.ply"

        header = b"""ply
format binary_little_endian 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
"""
        # Vertices: 4 vertices with x, y, z as floats
        vertices = struct.pack('<12f',
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0
        )
        # Faces: 2 triangles
        faces = struct.pack('<B3iB3i',
            3, 0, 1, 2,  # First triangle
            3, 0, 2, 3   # Second triangle
        )

        with open(ply_path, 'wb') as f:
            f.write(header)
            f.write(vertices)
            f.write(faces)

        return str(ply_path)

    @pytest.mark.parametrize("fixture_name", ["ascii_ply_file", "binary_ply_file"])
    def test_parse_ply_formats(self, fixture_name, request):
        """_parse_ply correctly parses both ASCII and binary PLY files."""
        from sgwt.util import _parse_ply
        ply_file = request.getfixturevalue(fixture_name)
        vertices, faces, vertex_count = _parse_ply(ply_file)

        assert vertex_count == 4
        assert len(vertices) == 4
        assert len(faces) == 2
        # Check vertices (use allclose for float precision)
        assert np.allclose(vertices[0], (0.0, 0.0, 0.0))
        assert np.allclose(vertices[1], (1.0, 0.0, 0.0))
        # Check first face
        assert faces[0] == [0, 1, 2]

    def test_parse_ply_unsupported_format_raises(self, tmp_path):
        """_parse_ply raises ValueError for unsupported formats."""
        from sgwt.util import _parse_ply
        ply_content = """ply
format binary_big_endian 1.0
element vertex 1
property float x
property float y
property float z
end_header
"""
        ply_path = tmp_path / "unsupported.ply"
        ply_path.write_text(ply_content)

        with pytest.raises(ValueError, match="Unsupported PLY format"):
            _parse_ply(str(ply_path))

    @pytest.mark.parametrize("fixture_name", ["ascii_ply_file", "binary_ply_file"])
    def test_load_ply_laplacian(self, fixture_name, request):
        """load_ply_laplacian returns valid Laplacian from PLY files."""
        from sgwt.util import load_ply_laplacian
        ply_file = request.getfixturevalue(fixture_name)
        L = load_ply_laplacian(ply_file)

        assert L.format == "csc"
        assert L.shape == (4, 4)  # 4 vertices
        # Laplacian should be symmetric
        assert np.allclose(L.toarray(), L.T.toarray())
        # Diagonal should be positive (vertex degrees)
        assert np.all(L.diagonal() >= 0)

    @pytest.mark.parametrize("fixture_name", ["ascii_ply_file", "binary_ply_file"])
    def test_load_ply_xyz(self, fixture_name, request):
        """load_ply_xyz returns (N, 3) array from PLY files."""
        from sgwt.util import load_ply_xyz
        ply_file = request.getfixturevalue(fixture_name)
        xyz = load_ply_xyz(ply_file)

        assert isinstance(xyz, np.ndarray)
        assert xyz.shape == (4, 3)
        assert np.allclose(xyz[0], [0.0, 0.0, 0.0])
        assert np.allclose(xyz[1], [1.0, 0.0, 0.0])

    def test_laplacian_xyz_consistency(self, ascii_ply_file):
        """Laplacian and XYZ have consistent vertex counts."""
        from sgwt.util import load_ply_laplacian, load_ply_xyz
        L = load_ply_laplacian(ascii_ply_file)
        xyz = load_ply_xyz(ascii_ply_file)

        assert L.shape[0] == xyz.shape[0]

    def test_parse_ply_with_blank_lines_in_header(self, tmp_path):
        """_parse_ply handles blank lines in header."""
        from sgwt.util import _parse_ply
        ply_content = """ply
format ascii 1.0

element vertex 3
property float x
property float y
property float z

element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
3 0 1 2
"""
        ply_path = tmp_path / "blank_lines.ply"
        ply_path.write_text(ply_content)

        vertices, faces, vertex_count = _parse_ply(str(ply_path))
        assert vertex_count == 3
        assert len(vertices) == 3
        assert len(faces) == 1

    def test_parse_ply_binary_non_xyz_property_names(self, tmp_path):
        """_parse_ply handles binary PLY with non-standard property names."""
        import struct
        ply_path = tmp_path / "non_xyz.ply"

        # Use property names that are NOT x, y, z
        header = b"""ply
format binary_little_endian 1.0
element vertex 3
property float px
property float py
property float pz
element face 1
property list uchar int vertex_indices
end_header
"""
        vertices = struct.pack('<9f',
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.5, 1.0, 0.0
        )
        faces = struct.pack('<B3i', 3, 0, 1, 2)

        with open(ply_path, 'wb') as f:
            f.write(header)
            f.write(vertices)
            f.write(faces)

        from sgwt.util import _parse_ply
        verts, _, count = _parse_ply(str(ply_path))

        assert count == 3
        assert len(verts) == 3
        # Should use first 3 properties (px, py, pz) as coordinates
        assert np.allclose(verts[0], (0.0, 0.0, 0.0))
        assert np.allclose(verts[1], (1.0, 0.0, 0.0))
        assert np.allclose(verts[2], (0.5, 1.0, 0.0))

    def test_parse_ply_with_extra_elements(self, tmp_path):
        """_parse_ply handles PLY files with extra element types (e.g., edge)."""
        from sgwt.util import _parse_ply
        # PLY file with face element followed by additional properties/elements
        ply_content = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
element edge 2
property int vertex1
property int vertex2
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
3 0 1 2
0 1
1 2
"""
        ply_path = tmp_path / "extra_elements.ply"
        ply_path.write_text(ply_content)

        vertices, faces, vertex_count = _parse_ply(str(ply_path))
        assert vertex_count == 3
        assert len(vertices) == 3
        assert len(faces) == 1


class TestMeshLaplacians:
    """Tests for built-in mesh Laplacians loaded from PLY files."""

    @pytest.mark.parametrize("laplacian_name", [
        "MESH_BUNNY", "MESH_HORSE", "MESH_LBRAIN"
    ])
    def test_mesh_laplacian_is_valid(self, laplacian_name):
        """Built-in mesh Laplacians are valid symmetric CSC matrices."""
        L = getattr(sgwt, laplacian_name)

        assert L.format == "csc", f"{laplacian_name} should be CSC format"
        assert L.shape[0] == L.shape[1], f"{laplacian_name} should be square"


class TestModuleGetattr:
    """Tests for module __getattr__ function."""

    def test_invalid_attribute_raises(self):
        """Accessing non-existent attribute raises AttributeError."""
        import sgwt.util
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = sgwt.util.NONEXISTENT_RESOURCE_NAME


class TestChebyKernelCoverage:
    """Additional tests for ChebyKernel edge cases."""

    def test_from_function_all_negligible_coefficients(self):
        """Fitting a function where all higher-order coefficients are negligible."""
        # A constant function should result in only the constant term being kept
        kern = sgwt.ChebyKernel.from_function(
            lambda x: np.full_like(x, 1e-20),  # Nearly zero constant
            order=10,
            spectrum_bound=1.0
        )
        # Should keep at least the constant term
        assert kern.C.shape[0] >= 1

    @pytest.mark.parametrize("sampling", ['linear', 'quadratic', 'logarithmic'])
    def test_from_function_sampling_strategies(self, sampling):
        """Test from_function with various sampling strategies."""
        kern = sgwt.ChebyKernel.from_function(
            lambda x: np.exp(-x),
            order=5,
            spectrum_bound=2.0,
            sampling=sampling
        )
        assert kern.C.shape[0] > 0
        # Verify the kernel approximates reasonably well
        x_test = np.linspace(0, 2.0, 20)
        result = kern.evaluate(x_test)
        expected = np.exp(-x_test)
        np.testing.assert_allclose(result.flatten(), expected, atol=0.1)

    def test_from_function_adaptive_fitting(self):
        """Test from_function with adaptive order selection."""
        kern = sgwt.ChebyKernel.from_function(
            lambda x: np.exp(-x),
            order=5,  # Starting order
            spectrum_bound=2.0,
            adaptive=True,
            target_error=0.01,
            max_order=50
        )
        assert kern.C.shape[0] > 0
        # Adaptive fitting should find appropriate order
        x_test = np.linspace(0, 2.0, 100)
        result = kern.evaluate(x_test)
        expected = np.exp(-x_test)
        # Should achieve target error approximately
        rel_error = np.max(np.abs(result.flatten() - expected) / np.maximum(np.abs(expected), 1e-15))
        assert rel_error < 0.1  # Allow some slack in convergence

    def test_from_function_adaptive_reaches_max_order(self):
        """Test adaptive fitting that hits max_order."""
        # Use a simple function but with impossibly tight target
        kern = sgwt.ChebyKernel.from_function(
            lambda x: np.exp(-x),
            order=5,
            spectrum_bound=2.0,
            adaptive=True,
            target_error=1e-20,  # Impossibly tight - will hit max_order
            max_order=10  # Very low max to finish quickly
        )
        # Should still produce a valid kernel even if target not met
        assert kern.C.shape[0] > 0


class TestListGraphs:
    """Tests for list_graphs utility function."""

    def test_list_graphs_prints_table(self, capsys):
        """list_graphs prints a table of available graphs including DELAY graphs."""
        from sgwt.util import list_graphs
        list_graphs()

        captured = capsys.readouterr()
        # Should print header
        assert "Graph Name" in captured.out
        assert "Vertices" in captured.out
        assert "Edges" in captured.out
        # Should include known graphs
        assert "DELAY_TEXAS" in captured.out

    def test_list_graphs_with_no_graphs(self, capsys):
        """list_graphs handles case with no graphs gracefully."""
        from unittest.mock import patch
        from sgwt.util import list_graphs

        # Mock an empty registry
        with patch('sgwt.util._ensure_registry', return_value={}):
            list_graphs()

        captured = capsys.readouterr()
        assert "No graphs found" in captured.out

    def test_list_graphs_with_non_sparse_entry(self, capsys):
        """list_graphs skips entries that aren't sparse matrices."""
        from unittest.mock import patch
        from sgwt.util import list_graphs

        # Mock registry with a non-sparse matrix entry
        mock_registry = {
            'DELAY_FAKE': lambda: np.array([1, 2, 3])  # Not a sparse matrix
        }
        with patch('sgwt.util._ensure_registry', return_value=mock_registry):
            list_graphs()

        captured = capsys.readouterr()
        # Should print header but not the fake entry (no nnz)
        assert "Graph Name" in captured.out

    def test_list_graphs_with_loader_exception(self, capsys):
        """list_graphs handles exceptions from loaders gracefully."""
        from unittest.mock import patch
        from sgwt.util import list_graphs

        def raise_error():
            raise RuntimeError("Failed to load")

        mock_registry = {'DELAY_BAD': raise_error}
        with patch('sgwt.util._ensure_registry', return_value=mock_registry):
            list_graphs()

        captured = capsys.readouterr()
        # Should still print header, skip bad entry
        assert "Graph Name" in captured.out

    def test_list_graphs_fallback_edge_calculation(self, capsys):
        """list_graphs uses fallback edge calculation when no diagonal method."""
        from unittest.mock import patch, MagicMock
        from sgwt.util import list_graphs

        # Create mock sparse matrix without diagonal method
        mock_matrix = MagicMock()
        mock_matrix.shape = (10, 10)
        mock_matrix.nnz = 30
        del mock_matrix.diagonal  # Remove diagonal method

        mock_registry = {'DELAY_NODIAG': lambda: mock_matrix}
        with patch('sgwt.util._ensure_registry', return_value=mock_registry):
            list_graphs()

        captured = capsys.readouterr()
        assert "DELAY_NODIAG" in captured.out
        # Fallback: (30 - 10) // 2 = 10 edges
        assert "10" in captured.out


class TestDiscoverResourcesEdgeCases:
    """Tests for resource discovery edge cases."""

    def test_discover_resources_handles_typeerror(self):
        """_discover_resources handles TypeError on iterdir gracefully."""
        from unittest.mock import patch, MagicMock
        from sgwt.util import _discover_resources

        # Create a mock folder that raises TypeError on iterdir
        mock_folder = MagicMock()
        mock_folder.iterdir.side_effect = TypeError("not iterable")

        # Mock the path chain: files("sgwt") / "library" / cfg.folder
        mock_library = MagicMock()
        mock_library.__truediv__ = MagicMock(return_value=mock_folder)

        mock_sgwt = MagicMock()
        mock_sgwt.__truediv__ = MagicMock(return_value=mock_library)

        with patch('sgwt.util.files', return_value=mock_sgwt):
            registry = _discover_resources()
            assert isinstance(registry, dict)

    def test_discover_resources_handles_filenotfounderror(self):
        """_discover_resources handles FileNotFoundError on iterdir gracefully."""
        from unittest.mock import patch, MagicMock
        from sgwt.util import _discover_resources

        # Create a mock folder that raises FileNotFoundError on iterdir
        mock_folder = MagicMock()
        mock_folder.iterdir.side_effect = FileNotFoundError("folder not found")

        mock_library = MagicMock()
        mock_library.__truediv__ = MagicMock(return_value=mock_folder)

        mock_sgwt = MagicMock()
        mock_sgwt.__truediv__ = MagicMock(return_value=mock_library)

        with patch('sgwt.util.files', return_value=mock_sgwt):
            registry = _discover_resources()
            assert isinstance(registry, dict)

    def test_discover_resources_skips_wrong_extension(self):
        """_discover_resources skips files with wrong extension."""
        from unittest.mock import patch, MagicMock
        from sgwt.util import _discover_resources

        # Create mock file with wrong extension
        mock_file = MagicMock()
        mock_file.name = "test.txt"  # Wrong extension for any config

        mock_folder = MagicMock()
        mock_folder.iterdir.return_value = [mock_file]

        mock_library = MagicMock()
        mock_library.__truediv__ = MagicMock(return_value=mock_folder)

        mock_sgwt = MagicMock()
        mock_sgwt.__truediv__ = MagicMock(return_value=mock_library)

        with patch('sgwt.util.files', return_value=mock_sgwt):
            registry = _discover_resources()
            # Should return empty dict since no files match expected extensions
            assert isinstance(registry, dict)
  