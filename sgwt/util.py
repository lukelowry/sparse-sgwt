"""General Utilities

Description: Utilities for accessing built-in data, VFKern, and impulse helper function.

Author: Luke Lowery (lukel@tamu.edu)
"""

import sys
import os

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:  # pragma: no cover
    from importlib_resources import as_file, files

from ctypes import CDLL
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, linalg

from json import load as jsonload
from typing import Any, Callable, Dict, List, Union, Optional, Tuple


@dataclass
class _FolderConfig:
    """Configuration for auto-discovering resources in a library folder."""
    folder: str
    extension: str
    key_fn: Callable[[str], str]
    loader_fn: Callable[[str], Callable[[], Any]]
    secondary_fn: Optional[Callable[[str], Tuple[str, Callable[[], Any]]]] = None


@dataclass
class ChebyKernel:
    """Stores Chebyshev polynomial approximations for one or more kernels."""
    C: np.ndarray
    """Coefficient matrix of shape (order + 1, n_dims)."""
    spectrum_bound: float
    """Shared upper spectrum bound for all kernels."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChebyKernel':
        """Loads kernel data from a dictionary."""
        approxs = data.get('approximations', [])
        bound = data.get('spectrum_bound', 0.0)
        if not approxs:
            return cls(C=np.empty((0, 0)), spectrum_bound=bound)
        coeffs = [np.asarray(a.get('coeffs', [])) for a in approxs]
        if any(len(c) != len(coeffs[0]) for c in coeffs):
            raise ValueError("All 'coeffs' arrays must have the same length.")
        return cls(C=np.stack(coeffs, axis=1), spectrum_bound=bound)
    
    @classmethod
    def from_function(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        order: int,
        spectrum_bound: float,
        n_samples: int = None,
        sampling: str = 'chebyshev',
        min_lambda: float = 0.0,
        rtol: float = 1e-12,
        adaptive: bool = False,
        max_order: int = 500,
        target_error: float = 1e-10
    ) -> 'ChebyKernel':
        """Creates a ChebyKernel by fitting a vectorized function.
        
        Parameters
        ----------
        f : Callable[[np.ndarray], np.ndarray]
            The vectorized function to approximate.
        order : int
            Order of the Chebyshev polynomial to fit.
        spectrum_bound : float
            Upper bound of the function's domain.
        n_samples : int, optional
            Number of sample points (only used for non-Chebyshev sampling).
        sampling : str, default 'chebyshev'
            Sampling strategy: 'chebyshev' (optimal), 'linear', 'quadratic', 'logarithmic'.
        min_lambda : float, default 0.0
            Lower bound of the sampling range.
        rtol : float, default 1e-12
            Relative tolerance for truncating negligible coefficients.
        adaptive : bool, default False
            If True, automatically determines optimal order to achieve target_error.
        max_order : int, default 500
            Maximum order for adaptive mode.
        target_error : float, default 1e-10
            Target approximation error for adaptive mode.
        """
        if order < 1:
            raise ValueError("Order must be >= 1")
        
        if adaptive:
            return cls._adaptive_fit(f, order, spectrum_bound, min_lambda, 
                                     sampling, rtol, max_order, target_error)
        
        coeffs = cls._compute_coefficients(f, order, spectrum_bound, min_lambda, 
                                           n_samples, sampling)
        coeffs = cls._ensure_2d(coeffs)
        coeffs = cls._truncate(coeffs, rtol)
        
        return cls(C=coeffs, spectrum_bound=spectrum_bound)
    
    @classmethod
    def _compute_coefficients(cls, f, order, spectrum_bound, min_lambda, n_samples, sampling):
        """Compute Chebyshev coefficients using optimal or fallback method."""
        lambda_range = spectrum_bound - min_lambda
        lambda_mid = (spectrum_bound + min_lambda) / 2.0
        
        if sampling == 'chebyshev':
            # Chebyshev-Gauss-Lobatto nodes: optimal for polynomial interpolation
            n = order + 1
            k = np.arange(n)
            x_cheb = np.cos(np.pi * k / order)  # Nodes in [-1, 1]
            sample_x = lambda_mid + (lambda_range / 2.0) * x_cheb
            f_values = cls._ensure_2d(f(sample_x))
            
            # Compute coefficients via discrete orthogonality (DCT-like)
            coeffs = np.zeros((n, f_values.shape[1]))
            w = np.ones(n); w[0] = w[-1] = 0.5  # Endpoint weights
            
            for j in range(n):
                T_j = np.cos(j * np.pi * k / order)
                scale = 2.0 / order if 0 < j < order else 1.0 / order
                coeffs[j] = scale * np.sum(w[:, None] * f_values * T_j[:, None], axis=0)
            
            return coeffs
        
        # Fallback: least-squares fitting for other sampling strategies
        n_samples = n_samples or max(4 * (order + 1), 1000)
        t = np.linspace(0, 1, n_samples)
        
        if sampling == 'quadratic':
            sample_x = min_lambda + lambda_range * (t ** 2)
        elif sampling == 'logarithmic':
            eps = max(min_lambda * 0.001, 1e-10)
            sample_x = np.exp(np.log(min_lambda + eps) + t * np.log(spectrum_bound / (min_lambda + eps)))
        else:  # linear
            sample_x = min_lambda + lambda_range * t
        
        x_scaled = 2.0 * (sample_x - min_lambda) / lambda_range - 1.0
        
        # Chebyshev-weighted least squares
        weights = 1.0 / np.sqrt(1.0 - np.clip(x_scaled ** 2, 0, 0.9999))
        return np.polynomial.chebyshev.chebfit(x_scaled, f(sample_x), order, w=weights)
    
    @classmethod
    def _adaptive_fit(cls, f, start_order, spectrum_bound, min_lambda,
                      sampling, rtol, max_order, target_error):
        """Adaptively determine optimal polynomial order."""
        test_x = np.linspace(min_lambda, spectrum_bound, 1000)
        f_exact = cls._ensure_2d(f(test_x))
        order = max(start_order, 8)

        while True:
            coeffs = cls._ensure_2d(
                cls._compute_coefficients(f, order, spectrum_bound, min_lambda, None, sampling)
            )

            # Evaluate and compute relative error
            x_scaled = 2.0 * test_x / spectrum_bound - 1.0
            f_approx = np.polynomial.chebyshev.chebval(x_scaled, coeffs)
            f_approx = f_approx.T if f_approx.ndim > 1 else f_approx[:, None]

            rel_error = np.max(np.abs(f_exact - f_approx) / np.maximum(np.abs(f_exact), 1e-15))

            if rel_error <= target_error or order >= max_order:
                break
            order = min(int(order * 1.5) + 1, max_order)

        return cls(C=cls._truncate(coeffs, rtol), spectrum_bound=spectrum_bound)
    
    @classmethod
    def _ensure_2d(cls, arr):
        """Ensure array is 2D with shape (n, dims)."""
        arr = np.atleast_1d(arr)
        return arr[:, None] if arr.ndim == 1 else arr
    
    @classmethod
    def _truncate(cls, coeffs, rtol):
        """Truncate negligible higher-order coefficients."""
        threshold = rtol * np.max(np.abs(coeffs))
        row_max = np.max(np.abs(coeffs), axis=1)
        significant = np.where(row_max > threshold)[0]
        last_idx = significant[-1] + 1 if significant.size > 0 else 1
        return coeffs[:last_idx]
    
    @classmethod
    def from_function_on_graph(cls, L: csc_matrix, f: Callable[[np.ndarray], np.ndarray], 
                               order: int, **kwargs) -> 'ChebyKernel':
        """Creates a ChebyKernel fitted to a graph's spectrum."""
        return cls.from_function(f, order, estimate_spectral_bound(L), **kwargs)
    
    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        """Maps points from [0, spectrum_bound] to Chebyshev domain [-1, 1]."""
        return (2.0 / self.spectrum_bound) * x - 1.0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the Chebyshev approximation at given points."""
        if self.C.size == 0:
            return np.empty((len(x), 0))
        y = np.polynomial.chebyshev.chebval(self._scale_x(x), self.C)
        return y.T if y.ndim > 1 else y
    
def estimate_spectral_bound(L: csc_matrix) -> float:
    """
    Estimates the largest eigenvalue (spectral bound) of a matrix.

    This is typically used to find the domain [0, lambda_max] for Chebyshev
    polynomial approximations.

    Parameters
    ----------
    L : csc_matrix
        The matrix (e.g., Graph Laplacian) for which to estimate the bound.

    Returns
    -------
    float
        An estimate of the largest eigenvalue, scaled by 1.01 for safety.
    """
    # Note: Using eigs from scipy.sparse.linalg
    e_max = linalg.eigs(L, k=1, which='LM', return_eigenvectors=False)
    return float(e_max[0].real) * 1.01


@dataclass
class VFKernel:
    """Vector Fitting Kernel representation.

    A dataclass to store the components of a rational kernel approximation
    obtained from Vector Fitting.
    """

    R: np.ndarray
    """Residue matrix of shape (n_poles, n_dims)."""

    Q: np.ndarray
    """Poles vector of shape (n_poles,)."""

    D: np.ndarray
    """Direct term (offset) of shape (n_dims,)."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VFKernel':
        """Loads kernel data from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the kernel parameters, typically loaded
            from a JSON file. It should have 'poles' and 'd' keys.

        Returns
        -------
        VFKernel
            A new instance of the VFKernel class.
        """
        poles = data.get('poles', [])
        return cls(
            R=np.array([p.get('r', []) for p in poles]),
            Q=np.array([p.get('q', 0) for p in poles]),
            D=np.array(data.get('d', []))
        )


def impulse(lap: csc_matrix, n: int = 0, n_timesteps: int = 1) -> np.ndarray:
    """
    Generates a Dirac impulse signal at a specified vertex.

    Parameters
    ----------
    lap : csc_matrix
        Graph Laplacian defining the number of vertices.
    n : int
        Index of the vertex where the impulse is applied.
    n_timesteps : int
        Number of time steps (columns) in the resulting signal.

    Returns
    -------
    np.ndarray
        (n_vertices, n_timesteps) array with 1.0 at index n and 0.0 elsewhere, in Fortran order.
    """
    b: np.ndarray = np.zeros((lap.shape[0], n_timesteps), order='F')
    b[n] = 1

    return b

def _load_dll(dll_name: str) -> CDLL:
    """Locates and loads a shared library from the library/dll directory.
    
    Handles platform-specific path adjustments to ensure the DLL can be found
    and loaded by ctypes.
    
    Raises
    ------
    OSError
        If the DLL file cannot be loaded.
    Exception
        For other unexpected errors during loading.
    
    Returns
    -------
    ctypes.CDLL
        The loaded DLL object.
    """
    resource = files("sgwt") / "library" / "dll" / dll_name
    with as_file(resource) as dll_path:
        dll_dir = os.path.dirname(dll_path)
        # On Windows, add the DLL's directory to the search path for dependencies
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        else:  # pragma: no cover
            os.environ['PATH'] = str(dll_dir) + os.pathsep + os.environ['PATH']
        try:
            return CDLL(str(dll_path))
        except OSError as e:
            raise OSError(f"Failed to load DLL at {dll_path}. Error: {e}")
        except Exception as e:  # pragma: no cover
            raise Exception(f"Unexpected error loading DLL: {e}")
            
def get_cholmod_dll() -> CDLL:
    """Locates and loads the CHOLMOD shared library."""
    return _load_dll("cholmod.dll")

def get_klu_dll() -> CDLL:
    """Locates and loads the KLU shared library."""
    return _load_dll("klu.dll")

def _load_resource(path: str, loader: Callable[[str], Any]) -> Any:
    """Centralized resource loader using importlib.resources."""
    with as_file(files("sgwt").joinpath(path)) as file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resource not found: {file_path}")
        return loader(str(file_path))


def _mat_loader(path: str, to_csc: bool = False) -> Union[np.ndarray, csc_matrix]:
    """
    Loads data from a .mat file.
    
    If a single variable is present, it is returned. If multiple variables
    are found, they are flattened and stacked into columns of a single array.
    """
    data = loadmat(path, squeeze_me=False)
    keys = [k for k in data if not k.startswith("__")]
    
    if not keys:
        raise ValueError(f"No data variables found in MAT file: {path}")

    res = data[keys[0]]
    if to_csc:
        # Data may already be sparse from loadmat; use hasattr to avoid
        # pytest-cov instrumentation issues with scipy.sparse.issparse()
        if hasattr(res, "tocsc"):
            return res.tocsc()
        return csc_matrix(res)

    if len(keys) > 1:
        return np.stack([data[k].flatten() for k in keys], axis=1)

    return res.T if (res.ndim == 2 and res.shape[0] == 1) else res

def _json_kern_loader(path: str) -> Dict[str, Any]:
    """Loads a VFKern from a JSON file."""
    with open(path, "r") as f:
        return jsonload(f)

def _parse_ply(filepath: str) -> tuple:
    """
    Parses a .ply mesh file and returns vertices and faces.

    Parameters
    ----------
    filepath : str
        Path to the .ply file.

    Returns
    -------
    tuple
        (vertices, faces, vertex_count) where vertices is a list of (x,y,z) tuples,
        faces is a list of vertex index lists, and vertex_count is the number of vertices.
    """
    import struct

    with open(filepath, 'rb') as f:
        # Parse header
        fmt = "ascii"
        vertex_count = 0
        face_count = 0
        vertex_props = []
        current_element = None

        while True:
            line = f.readline().strip().decode('ascii', errors='ignore')
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    face_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                vertex_props.append((parts[2], parts[1]))

        vertices = []
        faces = []

        if fmt == "ascii":
            lines = f.readlines()
            for i in range(vertex_count):
                parts = lines[i].strip().split()
                vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
            for i in range(face_count):
                parts = lines[vertex_count + i].strip().split()
                faces.append([int(x) for x in parts[1:]])

        elif fmt == "binary_little_endian":
            np_type_map = {
                'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
                'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8'
            }
            dtype_fields = [(name, np_type_map.get(t, 'f4')) for name, t in vertex_props]
            vertex_dtype = np.dtype(dtype_fields)

            vertex_data = f.read(vertex_count * vertex_dtype.itemsize)
            v_arr = np.frombuffer(vertex_data, dtype=vertex_dtype)

            if 'x' in v_arr.dtype.names and 'y' in v_arr.dtype.names and 'z' in v_arr.dtype.names:
                vertices = list(zip(v_arr['x'], v_arr['y'], v_arr['z']))
            else:
                names = v_arr.dtype.names
                vertices = list(zip(v_arr[names[0]], v_arr[names[1]], v_arr[names[2]]))

            for _ in range(face_count):
                n = struct.unpack('<B', f.read(1))[0]
                faces.append(list(struct.unpack(f'<{n}i', f.read(n * 4))))
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    return vertices, faces, vertex_count


def load_ply_laplacian(filepath: str) -> csc_matrix:
    """
    Loads a .ply mesh file and returns its graph Laplacian.

    This is a convenience function for loading mesh data directly into
    the sparse format required by the convolution solvers.

    Parameters
    ----------
    filepath : str
        Path to the .ply file.

    Returns
    -------
    csc_matrix
        The graph Laplacian matrix L = B @ B.T where B is the incidence matrix.

    Examples
    --------
    >>> from sgwt import load_ply_laplacian, Convolve
    >>> L = load_ply_laplacian("my_mesh.ply")
    >>> with Convolve(L) as conv:
    ...     coeffs = conv.lowpass(signal, scales=[1.0])
    """
    vertices, faces, vertex_count = _parse_ply(filepath)

    # Extract unique edges from faces
    unique_edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            u, v = face[i], face[(i + 1) % n]
            unique_edges.add((u, v) if u < v else (v, u))

    edges = np.array(sorted(unique_edges), dtype=int)
    num_edges = len(edges)

    # Build incidence matrix B and compute Laplacian L = B @ B.T
    rows = edges.ravel()
    cols = np.repeat(np.arange(num_edges), 2)
    data = np.tile([1.0, -1.0], num_edges)

    B = csc_matrix((data, (rows, cols)), shape=(vertex_count, num_edges))
    return B @ B.T


def load_ply_xyz(filepath: str) -> np.ndarray:
    """
    Loads a .ply mesh file and returns the vertex coordinates.

    Parameters
    ----------
    filepath : str
        Path to the .ply file.

    Returns
    -------
    np.ndarray
        An (N, 3) array of vertex coordinates (x, y, z).

    Examples
    --------
    >>> from sgwt import load_ply_xyz
    >>> xyz = load_ply_xyz("my_mesh.ply")
    >>> print(xyz.shape)  # (num_vertices, 3)
    """
    vertices, _, _ = _parse_ply(filepath)
    return np.array(vertices)


# Factory helpers
def _lap(k: str, r: str) -> csc_matrix:
    """Loads a Laplacian from library/{TYPE}/{REGION}.mat."""
    return _load_resource(f"library/{k}/{r}.mat", lambda p: _mat_loader(p, to_csc=True))  # type: ignore


def _sig(r: str) -> np.ndarray:
    """Loads a signal from library/SIGNALS/{REGION}.mat."""
    return _load_resource(f"library/SIGNALS/{r}.mat", _mat_loader)  # type: ignore


def _kern(n: str) -> Dict[str, Any]:
    """Loads a kernel from library/KERNELS/{NAME}.json."""
    return _load_resource(f"library/KERNELS/{n}.json", _json_kern_loader)


def _ply_lap(n: str) -> csc_matrix:
    """Loads a mesh Laplacian from library/MESH/{NAME}.ply."""
    return _load_resource(f"library/MESH/{n}.ply", load_ply_laplacian)  # type: ignore


def _ply_xyz(n: str) -> np.ndarray:
    """Loads mesh coordinates from library/MESH/{NAME}.ply."""
    return _load_resource(f"library/MESH/{n}.ply", load_ply_xyz)  # type: ignore


# Auto-discovery configuration for library folders
_FOLDER_CONFIGS: List[_FolderConfig] = [
    _FolderConfig("KERNELS", ".json",
                  lambda s: s,
                  lambda s: lambda: _kern(s)),
    _FolderConfig("DELAY", ".mat",
                  lambda s: f"DELAY_{s}",
                  lambda s: lambda: _lap("DELAY", s)),
    _FolderConfig("IMPEDANCE", ".mat",
                  lambda s: f"IMPEDANCE_{s}",
                  lambda s: lambda: _lap("IMPEDANCE", s)),
    _FolderConfig("LENGTH", ".mat",
                  lambda s: f"LENGTH_{s}",
                  lambda s: lambda: _lap("LENGTH", s)),
    _FolderConfig("SIGNALS", ".mat",
                  lambda s: f"COORD_{s}",
                  lambda s: lambda: _sig(s)),
    _FolderConfig("MESH", ".ply",
                  lambda s: f"MESH_{s}",
                  lambda s: lambda: _ply_lap(s),
                  lambda s: (f"{s}_XYZ", lambda: _ply_xyz(s))),
]

_LAZY_REGISTRY: Optional[Dict[str, Callable[[], Any]]] = None


def _discover_resources() -> Dict[str, Callable[[], Any]]:
    """Scans library folders and builds the lazy-loading registry."""
    registry: Dict[str, Callable[[], Any]] = {}
    library = files("sgwt") / "library"

    for cfg in _FOLDER_CONFIGS:
        folder = library / cfg.folder
        try:
            items = list(folder.iterdir())
        except (TypeError, FileNotFoundError):
            continue

        for item in items:
            if not item.name.endswith(cfg.extension):
                continue
            stem = item.name[:-len(cfg.extension)]

            # Add primary entry
            registry[cfg.key_fn(stem)] = cfg.loader_fn(stem)

            # Add secondary entry if configured (e.g., MESH -> XYZ)
            if cfg.secondary_fn:
                key, loader = cfg.secondary_fn(stem)
                registry[key] = loader

    return registry


def _ensure_registry() -> Dict[str, Callable[[], Any]]:
    """Returns the registry, building it on first access."""
    global _LAZY_REGISTRY
    if _LAZY_REGISTRY is None:
        _LAZY_REGISTRY = _discover_resources()
    return _LAZY_REGISTRY


def list_graphs() -> None:
    """
    Prints a table of all available graph Laplacians in the library.
    
    Displays the graph name, number of vertices, and number of edges.
    """
    registry = _ensure_registry()
    
    # Filter for graph keys based on known prefixes
    graph_prefixes = ("DELAY_", "IMPEDANCE_", "LENGTH_", "MESH_")
    graph_keys = [k for k in registry.keys() if k.startswith(graph_prefixes)]
    
    if not graph_keys:
        print("No graphs found in the library.")
        return

    # Header
    print(f"{'Graph Name':<30} {'Vertices':<10} {'Edges':<10}")
    print("-" * 52)
    
    for key in sorted(graph_keys):
        try:
            # Load the graph
            L = registry[key]()
            
            # Check if it looks like a sparse matrix
            if hasattr(L, 'shape') and hasattr(L, 'nnz'):
                n_vertices = L.shape[0]
                
                # Calculate edges: (nnz - non_zero_diagonal_elements) / 2
                # This handles cases with isolated nodes (0 on diagonal)
                if hasattr(L, 'diagonal'):
                    diag_nnz = np.count_nonzero(L.diagonal())
                    n_edges = (L.nnz - diag_nnz) // 2
                else:
                    # Fallback
                    n_edges = (L.nnz - n_vertices) // 2

                print(f"{key:<30} {n_vertices:<10} {n_edges:<10}")
        except Exception:
            continue


def __getattr__(name: str) -> Any:
    registry = _ensure_registry()
    if name in registry:
        return registry[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return list(globals().keys()) + list(_ensure_registry().keys())


__all__ = ["ChebyKernel", "VFKernel", "impulse", "get_cholmod_dll", "get_klu_dll", "estimate_spectral_bound", "load_ply_laplacian", "load_ply_xyz", "list_graphs"]