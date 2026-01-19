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
from typing import Any, Callable, Dict, List, Union, Optional


@dataclass
class ChebyKernel:
    """Stores Chebyshev polynomial approximations for one or more kernels."""

    C: np.ndarray
    """Coefficient matrix of shape (order + 1, n_dims)."""

    spectrum_bound: float
    """Shared upper spectrum bound for all kernels."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChebyKernel':
        """Loads kernel data from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with a "spectrum_bound" key and an "approximations"
            key containing a list of `{"coeffs": [...]}` objects.

        Returns
        -------
        ChebyKernel
            A new instance of the ChebyKernel class.
        """
        approxs = data.get('approximations', [])
        bound = data.get('spectrum_bound', 0.0)

        if not approxs:
            return cls(C=np.empty((0, 0)), spectrum_bound=bound)

        coeffs = [np.asarray(a.get('coeffs', [])) for a in approxs]
        if any(len(c) != len(coeffs[0]) for c in coeffs):
            raise ValueError("All 'coeffs' arrays must have the same length.")

        return cls(C=np.stack(coeffs, axis=1), spectrum_bound=bound)

    @classmethod
    def from_function(cls, f: Callable[[np.ndarray], np.ndarray], order: int, spectrum_bound: float, n_samples: int = 10000, sampling: str = 'quadratic', min_lambda: float = 0.0) -> 'ChebyKernel':
        """Creates a ChebyKernel by fitting a vectorized function.

        Parameters
        ----------
        f : Callable[[np.ndarray], np.ndarray]
            The vectorized function to approximate.
        order : int
            Order of the Chebyshev polynomial to fit.
        spectrum_bound : float
            Upper bound of the function's domain.
        n_samples : int, default 10000
            Number of points to sample.
        sampling : str, default 'quadratic'
            Sampling strategy: 'linear' or 'quadratic'. Quadratic sampling
            (t^2) clusters points near 0 to better capture sharp filter features.
        min_lambda : float, default 0.0
            The lower bound of the sampling range.

        Returns
        -------
        ChebyKernel
            A new instance of the ChebyKernel class with the fitted coefficients.
        """
        if order < 1:
            raise ValueError("Order must be >= 1")

        t = np.linspace(0, 1, n_samples)
        sample_x = min_lambda + (spectrum_bound - min_lambda) * (t**2 if sampling == 'quadratic' else t)

        f_values = f(sample_x)
        x_for_fit = (2.0 / spectrum_bound) * sample_x - 1.0  # Map [0, bound] to [-1, 1]
        coeffs = np.polynomial.chebyshev.chebfit(x_for_fit, f_values, order)

        if coeffs.ndim == 1:
            coeffs = coeffs[:, np.newaxis]

        # Truncate negligible higher-order coefficients to optimize convolution
        # Find the highest degree that has a non-negligible coefficient in any dimension
        abs_coeffs = np.abs(coeffs)
        row_max = np.max(abs_coeffs, axis=1)
        nonzero_indices = np.where(row_max > 1e-15)[0]
        
        if nonzero_indices.size > 0:
            coeffs = coeffs[:np.max(nonzero_indices) + 1, :]
        else:
            coeffs = coeffs[:1, :] # Keep at least the constant term

        return cls(C=coeffs, spectrum_bound=spectrum_bound)

    @classmethod
    def from_function_on_graph(cls, L: csc_matrix, f: Callable[[np.ndarray], np.ndarray], order: int, **kwargs) -> 'ChebyKernel':
        """
        Creates a ChebyKernel by fitting a function to a graph's spectrum.

        This is a convenience method that automatically estimates the spectral
        bound (`lambda_max`) of the graph Laplacian `L` before fitting.

        Parameters
        ----------
        L : csc_matrix
            The Graph Laplacian.
        f : Callable[[np.ndarray], np.ndarray]
            The vectorized function to approximate.
        order : int
            Order of the Chebyshev polynomial to fit.
        **kwargs
            Additional arguments passed to `ChebyKernel.from_function`.

        Returns
        -------
        ChebyKernel
            A new instance with the fitted coefficients.
        """
        spectrum_bound = estimate_spectral_bound(L)
        return cls.from_function(f, order, spectrum_bound, **kwargs)

    def _scale_x(self, x: np.ndarray) -> np.ndarray:
        """Maps points from [0, spectrum_bound] to the Chebyshev domain [-1, 1]."""
        return (2.0 / self.spectrum_bound) * x - 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the Chebyshev approximation for this kernel.

        Parameters
        ----------
        x : np.ndarray
            Points in the domain [0, spectrum_bound] to evaluate.

        Returns
        -------
        np.ndarray
            Evaluated function values at points in `x`.
        """
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

def _txt_loader(path: str) -> np.ndarray:
    """Loads a space-separated text file into a NumPy array."""
    return np.loadtxt(path, skiprows=1)

# Factory helpers
def _lap(k: str, r: str) -> csc_matrix: return _load_resource(f"library/{k}/{r}_{k}.mat", lambda p: _mat_loader(p, to_csc=True)) # type: ignore
def _sig(k: str, r: str) -> np.ndarray: return _load_resource(f"library/SIGNALS/{r}_{k}.mat", _mat_loader) # type: ignore
def _kern(n: str) -> Dict[str, Any]:   return _load_resource(f"library/KERNELS/{n}.json", _json_kern_loader)
def _mesh_lap(n: str) -> csc_matrix: return _load_resource(f"library/MESH/{n}.mat", lambda p: _mat_loader(p, to_csc=True)) # type: ignore
def _mesh_sig(n: str) -> np.ndarray: return _load_resource(f"library/SIGNALS/{n}_XYZ.txt", _txt_loader) # type: ignore

# Lazy loading registry
_LAZY_REGISTRY = {
    # Kernels
    "MEXICAN_HAT":     lambda: _kern("MEXICAN_HAT"),
    "GAUSSIAN_WAV":    lambda: _kern("GAUSSIAN_WAV"),
    "MODIFIED_MORLET": lambda: _kern("MODIFIED_MORLET"),
    "SHANNON":         lambda: _kern("SHANNON"),

    # Laplacians
    "DELAY_EASTWEST":  lambda: _lap("DELAY", "EASTWEST"),
    "DELAY_HAWAII":    lambda: _lap("DELAY", "HAWAII"),
    "DELAY_TEXAS":     lambda: _lap("DELAY", "TEXAS"),
    "DELAY_USA":       lambda: _lap("DELAY", "USA"),
    "DELAY_WECC":      lambda: _lap("DELAY", "WECC"),

    "IMPEDANCE_EASTWEST": lambda: _lap("IMPEDANCE", "EASTWEST"),
    "IMPEDANCE_HAWAII":   lambda: _lap("IMPEDANCE", "HAWAII"),
    "IMPEDANCE_TEXAS":    lambda: _lap("IMPEDANCE", "TEXAS"),
    "IMPEDANCE_USA":      lambda: _lap("IMPEDANCE", "USA"),
    "IMPEDANCE_WECC":     lambda: _lap("IMPEDANCE", "WECC"),

    "LENGTH_EASTWEST": lambda: _lap("LENGTH", "EASTWEST"),
    "LENGTH_HAWAII":   lambda: _lap("LENGTH", "HAWAII"),
    "LENGTH_TEXAS":    lambda: _lap("LENGTH", "TEXAS"),
    "LENGTH_USA":      lambda: _lap("LENGTH", "USA"),
    "LENGTH_WECC":     lambda: _lap("LENGTH", "WECC"),

    # Mesh Laplacians
    "MESH_BUNNY":      lambda: _mesh_lap("BUNNY"),
    "MESH_HORSE":      lambda: _mesh_lap("HORSE"),
    "MESH_LBRAIN":     lambda: _mesh_lap("LBRAIN"),


    # Signals
    "COORD_EASTWEST":  lambda: _sig("COORDS", "EASTWEST"),
    "COORD_HAWAII":    lambda: _sig("COORDS", "HAWAII"),
    "COORD_TEXAS":     lambda: _sig("COORDS", "TEXAS"),
    "COORD_USA":       lambda: _sig("COORDS", "USA"),

    # Mesh Signals
    "BUNNY_XYZ":       lambda: _mesh_sig("BUNNY"),
    "HORSE_XYZ":       lambda: _mesh_sig("HORSE"),
    "LBRAIN_XYZ":      lambda: _mesh_sig("LBRAIN"),
}

def __getattr__(name: str) -> Any:
    if name in _LAZY_REGISTRY:
        return _LAZY_REGISTRY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    return list(globals().keys()) + list(_LAZY_REGISTRY.keys())

__all__ = list(_LAZY_REGISTRY.keys()) + ["ChebyKernel", "VFKernel", "impulse", "get_cholmod_dll", "get_klu_dll", "estimate_spectral_bound"]