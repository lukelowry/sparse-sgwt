"""General Utilities

Description: Utilities for accessing built-in data, VFKern, and impulse helper function.

Author: Luke Lowery (lukel@tamu.edu)
"""

import sys
import os

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files

from ctypes import CDLL
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix

from json import load as jsonload
from typing import Any, Callable, Dict, Union

@dataclass
class VFKern:
    """Vector Fitting Kernel representation.

    A dataclass to store the components of a rational kernel approximation
    obtained from Vector Fitting.

    Attributes
    ----------
    R : np.ndarray
        Residue matrix of shape (n_poles, n_dims).
    Q : np.ndarray
        Poles vector of shape (n_poles,).
    D : np.ndarray
        Direct term (offset) of shape (n_dims,).
    """
    R: np.ndarray
    Q: np.ndarray
    D: np.ndarray

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VFKern':
        """Loads kernel data from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing the kernel parameters, typically loaded
            from a JSON file. It should have 'poles' and 'd' keys.

        Returns
        -------
        VFKern
            A new instance of the VFKern class.
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

def get_cholmod_dll() -> CDLL:
    """Locates and loads the CHOLMOD shared library.

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
        The loaded CHOLMOD DLL object.
    """

    resource = files("sgwt") / "library" / "dll" / "cholmod.dll"

    with as_file(resource) as dll_path:
        dll_dir = os.path.dirname(dll_path)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        else:
            os.environ['PATH'] = str(dll_dir) + os.pathsep + os.environ['PATH']

        try:
            return CDLL(str(dll_path))
        except OSError as e:
            raise OSError(f"Failed to load DLL at {dll_path}. Error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading DLL: {e}")


def _load_resource(path: str, loader: Callable[[str], Any]) -> Any:
    """Centralized resource loader using importlib.resources."""
    with as_file(files("sgwt").joinpath(path)) as file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resource not found: {file_path}")
        return loader(str(file_path))


def _mat_loader(path: str, to_csc: bool = False) -> Union[np.ndarray, csc_matrix]:
    """Loads the first data variable from a .mat file."""
    data = loadmat(path, squeeze_me=False)
    keys = [k for k in data if not k.startswith("__")]
    
    if not keys:
        raise ValueError(f"No data variables found in MAT file: {path}")

    if to_csc:
        return csc_matrix(data[keys[0]])

    if len(keys) > 1:
        return np.stack([data[k].flatten() for k in keys], axis=1)

    res = data[keys[0]]
    if res.ndim == 2 and res.shape[0] == 1 and res.shape[1] > 1:
        return res.T
    return res


def _json_kern_loader(path: str) -> Dict[str, Any]:
    """Loads a VFKern from a JSON file."""
    with open(path, "r") as f:
        return jsonload(f)

# Factory helpers
def _lap(k: str, r: str) -> csc_matrix: return _load_resource(f"library/{k}/{r}_{k}.mat", lambda p: _mat_loader(p, to_csc=True)) # type: ignore
def _sig(k: str, r: str) -> np.ndarray: return _load_resource(f"library/SIGNALS/{r}_{k}.mat", _mat_loader) # type: ignore
def _kern(n: str) -> Dict[str, Any]:   return _load_resource(f"library/KERNELS/{n}.json", _json_kern_loader)

# Kernels
MEXICAN_HAT     = _kern("MEXICAN_HAT")
GAUSSIAN_WAV    = _kern("GAUSSIAN_WAV")
MODIFIED_MORLET = _kern("MODIFIED_MORLET")
SHANNON         = _kern("SHANNON")

# Laplacians
DELAY_EASTWEST = _lap("DELAY", "EASTWEST")
DELAY_HAWAII   = _lap("DELAY", "HAWAII")
DELAY_TEXAS    = _lap("DELAY", "TEXAS")
DELAY_USA      = _lap("DELAY", "USA")
DELAY_WECC     = _lap("DELAY", "WECC")

IMPEDANCE_EASTWEST = _lap("IMPEDANCE", "EASTWEST")
IMPEDANCE_HAWAII   = _lap("IMPEDANCE", "HAWAII")
IMPEDANCE_TEXAS    = _lap("IMPEDANCE", "TEXAS")
IMPEDANCE_USA      = _lap("IMPEDANCE", "USA")
IMPEDANCE_WECC     = _lap("IMPEDANCE", "WECC")

LENGTH_EASTWEST = _lap("LENGTH", "EASTWEST")
LENGTH_HAWAII   = _lap("LENGTH", "HAWAII")
LENGTH_TEXAS    = _lap("LENGTH", "TEXAS")
LENGTH_USA      = _lap("LENGTH", "USA")
LENGTH_WECC     = _lap("LENGTH", "WECC")

# Signals
COORD_EASTWEST = _sig("COORDS", "EASTWEST")
COORD_HAWAII   = _sig("COORDS", "HAWAII")
COORD_TEXAS    = _sig("COORDS", "TEXAS")
COORD_USA      = _sig("COORDS", "USA")