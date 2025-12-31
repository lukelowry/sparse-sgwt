'''
I/O Utilities
Access to laplacians, signals, and kernels.
'''

from .ration import VFKern

import numpy as np
import os
from ctypes import CDLL
from importlib.resources import as_file, files
from json import load as jsonload
from typing import Any, Callable

from scipy.io import loadmat
from scipy.sparse import csc_matrix


def get_cholmod_dll():
    '''
    Adds the library/dll directory with necessary DLLs to the system search path
    '''
    resource = files("sgwt.library") / "dll" / "cholmod.dll"

    with as_file(resource) as dll_path:
        dll_dir = os.path.dirname(dll_path)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        try:
            return CDLL(str(dll_path))
        except:
            raise Exception("Error Loading DLL")


def _load_resource(path: str, loader: Callable[[str], Any]):
    """Centralized resource loader using importlib.resources."""
    with as_file(files("sgwt").joinpath(path)) as file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resource not found: {file_path}")
        return loader(str(file_path))


def _mat_loader(path: str, to_csc: bool = False):
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


def _json_kern_loader(path: str):
    """Loads a VFKern from a JSON file."""
    with open(path, "r") as f:
        return VFKern.from_json(jsonload(f))

# Factory helpers
def _lap(k, r): return _load_resource(f"library/{k}/{r}_{k}.mat", lambda p: _mat_loader(p, to_csc=True))
def _sig(k, r): return _load_resource(f"library/SIGNALS/{r}_{k}.mat", _mat_loader)
def _kern(n):   return _load_resource(f"library/KERNELS/{n}.json", _json_kern_loader)

''' KERNELS '''
MEXICAN_HAT     = _kern("MEXICAN_HAT")
GAUSSIAN_WAV    = _kern("GAUSSIAN_WAV")
MODIFIED_MORLET = _kern("MODIFIED_MORLET")
SHANNON         = _kern("SHANNON")

''' LAPLACIANS '''
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

''' SIGNALS '''
COORD_EASTWEST = _sig("COORDS", "EASTWEST")
COORD_HAWAII   = _sig("COORDS", "HAWAII")
COORD_TEXAS    = _sig("COORDS", "TEXAS")
COORD_USA      = _sig("COORDS", "USA")