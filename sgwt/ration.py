# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/ration.py
Description: Data structures for Vector Fitting (VF) kernel representations.
"""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class VFKern:
    """
    Vector Fitting Kernel representation.
    R: Residual Matrix (nPoles x nScales)
    Q: Poles Vector (nPoles x 1)
    D: Offset (nDim x 1)
    """
    R: npt.NDArray
    Q: npt.NDArray
    D: npt.NDArray

    @classmethod
    def from_dict(cls, data: dict) -> 'VFKern':
        """Loads kernel data from a dictionary/JSON structure."""
        poles = data.get('poles', [])
        return cls(
            R=np.array([p['r'] for p in poles]),
            Q=np.array([p['q'] for p in poles]),
            D=np.array(data.get('d', []))
        )