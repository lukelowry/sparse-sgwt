from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass(frozen=True)
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
    def from_json(cls, data: dict) -> 'VFKern':
        """Loads kernel data from a dictionary/JSON structure."""
        poles = data.get('poles', [])
        return cls(
            R=np.array([p['r'] for p in poles]),
            Q=np.array([p['q'] for p in poles]),
            D=np.array(data.get('d', []))
        )