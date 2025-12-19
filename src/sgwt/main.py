"""
main.py

Will be abstracted eventually. Core class for now, implementing
the VF version of the SGWT.

Author: Luke Lowery (lukel@tamu.edu)
"""

from abc import ABC, abstractmethod

class FastSGWT(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def scaling_coeffs(self, f, scales):
        pass

    @abstractmethod
    def wavelet_coeffs(self, f, scales):
        pass

    @abstractmethod
    def highpass_coeffs(self, f, scales):
        pass