from abc import ABC, abstractmethod

class AnalyticFilters(ABC):

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