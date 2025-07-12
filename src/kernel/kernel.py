from abc import ABC, abstractmethod


# An Abstract SGWT Kernel incase multiple version of the transformation are implemented

class AbstractKernel:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def h(self, x):
        '''
        Description:
            The scaling kerenl h(x) evaluating the 'DC-like' spectrum
        Parameters:
            Vector x, the spectrum domain to evaluate.
        Returns:
            Spectral domain scaling kerenel
        '''

    @abstractmethod
    def g(self, x):
        '''
        Description:
            The wavelet generating kerenl g(x) evaluating the un-scaled wavelet
        Parameters:
            Vector x, the spectrum domain to evaluate.
        Returns:
            Spectral domain wavelet kerenel
        '''



