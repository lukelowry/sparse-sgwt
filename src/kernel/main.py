
# This is the main class the user interacts with.
from kernel import AbstractKernel
from fitting import WaveletFitting
from numpy import log, savez, geomspace

class KernelDesign(AbstractKernel):
    ''' 
    Class holding the spectral form of the wavelet function
    '''

    def __init__(
            self, 
            spectrum_range = (1e-7, 1e2),
            scale_range    = (1e-2, 1e5),
            pole_min       = 1e-5,
            nscales        = 10, 
            npoles         = 10, 
            nsamples       = 300,
            order          = 2 
        ):

        # Scales, Domain, and initial poles
        s  = self.logsamp(*scale_range   , nscales )   
        x  = self.logsamp(*spectrum_range, nsamples)  
        Q0 = self.logsamp(pole_min,spectrum_range[1], npoles  )  

        # Sample the function for all scales
        # (Scales x lambda)
        G = self.g(x*s.T, order=order)

        wf = WaveletFitting(
            domain        = x, 
            samples       = G, 
            initial_poles = Q0
        )

        # Fit and return pole and residues of apporimation
        R, Q = wf.fit()

        # Calculate the interval of scales on log scale
        self.ds = log(s[1]/s[0])[0]

        # Assign Poles, residues, scales
        self.wf  = wf
        self.__q = Q
        self.__r = R
        self.__s = s
        self.nscales  = nscales 
        self.npoles   = npoles
        self.nsamples = nsamples

        # Useful for debugging
        self.x = x
        self.G = G
        
    def logsamp(self, start, end, N=5):
        '''
        Description:
            Helper sampling function for log scales
        Parameters:
            start: first value
            end: last value
            N: number of log-spaced values between start and end
        Returns:
            Samples array: shape is  (N x 1)
        '''
        return geomspace(start, [end],N)
    
    def g(self, x, order=1):
        '''
        Description:
            Default kernel function evaluator
        Parameters:
            x: domain to evaluate (array)
            order: higher order -> narrower bandwidth
        Returns:
            g(x): same shape as x
        '''
        f = 2*x/(1+x**2)
        return f**order
    
    def h(self, x):
        '''
        Description:
            The scaling kerenl h(x) evaluating the 'DC-like' spectrum
        Parameters:
            x: domain to evaluate (array)
        Returns:
           h(x): same shape as x
        '''
        f = 1/(1+x**2)
        return f
    
    def get_approx(self):

        V, R = self.wf.V, self.R

        return V@R

    @property
    def R(self):
        '''Residue Matrix where each column is a scale
        and each row corresponds to a pole '''
        return self.__r

    @property
    def Q(self):
        '''Vector of Poles'''
        return self.__q
    
    @property
    def scales(self):
        '''Vector of Scales'''
        return self.__s
    
    def write(self, fname):
        '''
        Description:
            Writes poles & residue model to npz file format.
            Post-fix .npz not needed, just write desired name.
        Parameters:
            fname: Filename/directory if needed
        Returns:
            None
        '''
        savez(f'{fname}.npz', R=self.R, Q=self.Q, S=self.scales)
