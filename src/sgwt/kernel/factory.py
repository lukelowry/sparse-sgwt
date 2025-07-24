

'''
Factory classes to generate numerical models
of the SGWT kernel.

Only used when designing a kernel.

'''


# This is the main class the user interacts with.
from .kernel import AbstractKernel
from .data import VFKernelData
from numpy import log, geomspace
from scipy.linalg import pinv

# This is the native vector fitting tool.
# No pole allocation in this version.

class WaveletFitting:
    '''
    Native vector fitting tool.

    Determines the residues and poles of a discrete
    set of frequnecy-domain wavelet kernels
    '''

    def __init__(self, domain, samples, initial_poles):
        '''
        Parameters:
            domain: (log spaced) sample points of signal
            samples: (on domain) kernel values, each col is different scale
            initial poles: (log spaced) initial pole locations
        '''

        # location, samples of VF, and initial poles
        self.x = domain
        self.G = samples # scale x lambda
        self.Q0 = initial_poles

    def eval_pole_matrix(self, Q, x):
        '''
        Description:
            Evaluates the 'pole matrix' over some domain x given poles Q
        Parameters:
            Q: Poles array (npoles x 1)
            x: domain to evaluate (nsamp x 1)
        Returns:
            Pole Matrix: shape is  (nsamps x npoles)
        '''
        return 1/(x + Q.T)
    
    def calc_residues(self, V, G):
        '''
        Description:
            Solves least square problem for residues for given set of poles
        Parameters:
            V: 'pole matrix' (use eval_pole_matrix)
            G: function being approximated
        Returns:
            Residue Matrix: shape is  (npoles x nscales)
        '''
        # Solve Equation: V@R = G
        return pinv(V)@G
    
    def fit(self):
        '''
        Description:
            Performs VF procedure on signal G.
        Returns:
            R, Q: shape is  (npoles x nscales), (npoles x 1)
        '''
        
        # (samples x poles)
        self.V = self.eval_pole_matrix(self.Q0, self.x)

        # (pole x scale)
        R = self.calc_residues(self.V, self.G)

        # TODO pole relalocation step here and iterative
        Q = self.Q0

        return R, Q
    

class KernelFactory:
    ''' 
    Class holding the spectral form of the wavelet function
    '''

    def __init__(
            self, 
            spectrum_range = (1e-7, 1e2),
            scale_range    = (1e-2, 1e5),
            nscales        = 10, 
            nsamples       = 300
        ):

        
        # Scales and Domain Vectors
        self.scales = self.logsamp(*scale_range   , nscales ) 
        self.domain = self.logsamp(*spectrum_range, nsamples)  

        # Calculate the interval of scales on log scale
        self.ds = log(self.scales[1]/self.scales[0])[0]

        # Meta Information
        self.nscales  = nscales 
        self.nsamples = nsamples
        self.spectrum_range = spectrum_range


    # TODO REMOVE
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
    
    # TODO REMOVE
    def get_approx(self):

        V, R = self.wf.V, self.kern.R

        return V@R
    
    def makeVF(
            self,
            kernfuncs:     AbstractKernel,
            pole_min: float          = 1e-5,
            npoles:   int            = 10
        ):

        # Extract Variables
        x = self.domain 
        s = self.scales 

        # Initial Poles
        Q0 = self.logsamp(
            start = pole_min,
            end   = self.spectrum_range[1], 
            N     = npoles
        ) 

        # Sample the function for all scales (nScales x lambda)
        G = kernfuncs.g(x*s.T)

        # Wavelet Fitting object
        wf = WaveletFitting(
            domain        = x, 
            samples       = G, 
            initial_poles = Q0
        )

        # Fit and return pole and residues of apporimation
        R, Q = wf.fit()


        # VF Kernel Dataclass
        kern = VFKernelData(
            R = R,
            Q = Q,
            S = s
        )
        
        # Useful for debugging
        self.G    = G
        self.wf   = wf
        self.kern = kern 

        return kern
  