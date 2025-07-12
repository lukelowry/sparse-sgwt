
from scipy.linalg import pinv

# This is the native vector fitting tool.
# No pole allocation in this version.

class WaveletFitting:
    '''A Helper class that determines the residues and poles of a discrete
    set of frequnecy-domain wavelet kernels'''

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
    