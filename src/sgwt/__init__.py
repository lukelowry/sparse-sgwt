"""
sgwt

Author: Luke Lowery (lukel@tamu.edu)


- Analytical
    - CHOLMOD (scikit)
    - CHOLMOD (DLL)
- Vector Fit
    - CHOLMOD (scikit)
    - CHOLMOD (DLL)
- Chebyshev Fit
    - scipy.sparse

"""



from .analytic import FiltersScikit, FiltersDLL
from .data import *