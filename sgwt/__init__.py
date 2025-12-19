"""
sgwt

Author: Luke Lowery (lukel@tamu.edu)


Analytical
    - CHOLMOD (scikit)
    - CHOLMOD (DLL)
Vector Fit
    - CHOLMOD (scikit)
    - CHOLMOD (DLL)

"""

from .analytic import FiltersScikit, Filters
from .fitted import VFConvolve, VFConvolveScikit, VFKern
from .data import *