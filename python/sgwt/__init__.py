"""
sgwt

Author: Luke Lowery (lukel@tamu.edu)

"""

from .library import *
from .fitted import VFKern

# Static Graphs (Typical use case)
from .static import Convolve, impulse

# Dynamic Graphs (Optimized performance, less versatile)
from .dynamic import DyConvolve