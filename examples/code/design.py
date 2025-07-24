"""
design.py

An example to demonstrate how to design the SGWT kernel.

Author: Luke Lowery (lukel@tamu.edu)
"""

from sgwt.kernel import KernelFactory, KernelSmoothRational

# General Kernel Specifications
factory = KernelFactory(
    spectrum_range = (1e-8, 1e3),
    scale_range    = (1e4, 3e5),#(1e2, 5e5), # (5e3, 1e5),
    nscales = 65,
    nsamples = 800,
)

# VF Based Kernel Model
kern = factory.makeVF(
    kernfuncs = KernelSmoothRational(),
    pole_min  = 1e-8,
    npoles    = 35 # 45
)

# Write
FNAME = r'C:\Users\wyattluke.lowery\Documents\GitHub\sparse-sgwt\examples\kernels\kernel_model'
kern.to_file(FNAME)