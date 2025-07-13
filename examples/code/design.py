"""
design.py

An example to demonstrate how to design the SGWT kernel.

Author: Luke Lowery (lukel@tamu.edu)
"""

from sgwt.kernel import KernelDesign

kern = KernelDesign(
    spectrum_range = (1e-7, 1e1),
    scale_range    = (1e2, 5e5), # (5e3, 1e5),
    pole_min       = 1e-6,
    nscales = 65,
    npoles  = 25,
    nsamples = 400,
    order = 1
)

kern.write('kernel_model')