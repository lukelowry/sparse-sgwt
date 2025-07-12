# Example to demonstrate how to design the SGWT kernel

from ..src.kernel.main import KernelDesign

# Create and Save Graph Laplacian
# L = wb.length_laplacian()*(1/2)
# save_npz('LAP80.npz', csc_matrix(L))

# SGWT Design and Callable SGWT object
kern = KernelDesign(
    spectrum_range = (1e-7, 1e1),
    scale_range    = (1e2, 5e5), # (5e3, 1e5),
    pole_min       = 1e-6,
    nscales = 65,
    npoles  = 25,
    nsamples = 400,
    order = 1
)