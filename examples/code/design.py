from sgwt.kernel import KernelDesign
from scipy.sparse import csc_matrix, save_npz

# Example to demonstrate how to design the SGWT kernel

# TODO Re-render laplacians to be certain constructed as intended

# Design the Kernel
kern = KernelDesign(
    spectrum_range = (1e-7, 1e1),
    scale_range    = (1e2, 5e5), # (5e3, 1e5),
    pole_min       = 1e-6,
    nscales = 65,
    npoles  = 25,
    nsamples = 400,
    order = 1
)

# Save the Kernel
kern.write('kernel_model')